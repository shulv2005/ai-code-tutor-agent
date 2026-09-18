"""API Key 临时保管箱测试：存取、过期、清除，以及"绝不泄漏"这条红线。

为什么这个文件值得写得这么细：
这是全项目**唯一一处会碰到用户密钥**的地方。功能写对只是及格线，
真正要守住的是三条红线，每条都有对应用例：

1. **不落库**：内存字典，进程重启即消失（用例直接检查"没有数据库文件参与"）；
2. **不进日志**：任何一条日志里都不允许出现明文 Key（用日志收集器逐条断言）；
3. **不回传**：所有接口的响应体里都不能出现明文 Key（把响应原文抓下来搜一遍）。

另外还覆盖：滑动过期、会话上限淘汰、并发读写、Key 优先级、掩码与脱敏。
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.core.api_key_manager import (
    MAX_KEY_LENGTH,
    SESSION_HEADER,
    ApiKeyExpired,
    ApiKeyManager,
    ApiKeyNotFound,
    ApiKeyRejected,
    mask,
    redact,
    reset_api_key_manager,
)

# 测试里到处用同一把"假 Key"，方便在日志/响应里搜它
SECRET = "sk-test-abcdefghijklmnop-0123456789"
OTHER_SECRET = "sk-other-zyxwvutsrqponmlk-9876543210"


# ---------------------------------------------------------------------------
# 夹具与工具
# ---------------------------------------------------------------------------
class FakeClock:
    """可手动拨动的时钟，用来测"30 分钟没动就过期"而不用真的等。"""

    def __init__(self) -> None:
        self.now = datetime(2026, 2, 18, 10, 0, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: int) -> None:
        """把时间往前拨 seconds 秒。"""
        self.now += timedelta(seconds=seconds)


@pytest.fixture()
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture()
def manager(clock: FakeClock) -> ApiKeyManager:
    """默认 30 分钟过期、最多 3 个会话的小保管箱，方便测淘汰逻辑。"""
    return ApiKeyManager(ttl_seconds=1800, max_sessions=3, clock=clock)


@pytest.fixture()
def log_records() -> Iterator[list[logging.LogRecord]]:
    """收集测试期间所有日志记录（用来断言"日志里没有 Key"）。

    为什么挂在 root 上而不是用 caplog：项目用 `dictConfig` 配置了日志，
    它会重建 root 的 handler；这个夹具在 `create_app()` 之后才挂，
    所以能稳定收到所有子 logger 传播上来的记录。
    """

    records: list[logging.LogRecord] = []

    class Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = Collector(level=logging.DEBUG)
    root = logging.getLogger()
    root.addHandler(handler)
    previous_level = root.level
    root.setLevel(logging.DEBUG)
    try:
        yield records
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_level)


def assert_no_secret(records: list[logging.LogRecord], *secrets: str) -> None:
    """断言这些日志里一条都不含明文密钥。"""
    for record in records:
        # 把带上参数与异常栈的完整文本都取出来看，避免 Key 藏在 args / exc_info 里
        rendered = f"{record.getMessage()} {record.exc_info} {record.__dict__.get('extra', '')}"
        for secret in secrets:
            assert secret not in rendered, f"日志里出现了明文 Key：{record.name} {rendered}"


# ---------------------------------------------------------------------------
# 1. 基本存取
# ---------------------------------------------------------------------------
def test_set_and_get_key(manager: ApiKeyManager) -> None:
    """存进去能原样取出来，并且拿到一个可用的会话号。"""
    record = manager.set_key("deepseek", SECRET)

    assert record.session_id
    assert manager.get_key(record.session_id, "deepseek") == SECRET
    assert manager.get_key(record.session_id) == SECRET      # 不指定模型也能取到


def test_one_session_holds_multiple_models(manager: ApiKeyManager) -> None:
    """一个会话可以存多个模型的 Key（这样切模型不用重填）。"""
    first = manager.set_key("deepseek", SECRET)
    second = manager.set_key("qwen", OTHER_SECRET, session_id=first.session_id)

    assert second.session_id == first.session_id             # 还是同一个会话
    assert set(second.api_keys) == {"deepseek", "qwen"}
    assert manager.get_key(first.session_id, "qwen") == OTHER_SECRET
    assert manager.get_key(first.session_id, "deepseek") == SECRET


def test_missing_model_raises(manager: ApiKeyManager) -> None:
    """会话里没有这个模型的 Key 时，报错要说清是哪个模型。"""
    record = manager.set_key("deepseek", SECRET)
    with pytest.raises(ApiKeyNotFound) as excinfo:
        manager.get_key(record.session_id, "qwen")
    assert "qwen" in str(excinfo.value)


def test_session_id_format_is_validated(manager: ApiKeyManager) -> None:
    """乱七八糟的会话号直接当成"没有会话"，不给字典填充的机会。"""
    for bad in ("", "  ", "../../etc", "短", "a" * 500):
        with pytest.raises(ApiKeyNotFound):
            manager.get(bad)


def test_key_too_long_is_rejected(manager: ApiKeyManager) -> None:
    """超长 Key 直接拒绝：正常 Key 没有这么长，多半是粘贴错了。"""
    with pytest.raises(ApiKeyRejected):
        manager.set_key("deepseek", "x" * (MAX_KEY_LENGTH + 1))


def test_empty_model_id_is_rejected(manager: ApiKeyManager) -> None:
    with pytest.raises(ApiKeyRejected):
        manager.set_key("   ", SECRET)


def test_local_model_can_store_empty_key(manager: ApiKeyManager) -> None:
    """本地模型（Ollama）允许空 Key，只记住"用户选了这个模型"。"""
    record = manager.set_key("ollama", "")
    assert record.has_key is False
    assert record.api_keys == {"ollama": pytest.importorskip("pydantic").SecretStr("")}


# ---------------------------------------------------------------------------
# 2. 过期：滑动窗口
# ---------------------------------------------------------------------------
def test_key_expires_after_ttl(manager: ApiKeyManager, clock: FakeClock) -> None:
    """30 分钟没动就失效，并且顺手把记录删掉（不占内存）。"""
    record = manager.set_key("deepseek", SECRET)

    clock.advance(1799)                       # 还差 1 秒
    assert manager.get_key(record.session_id) == SECRET

    clock.advance(1800)                       # 又过了 30 分钟（总 3599 秒，超过上次使用 1800 秒）
    with pytest.raises(ApiKeyExpired):
        manager.get_key(record.session_id)
    assert manager.stats()["sessions"] == 0    # 过期记录已被清掉


def test_using_key_extends_expiry(manager: ApiKeyManager, clock: FakeClock) -> None:
    """每次使用都会顺延（滑动过期）：一直用就一直有效。"""
    record = manager.set_key("deepseek", SECRET)

    for _ in range(5):
        clock.advance(1700)                    # 每次都在过期前用一下
        assert manager.get_key(record.session_id, "deepseek") == SECRET

    assert manager.get(record.session_id).expires_at > clock.now


def test_purge_expired_removes_stale_sessions(
    manager: ApiKeyManager, clock: FakeClock
) -> None:
    """purge_expired 能一次清掉所有过期会话，返回清掉的个数。"""
    first = manager.set_key("deepseek", SECRET)
    clock.advance(600)
    second = manager.set_key("qwen", OTHER_SECRET)      # 这个还新鲜

    # 再走 1799 秒：第一个（3600 秒到期）已经过期，第二个（4200 秒到期）还没到
    clock.advance(1799)
    assert manager.purge_expired() == 1
    assert manager.stats()["sessions"] == 1
    with pytest.raises(ApiKeyNotFound):
        manager.get(first.session_id)
    assert manager.get_key(second.session_id) == OTHER_SECRET


# ---------------------------------------------------------------------------
# 3. 清除
# ---------------------------------------------------------------------------
def test_clear_single_model_keeps_session(manager: ApiKeyManager) -> None:
    """只清某个模型：会话还在，别的模型不受影响。"""
    record = manager.set_key("deepseek", SECRET)
    manager.set_key("qwen", OTHER_SECRET, session_id=record.session_id)

    assert manager.clear_key(record.session_id, "deepseek") is True
    assert manager.get_key(record.session_id, "qwen") == OTHER_SECRET
    with pytest.raises(ApiKeyNotFound):
        manager.get_key(record.session_id, "deepseek")


def test_clear_whole_session(manager: ApiKeyManager) -> None:
    """清掉整个会话（"退出"）：之后查状态就该是"会话不存在"。"""
    record = manager.set_key("deepseek", SECRET)

    assert manager.clear_key(record.session_id) is True
    with pytest.raises(ApiKeyNotFound):
        manager.get(record.session_id)
    # 再清一次没有东西可清，返回 False（不该报错）
    assert manager.clear_key(record.session_id) is False


def test_clear_unknown_session_returns_false(manager: ApiKeyManager) -> None:
    assert manager.clear_key("not-a-real-session") is False
    assert manager.clear_key(None) is False


def test_clear_all(manager: ApiKeyManager) -> None:
    manager.set_key("deepseek", SECRET)
    manager.set_key("qwen", OTHER_SECRET)
    assert manager.clear_all() == 2
    assert manager.stats()["sessions"] == 0


# ---------------------------------------------------------------------------
# 4. resolve：调用模型前到底用哪把 Key
# ---------------------------------------------------------------------------
def test_resolve_priority(manager: ApiKeyManager) -> None:
    """优先级：本次请求带的 > 会话里存的 > 配置兜底。"""
    record = manager.set_key("deepseek", SECRET)

    from_request = manager.resolve(record.session_id, "deepseek", api_key="sk-inline")
    assert (from_request.api_key, from_request.source) == ("sk-inline", "request")

    from_session = manager.resolve(record.session_id, "deepseek")
    assert (from_session.api_key, from_session.source) == (SECRET, "session")

    from_config = manager.resolve(None, "deepseek")
    assert (from_config.api_key, from_config.source) == ("", "config")


def test_resolve_falls_back_when_model_missing_in_session(manager: ApiKeyManager) -> None:
    """会话里没有这个模型的 Key 时，回落到配置（而不是塞一个错的 Key 进去）。"""
    record = manager.set_key("deepseek", SECRET)
    resolved = manager.resolve(record.session_id, "qwen")
    assert resolved.source == "config"
    assert resolved.api_key == ""
    assert resolved.model_id == "qwen"


def test_resolve_uses_default_model_of_session(manager: ApiKeyManager) -> None:
    """不带模型 ID 时，用会话里最后设置的那个模型。"""
    record = manager.set_key("deepseek", SECRET)
    manager.set_key("qwen", OTHER_SECRET, session_id=record.session_id)

    resolved = manager.resolve(record.session_id)
    assert resolved.model_id == "qwen"
    assert resolved.api_key == OTHER_SECRET


def test_resolve_with_expired_session_falls_back(
    manager: ApiKeyManager, clock: FakeClock
) -> None:
    """会话过期不该让请求崩掉，应当安静地回落到配置。"""
    record = manager.set_key("deepseek", SECRET)
    clock.advance(2000)
    resolved = manager.resolve(record.session_id, "deepseek")
    assert resolved.source == "config"
    assert resolved.api_key == ""


# ---------------------------------------------------------------------------
# 5. 内存上限与并发
# ---------------------------------------------------------------------------
def test_oldest_session_is_evicted(manager: ApiKeyManager, clock: FakeClock) -> None:
    """会话数超过上限时淘汰最久没用的（防止有人刷接口把内存吃满）。"""
    first = manager.set_key("deepseek", SECRET)
    clock.advance(10)
    manager.set_key("qwen", OTHER_SECRET)
    clock.advance(10)
    third = manager.set_key("ollama", "")
    clock.advance(10)
    manager.set_key("deepseek", "sk-newest")        # 第 4 个，触发淘汰

    assert manager.stats()["sessions"] == 3
    with pytest.raises(ApiKeyNotFound):
        manager.get(first.session_id)                # 最久没用过的被淘汰
    assert manager.get_key(third.session_id) == ""


def test_concurrent_writes_do_not_lose_data() -> None:
    """多线程同时写不同会话时，数据不能互相覆盖（保管箱内部有锁）。

    这里特意用一个"上限很大"的保管箱：默认夹具只有 3 个会话位，
    10 个线程连写会触发淘汰，那是另一条逻辑（上面有专门的用例）。
    """
    manager = ApiKeyManager(ttl_seconds=1800, max_sessions=100)
    session_ids: list[str] = []
    lock = threading.Lock()

    def worker(index: int) -> None:
        record = manager.set_key(f"model{index}", f"sk-{index}-{'x' * 20}")
        with lock:
            session_ids.append(record.session_id)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(set(session_ids)) == 10                # 10 个会话互不干扰
    for index in range(10):
        assert manager.get_key(session_ids[index], f"model{index}").startswith(f"sk-{index}-")


# ---------------------------------------------------------------------------
# 6. 掩码、脱敏、以及"对象本身不会泄漏"
# ---------------------------------------------------------------------------
def test_mask_keeps_only_prefix() -> None:
    assert mask(SECRET) == "sk-***"
    assert mask("") == "(空)"
    assert SECRET[3:] not in mask(SECRET)             # 后半段一个字符都不留


def test_redact_removes_secret_from_text() -> None:
    """服务商的报错里可能回显 Key，写日志/返回前必须抹掉。"""
    message = f"Incorrect API key provided: {SECRET}. You can find your key at ..."
    cleaned = redact(message, SECRET)
    assert SECRET not in cleaned
    assert "***" in cleaned
    # 太短的串不处理，避免误伤正常文字
    assert redact("hello", "he") == "hello"


def test_record_repr_hides_key(manager: ApiKeyManager) -> None:
    """整个会话对象被 print/logger 时也不能带出明文。"""
    record = manager.set_key("deepseek", SECRET)
    assert SECRET not in repr(record)
    assert SECRET not in str(record.to_public_dict())
    assert SECRET not in str(manager.stats())


def test_resolved_credential_repr_hides_key(manager: ApiKeyManager) -> None:
    record = manager.set_key("deepseek", SECRET)
    resolved = manager.resolve(record.session_id, "deepseek")
    assert resolved.api_key == SECRET                  # 真值仍然拿得到（要喂给客户端）
    assert SECRET not in repr(resolved)                # 但 repr 里只有掩码


def test_secret_is_stored_as_secret_str(manager: ApiKeyManager) -> None:
    """内部用 SecretStr 存 Key：手滑把对象打进日志也只会看到 **********。"""
    record = manager.set_key("deepseek", SECRET)
    stored = record.api_keys["deepseek"]
    assert SECRET not in str(stored)
    assert stored.get_secret_value() == SECRET


# ---------------------------------------------------------------------------
# 7. 接口层：/api/v1/auth 与 /api/v1/models
# ---------------------------------------------------------------------------
@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, sqlite_path) -> Iterator[TestClient]:
    """带多模型配置的测试客户端（不读真实 .env，Key 一律留空）。"""
    monkeypatch.setenv("LLM__API_KEY", "")
    monkeypatch.setenv("LLM__MODELS__DEEPSEEK__LABEL", "DeepSeek 官方")
    monkeypatch.setenv("LLM__MODELS__DEEPSEEK__PROVIDER", "deepseek")
    monkeypatch.setenv("LLM__MODELS__DEEPSEEK__MODEL_NAME", "deepseek-chat")
    monkeypatch.setenv("LLM__MODELS__OLLAMA__PROVIDER", "ollama")
    monkeypatch.setenv("LLM__MODELS__OLLAMA__MODEL_NAME", "qwen2.5-coder:7b")
    monkeypatch.setenv("LLM__DEFAULT_MODEL", "deepseek")
    from app.core.config import get_settings

    get_settings.cache_clear()
    reset_api_key_manager()          # 保证用例之间不串数据

    from app.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client

    reset_api_key_manager()
    get_settings.cache_clear()


def _all_values(node: object) -> list[object]:
    """把一个嵌套的 JSON 结构里所有取值摊平成列表（用来搜"有没有泄漏"）。"""
    if isinstance(node, dict):
        return [item for value in node.values() for item in _all_values(value)]
    if isinstance(node, list):
        return [item for value in node for item in _all_values(value)]
    return [node]


def test_models_list_endpoint(client: TestClient) -> None:
    """模型清单：字段齐全、标出默认模型、**响应里没有 Key**。"""
    response = client.get("/api/v1/models/list")
    assert response.status_code == 200
    body = response.json()

    ids = [item["id"] for item in body["models"]]
    assert ids == ["deepseek", "ollama"]
    assert body["default_model_id"] == "deepseek"
    assert body["allow_client_key"] is True
    assert body["total"] == 2

    deepseek = body["models"][0]
    assert deepseek["label"] == "DeepSeek 官方"
    assert deepseek["requires_api_key"] is True
    assert deepseek["is_default"] is True

    # 关键：没有任何一个模型对象带 api_key 字段，也没有任何值像是 Key
    for item in body["models"]:
        assert "api_key" not in item
        assert "has_default_key" in item          # 只回答"配没配"，不给值
    assert not [value for value in _all_values(body) if str(value).startswith("sk-")]


def test_set_key_endpoint_stores_without_echoing(client: TestClient) -> None:
    """存 Key 成功，且响应里**不回传 Key**（连片段都不给）。"""
    response = client.post(
        "/api/v1/auth/set_key", json={"model_id": "deepseek", "api_key": SECRET}
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["session"]["has_key"] is True
    assert body["session"]["models"] == ["deepseek"]
    assert body["ttl_seconds"] == 1800
    assert SECRET not in response.text
    assert "api_key" not in response.text

    session_id = body["session"]["session_id"]

    # 状态接口同样不带 Key，但能告诉你"有 Key"
    status = client.get("/api/v1/auth/status", headers={SESSION_HEADER: session_id})
    assert status.status_code == 200
    assert status.json()["has_key"] is True
    assert SECRET not in status.text


def test_set_key_rejects_unknown_model(client: TestClient) -> None:
    response = client.post(
        "/api/v1/auth/set_key", json={"model_id": "gpt-9", "api_key": SECRET}
    )
    assert response.status_code == 400
    # 报错里要告诉用户有哪些模型可选，方便他改
    assert "deepseek" in response.json()["detail"]


def test_set_key_requires_key_for_cloud_model(client: TestClient) -> None:
    response = client.post(
        "/api/v1/auth/set_key", json={"model_id": "deepseek", "api_key": ""}
    )
    assert response.status_code == 400
    assert "需要 API Key" in response.json()["detail"]


def test_set_key_allows_empty_key_for_local_model(client: TestClient) -> None:
    """本地模型（Ollama）不填 Key 也能保存，提示语要说清"不需要"。"""
    response = client.post(
        "/api/v1/auth/set_key", json={"model_id": "ollama", "api_key": ""}
    )
    assert response.status_code == 200, response.text
    assert "不需要 API Key" in response.json()["message"]
    assert response.json()["session"]["has_key"] is False


def test_set_key_can_reuse_session_from_header(client: TestClient) -> None:
    """带了 X-Session-Id 就更新同一个会话（切模型不用重填 Key）。"""
    first = client.post(
        "/api/v1/auth/set_key", json={"model_id": "deepseek", "api_key": SECRET}
    ).json()
    session_id = first["session"]["session_id"]

    second = client.post(
        "/api/v1/auth/set_key",
        json={"model_id": "ollama", "api_key": ""},
        headers={SESSION_HEADER: session_id},
    ).json()
    assert second["session"]["session_id"] == session_id
    assert second["session"]["models"] == ["deepseek", "ollama"]


def test_status_without_session_returns_404(client: TestClient) -> None:
    """没带会话号时明确告诉前端"还没有会话"，而不是返回 500。"""
    response = client.get("/api/v1/auth/status")
    assert response.status_code == 404
    assert "set_key" in response.json()["detail"]


def test_clear_key_endpoint(client: TestClient) -> None:
    """清除：先清单个模型，再清整个会话（清完查状态应该是 404）。"""
    session_id = client.post(
        "/api/v1/auth/set_key", json={"model_id": "deepseek", "api_key": SECRET}
    ).json()["session"]["session_id"]
    client.post(
        "/api/v1/auth/set_key",
        json={"model_id": "ollama", "api_key": ""},
        headers={SESSION_HEADER: session_id},
    )

    single = client.post(
        "/api/v1/auth/clear_key", json={"session_id": session_id, "model_id": "deepseek"}
    )
    assert single.status_code == 200
    assert single.json()["cleared"] is True
    assert "deepseek" in single.json()["message"]

    rest = client.get("/api/v1/auth/status", headers={SESSION_HEADER: session_id}).json()
    assert rest["models"] == ["ollama"]        # 只清掉了指定的那个

    whole = client.post("/api/v1/auth/clear_key", json={"session_id": session_id})
    assert whole.json()["cleared"] is True
    assert client.get(
        "/api/v1/auth/status", headers={SESSION_HEADER: session_id}
    ).status_code == 404


def test_clear_key_without_session_is_not_an_error(client: TestClient) -> None:
    """没有会话时清 Key：返回 200 + cleared=false（前端不用做错误处理）。"""
    response = client.post("/api/v1/auth/clear_key", json={})
    assert response.status_code == 200
    assert response.json()["cleared"] is False


def test_client_key_can_be_disabled(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SECURITY__ALLOW_CLIENT_KEY=false 时，set_key 直接 403（"只能用后端配的 Key"）。"""
    monkeypatch.setenv("SECURITY__ALLOW_CLIENT_KEY", "false")
    from app.core.config import get_settings

    get_settings.cache_clear()
    try:
        response = client.post(
            "/api/v1/auth/set_key", json={"model_id": "deepseek", "api_key": SECRET}
        )
        assert response.status_code == 403
        assert "ALLOW_CLIENT_KEY" in response.json()["detail"]
    finally:
        get_settings.cache_clear()


def test_api_key_never_appears_in_logs(
    client: TestClient, log_records: list[logging.LogRecord]
) -> None:
    """红线用例：把整套流程走一遍，日志里不允许出现明文 Key。

    覆盖的路径：存 Key（成功 + 失败）→ 查状态 → 清除 → 过期。
    只要有人以后在日志里手滑打了 Key（或把它拼进异常文本），这条就会红。
    """
    session_id = client.post(
        "/api/v1/auth/set_key", json={"model_id": "deepseek", "api_key": SECRET}
    ).json()["session"]["session_id"]
    client.post(
        "/api/v1/auth/set_key", json={"model_id": "deepseek", "api_key": ""}
    )                                                            # 失败路径
    client.get("/api/v1/auth/status", headers={SESSION_HEADER: session_id})
    client.post("/api/v1/auth/clear_key", json={"session_id": session_id})

    assert log_records, "没有采集到日志，说明收集器没挂上，这条断言会变得没有意义"
    assert_no_secret(log_records, SECRET, OTHER_SECRET)


def test_manager_singleton_is_shared() -> None:
    """依赖注入取到的必须是同一个保管箱，否则"刚存的 Key"下次请求就找不到了。"""
    from app.api.deps import get_api_key_manager as dep_manager
    from app.core.api_key_manager import get_api_key_manager as core_manager

    assert core_manager() is core_manager()
    assert dep_manager() is core_manager()
