"""动态 API Key 测试：三个 AI 服务是否真的用了"网页上填的那把 Key"。

背景（这是第 5 步的核心）：
    以前 API Key 只能写在 `.env` 里，服务启动时就把客户端固定下来了。
    现在用户可以在网页上选模型、填自己的 Key，于是**服务层也必须认这把 Key**：

      POST /api/v1/check/code      ← 深度检测
      POST /api/v1/fix/code        ← 改错
      POST /api/v1/comment/generate← 注释生成

    三个接口都可以带 `model_id` / `api_key` 字段，或带 `X-Session-Id` 请求头
    （Key 存在服务端内存会话里时用这个）。

本文件守三件事：
  1. **没 Key 时的提示**：三个服务的 note 里必须出现"请先在网页上输入 API Key"；
  2. **动态 Key 真的传下去了**：服务会拿 session/model/key 去问 `api_key_manager`，
     并用解析出来的模型与 Key 构造客户端（用替身客户端断言，不发真实请求）；
  3. **老用法不受影响**：什么都不传时仍走 `.env` 配置。
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager

import pytest
from fastapi.testclient import TestClient

from app.core.api_key_manager import ApiKeyManager, redact, reset_api_key_manager
from app.core.config import get_settings
from app.core.llm_client import LLMMessage, LLMResponse, LLMUsage
from app.services.code_checker import CodeChecker
from app.services.code_fixer import CodeFixer
from app.services.comment_generator import CommentGenerator

PY_OK = "def add(a, b):\n    return a + b\n"
PY_BROKEN = "def add(a, b)\n    return a + b\n"


# ---------------------------------------------------------------------------
# 替身：假客户端 + 记录"服务到底要求用哪个模型/哪把 Key"
# ---------------------------------------------------------------------------
class RecordingLLM:
    """假客户端：记录被调用的次数，并返回合法的 JSON。"""

    def __init__(self, payload: str = '{"had_error": false, "score": 90}') -> None:
        self.payload = payload
        self.calls = 0

    @property
    def model(self) -> str:
        return "recording-model"

    @property
    def configured(self) -> bool:
        return True

    async def chat(self, messages: list[LLMMessage], **_: object) -> LLMResponse:
        self.calls += 1
        # 三个服务的提示词不同，各给一份能解析的 JSON 更省事
        content = self.payload
        if "注释" in messages[-1].content or "comment" in messages[0].content:
            content = '{"commented_code": "# 注释\\ndef add(a, b):\\n    return a + b\\n", "summary": "ok"}'
        if "fixed_code" in messages[0].content or "改错" in messages[0].content:
            content = (
                '{"had_error": true, "summary": "少写冒号", '
                '"fixed_code": "def add(a, b):\\n    return a + b\\n", '
                '"changes": [{"line": 1, "original": "def add(a, b)", '
                '"fixed": "def add(a, b):", "reason": "函数定义要冒号", '
                '"what": "少了冒号", "why": "语法要求", "how": "补上冒号", '
                '"avoid": "写完函数头就补冒号"}]}'
            )
        return LLMResponse(
            content=content, model="recording-model", finish_reason="stop",
            usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    async def aclose(self) -> None:
        """假客户端没什么要关的。"""


class UnconfiguredLLM(RecordingLLM):
    """假客户端：模拟"没有任何 Key"的情况（服务应降级并给出提示）。"""

    @property
    def configured(self) -> bool:
        return False

    async def chat(self, messages: list[LLMMessage], **_: object) -> LLMResponse:
        raise AssertionError("没配 Key 时不应该真的去调模型")


@pytest.fixture(autouse=True)
def _clean_manager() -> Iterator[None]:
    """每个用例前后都清干净内存会话，避免用例之间互相影响。"""
    reset_api_key_manager()
    yield
    reset_api_key_manager()


@contextmanager
def capture_request_args(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[dict[str, object]]]:
    """把"服务向 api_key_manager 要 Key 时传的参数"记下来。

    做法：替换 `llm_client_for_request`，它本来负责
    "拿 session/model/key 去问保管箱、再构造客户端"。
    这里换成记账版本，就能确认三个服务**确实把网页来的信息传下去了**。
    """
    captured: list[dict[str, object]] = []

    @asynccontextmanager
    async def fake_helper(settings, *, session_id=None, model_id=None, api_key=None, fallback=None):
        # 注意必须是**异步**上下文管理器：服务层用的是 `async with`
        captured.append(
            {"session_id": session_id, "model_id": model_id, "api_key": api_key,
             "used_fallback": session_id is None and model_id is None and api_key is None}
        )
        yield fallback if fallback is not None else RecordingLLM()

    import app.services.code_checker as checker_module
    import app.services.code_fixer as fixer_module
    import app.services.comment_generator as comment_module

    monkeypatch.setattr(checker_module, "llm_client_for_request", fake_helper)
    monkeypatch.setattr(fixer_module, "llm_client_for_request", fake_helper)
    monkeypatch.setattr(comment_module, "llm_client_for_request", fake_helper)
    yield captured


@pytest.fixture()
def settings():
    """项目配置。

    测试夹具（conftest 的 autouse）已经把 `LLM__API_KEY` 置空、把各目录指向临时目录，
    所以这里拿到的配置与开发机上的 `.env` 无关，结果可复现。
    """
    return get_settings()


# ---------------------------------------------------------------------------
# 1. 没有 Key：三个服务都要给出那句统一提示
# ---------------------------------------------------------------------------
async def test_check_without_key_tells_user_to_enter_it(settings) -> None:
    """没有 Key 时：不抛异常，返回本地结论 + 「请先在网页上输入 API Key」。"""
    checker = CodeChecker(settings, UnconfiguredLLM())
    outcome = await checker.check(PY_OK, language="python", filename="a.py")

    assert outcome.ai_available is False
    assert "请先在网页上输入 API Key" in outcome.note
    assert any("请先在网页上输入 API Key" in item for item in outcome.warnings)


async def test_fix_without_key_tells_user_to_enter_it(settings) -> None:
    fixer = CodeFixer(settings, UnconfiguredLLM())
    outcome = await fixer.fix(PY_BROKEN, language="python", filename="a.py")

    assert outcome.ai_available is False
    assert outcome.fixed_code == PY_BROKEN          # 原代码原样返回，不丢学生代码
    assert "请先在网页上输入 API Key" in outcome.note


async def test_comment_without_key_tells_user_to_enter_it(settings) -> None:
    generator = CommentGenerator(settings, UnconfiguredLLM())
    outcome = await generator.generate(PY_OK, language="python", filename="a.py")

    assert outcome.ai_available is False
    assert outcome.commented_code == PY_OK
    assert "请先在网页上输入 API Key" in outcome.note


# ---------------------------------------------------------------------------
# 2. 动态 Key 真的传到了保管箱
# ---------------------------------------------------------------------------
async def test_check_forwards_session_and_model(
    settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """服务把 session_id / model_id / api_key 原样交给"取凭据"的那一步。"""
    with capture_request_args(monkeypatch) as captured:
        checker = CodeChecker(settings, RecordingLLM())
        await checker.check(
            PY_OK, language="python", filename="a.py",
            session_id="sess-1", model_id="qwen", api_key="sk-inline",
        )

    assert captured and captured[0]["session_id"] == "sess-1"
    assert captured[0]["model_id"] == "qwen"
    assert captured[0]["api_key"] == "sk-inline"
    assert captured[0]["used_fallback"] is False


async def test_fix_and_comment_forward_credentials(
    settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """改错与注释生成同样要把网页来的凭据传下去（不能只顾检测）。"""
    with capture_request_args(monkeypatch) as captured:
        fixer = CodeFixer(settings, RecordingLLM())
        generator = CommentGenerator(settings, RecordingLLM())
        await fixer.fix(
            PY_BROKEN, language="python", filename="a.py",
            session_id="sess-2", model_id="deepseek",
        )
        await generator.generate(
            PY_OK, language="python", filename="a.py",
            session_id="sess-2", model_id="deepseek",
        )

    assert [item["session_id"] for item in captured] == ["sess-2", "sess-2"]
    assert all(item["model_id"] == "deepseek" for item in captured)


async def test_nothing_provided_keeps_old_behaviour(
    settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """三个参数都不传时，走既有路径（`fallback` = 构造时注入的客户端）。"""
    with capture_request_args(monkeypatch) as captured:
        checker = CodeChecker(settings, RecordingLLM())
        outcome = await checker.check(PY_OK, language="python", filename="a.py")

    assert captured and captured[0]["used_fallback"] is True
    assert outcome.ai_available is True          # 用的是注入的假客户端


# ---------------------------------------------------------------------------
# 3. 端到端：通过 HTTP 接口带会话号调用
# ---------------------------------------------------------------------------
@pytest.fixture()
def client(sqlite_path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """真实应用 + 多模型配置（不带任何真 Key）。"""
    monkeypatch.setenv("LLM__API_KEY", "")
    monkeypatch.setenv("LLM__MODELS__DEEPSEEK__PROVIDER", "deepseek")
    monkeypatch.setenv("LLM__MODELS__DEEPSEEK__MODEL_NAME", "deepseek-chat")
    from app.core.config import get_settings

    get_settings.cache_clear()
    from app.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client
    get_settings.cache_clear()


def test_deep_endpoints_degrade_with_clear_message(client: TestClient) -> None:
    """三个深度接口在"没 Key"时返回 200 + 明确提示，而不是 500/静默失败。

    这是刻意的设计：本地静态结论已经算出来了，不能因为没 Key 就把它扔掉，
    但必须在 `note` 里说清楚"请先在网页上输入 API Key"。
    """
    for path, body in (
        ("/api/v1/check/code", {"code": PY_OK, "language": "python"}),
        ("/api/v1/fix/code", {"code": PY_BROKEN, "language": "python"}),
        ("/api/v1/comment/generate", {"code": PY_OK, "language": "python"}),
    ):
        response = client.post(path, json=body)
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["ai_available"] is False
        assert "请先在网页上输入 API Key" in payload["note"]


def test_deep_endpoints_accept_model_and_key_fields(client: TestClient) -> None:
    """请求体里的 model_id / api_key 能被接受（不会 422），并如实降级。"""
    response = client.post(
        "/api/v1/check/code",
        json={"code": PY_OK, "language": "python",
              "model_id": "deepseek", "api_key": "sk-not-a-real-key"},
    )
    # Key 是假的，所以最终会调用失败并降级；但请求本身必须被正确解析
    assert response.status_code == 200, response.text
    assert response.json()["ai_available"] is False


def test_session_header_is_recognised(client: TestClient) -> None:
    """带一个不存在的会话号也要能正常返回（回落配置），不能 500。"""
    response = client.post(
        "/api/v1/check/code",
        json={"code": PY_OK, "language": "python", "model_id": "deepseek"},
        headers={"X-Session-Id": "not-a-real-session-id"},
    )
    assert response.status_code == 200, response.text


def test_error_detail_never_leaks_key(client: TestClient) -> None:
    """失败时的提示里不能出现明文 Key（脱敏函数兜底）。"""
    secret = "sk-should-never-appear-123456"
    response = client.post(
        "/api/v1/check/code",
        json={"code": PY_OK, "language": "python", "api_key": secret, "model_id": "deepseek"},
    )
    assert secret not in response.text
    assert secret not in redact(response.text, secret)


# ---------------------------------------------------------------------------
# 4. 保管箱与服务的约定（纯单元）
# ---------------------------------------------------------------------------
def test_manager_resolves_model_and_key_for_services(settings) -> None:
    """服务拿到的凭据来自保管箱：模型与 Key 都要对得上。"""
    manager = ApiKeyManager(ttl_seconds=1800)
    record = manager.set_key("deepseek", "sk-from-page")

    credential = manager.resolve(record.session_id, "deepseek", None)
    assert credential.source == "session"
    assert credential.model_id == "deepseek"
    assert credential.api_key == "sk-from-page"


def test_missing_key_note_mentions_web_page(settings) -> None:
    """统一提示语里必须有"请先在网页上输入 API Key"这几个字（需求原话）。"""
    from app.core.llm_client import MISSING_API_KEY_HINT, MISSING_API_KEY_NOTE

    assert "请先在网页上输入 API Key" in MISSING_API_KEY_HINT
    for extra in ("检测", "改错", "注释生成"):
        assert "请先在网页上输入 API Key" in MISSING_API_KEY_NOTE.format(extra=extra)
