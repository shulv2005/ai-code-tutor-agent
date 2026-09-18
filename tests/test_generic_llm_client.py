"""通用 LLM 客户端测试：动态传参、流式输出、以及"人话级"的异常提示。

第 1、2 步把"多模型配置"和"网页填 Key"做完了，这个文件守的是第三步：
**客户端能不能拿着动态传进来的 (model_name, base_url, api_key) 干活，
并且在出问题时给出学生看得懂的提示。**

重点覆盖四类情况：
1. 三种构造方式（直接给参数 / 给配置对象 / 从项目多模型配置里取）；
2. Key 为空时的提示语——必须是"请先在网页上输入 API Key"；
3. 异常映射——401 / 403 / 429 / 超时 / 连不上 / 5xx / 空内容，各有各的话；
4. 流式输出——把 SSE 分片拼起来，并保留"中途断了"的报错语义。

全程用 `httpx.MockTransport`，不联网、不需要真 Key。
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from app.core.config import LLMSettings
from app.core.llm_client import (
    MISSING_API_KEY_HINT,
    GenericLLMClient,
    LLMAuthError,
    LLMClientConfig,
    LLMConfigError,
    LLMConnectionError,
    LLMError,
    LLMMessage,
    LLMRateLimitError,
    LLMResponseError,
    LLMStreamChunk,
    LLMTimeoutError,
    StreamingLLMClient,
    build_llm_client,
    dynamic_llm_client,
    mask_keys_in_text,
)

MESSAGES = [LLMMessage(role="user", content="写一个冒泡排序")]


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def ok_payload(content: str = "代码在这里") -> dict[str, Any]:
    """构造一个标准的 OpenAI 格式响应。"""
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "test-model",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content},
             "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def sse_body(pieces: list[str], *, with_usage: bool = True) -> bytes:
    """拼一个流式响应体（OpenAI 的 SSE 格式）。"""
    lines: list[str] = []
    for index, piece in enumerate(pieces):
        last = index == len(pieces) - 1
        chunk: dict[str, Any] = {
            "id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "test-model",
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": piece} if index == 0
                else {"content": piece},
                "finish_reason": "stop" if last else None,
            }],
        }
        if last and with_usage:
            chunk["usage"] = {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
        lines.append("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n")
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode("utf-8")


def client_with(handler: Any, **options: Any) -> GenericLLMClient:
    """按动态参数造客户端，并把网络层换成桩。

    `api_key` 可以从 options 里覆盖（测"Key 为空"时要传 `api_key=""`），
    所以这里先把它取出来，避免位置参数与关键字参数撞车。
    """
    options.setdefault("retry_backoff_seconds", 0.0)      # 测试里不真的等
    options.setdefault("max_retries", 0)
    return GenericLLMClient.for_model(
        model_name=options.pop("model_name", "test-model"),
        base_url=options.pop("base_url", "https://api.example.com/v1"),
        api_key=options.pop("api_key", "sk-test-1234567890"),
        transport=httpx.MockTransport(handler),
        **options,
    )


# ---------------------------------------------------------------------------
# 1. 动态配置
# ---------------------------------------------------------------------------
async def test_for_model_builds_working_client() -> None:
    """需求里的核心用法：给 model_name / base_url / api_key 就能直接调。"""
    client = client_with(lambda request: httpx.Response(200, json=ok_payload()))
    try:
        response = await client.chat(MESSAGES)
    finally:
        await client.aclose()

    assert response.content == "代码在这里"
    assert response.model == "test-model"
    assert client.endpoint == "https://api.example.com/v1/chat/completions"
    assert client.configured is True


async def test_dynamic_parameters_reach_the_request() -> None:
    """传进来的模型名 / 地址 / Key 必须真的出现在请求里（这是"动态"的意义）。"""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=ok_payload())

    client = GenericLLMClient.for_model(
        model_name="qwen-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/",
        api_key="sk-qwen-abcdefgh",
        transport=httpx.MockTransport(handler),
    )
    try:
        await client.chat(MESSAGES, temperature=0.7, max_tokens=99, stop=["```"])
    finally:
        await client.aclose()

    assert captured["url"] == "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    assert captured["auth"] == "Bearer sk-qwen-abcdefgh"
    assert captured["body"]["model"] == "qwen-plus"
    assert captured["body"]["temperature"] == 0.7
    assert captured["body"]["max_tokens"] == 99
    assert captured["body"]["stop"] == ["```"]
    assert captured["body"]["stream"] is False


def test_config_object_validation_strips_and_checks() -> None:
    """配置对象自己做规整：去空格、去末尾斜杠；缺必填项直接报错。"""
    config = LLMClientConfig(
        model_name="  deepseek-chat  ",
        base_url=" https://api.deepseek.com/v1/ ",
        api_key="sk-abcdefgh",
    )
    assert config.model_name == "deepseek-chat"
    assert config.base_url == "https://api.deepseek.com/v1"

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        # 模型名为空是必填项缺失，构造时就该拦下（pydantic 的 ValidationError）
        LLMClientConfig(model_name="", base_url="https://x/v1")


def test_from_llm_settings_bridges_multi_model_and_session_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """从项目配置生成客户端配置：模型取自多模型表，Key 走"网页优先"的规则。"""
    monkeypatch.setenv("LLM__API_KEY", "")
    monkeypatch.setenv("LLM__MODELS__DEEPSEEK__PROVIDER", "deepseek")
    monkeypatch.setenv("LLM__MODELS__DEEPSEEK__MODEL_NAME", "deepseek-chat")
    monkeypatch.setenv("LLM__MODELS__OLLAMA__PROVIDER", "ollama")
    monkeypatch.setenv("LLM__MODELS__OLLAMA__MODEL_NAME", "qwen2.5-coder:7b")

    from app.core.config import get_settings

    get_settings.cache_clear()
    try:
        settings = get_settings().llm

        # 网页上传了 Key：用它
        with_web_key = LLMClientConfig.from_llm_settings(
            settings, "deepseek", api_key="sk-from-web-12345678"
        )
        assert with_web_key.model_name == "deepseek-chat"
        assert with_web_key.base_url == "https://api.deepseek.com/v1"
        assert with_web_key.api_key.get_secret_value() == "sk-from-web-12345678"
        assert with_web_key.requires_api_key is True

        # 本地模型：不需要 Key，也不该被判成"没配好"
        local = LLMClientConfig.from_llm_settings(settings, "ollama")
        assert local.requires_api_key is False
        assert local.is_local is True
        assert local.client_api_key()          # 会给一个占位串，SDK 才肯收
    finally:
        get_settings.cache_clear()


def test_llm_settings_can_still_be_passed_directly() -> None:
    """旧写法（直接传 LLMSettings）继续有效——项目里大量既有代码是这么用的。"""
    settings = LLMSettings(
        base_url="https://api.example.com/v1",
        api_key="sk-legacy-abcdefgh",
        model="legacy-model",
    )
    client = GenericLLMClient(settings, transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json=ok_payload())
    ))
    assert client.model == "legacy-model"
    assert client.configured is True


# ---------------------------------------------------------------------------
# 2. Key 为空时的提示
# ---------------------------------------------------------------------------
async def test_missing_api_key_message_is_student_friendly() -> None:
    """需求点名的文案：Key 为空时必须提示"请先在网页上输入 API Key"。"""
    called = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        called["n"] += 1                       # 有 Key 才会走到这里
        return httpx.Response(200, json=ok_payload())

    client = client_with(handler, api_key="")
    try:
        with pytest.raises(LLMConfigError) as excinfo:
            await client.chat(MESSAGES)
    finally:
        await client.aclose()

    message = str(excinfo.value)
    assert MISSING_API_KEY_HINT in message
    assert "请先在网页上输入 API Key" in message
    assert client.configured is False
    # 关键：**一个请求都没发出去**（不浪费一次往返，也不会拿到看不懂的 401）
    assert called["n"] == 0


async def test_missing_api_key_also_blocks_streaming() -> None:
    """流式调用同样要先检查 Key，不然会流到一半才报错。"""
    client = client_with(lambda request: httpx.Response(200), api_key="")
    try:
        with pytest.raises(LLMConfigError):
            async for _ in client.stream_chat(MESSAGES):
                pass
    finally:
        await client.aclose()


async def test_local_model_can_run_without_key() -> None:
    """本地模型（Ollama）没 Key 也能调：这是"没网也能演示"的前提。"""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=ok_payload("本地模型的回答"))

    client = GenericLLMClient.for_model(
        model_name="qwen2.5-coder:7b",
        base_url="http://127.0.0.1:11434/v1",
        api_key="",
        requires_api_key=False,
        transport=httpx.MockTransport(handler),
    )
    try:
        response = await client.chat(MESSAGES)
    finally:
        await client.aclose()

    assert response.content == "本地模型的回答"
    assert captured["auth"] == "Bearer not-needed"     # 占位串，本地服务不校验


# ---------------------------------------------------------------------------
# 3. 异常映射：每种失败都要有"人话"
# ---------------------------------------------------------------------------
async def test_invalid_key_maps_to_auth_error_without_retry() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, json={"error": {"message": "Invalid API key provided: sk-abcdefghijklmnop"}})

    client = client_with(handler, max_retries=3)
    try:
        with pytest.raises(LLMAuthError) as excinfo:
            await client.chat(MESSAGES)
    finally:
        await client.aclose()

    message = str(excinfo.value)
    assert "API Key 无效" in message
    assert calls["n"] == 1                 # 鉴权失败重试没意义
    # 服务商回显的 Key 必须被打码，不能出现在提示里
    assert "sk-abcdefghijklmnop" not in message


async def test_rate_limit_is_retried_then_reported() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    client = client_with(handler, max_retries=2)
    try:
        with pytest.raises(LLMRateLimitError) as excinfo:
            await client.chat(MESSAGES)
    finally:
        await client.aclose()

    assert calls["n"] == 3                  # 1 次 + 2 次重试
    assert "太频繁" in str(excinfo.value)


async def test_timeout_is_retried_then_reported() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ReadTimeout("too slow", request=request)

    client = client_with(handler, max_retries=1)
    try:
        with pytest.raises(LLMTimeoutError) as excinfo:
            await client.chat(MESSAGES)
    finally:
        await client.aclose()

    assert calls["n"] == 2
    assert "超时" in str(excinfo.value)


async def test_connection_error_is_reported_clearly() -> None:
    """断网 / 地址写错：提示要指向网络与 base_url，而不是抛一个英文堆栈。"""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = client_with(handler, max_retries=0)
    try:
        with pytest.raises(LLMConnectionError) as excinfo:
            await client.chat(MESSAGES)
    finally:
        await client.aclose()

    message = str(excinfo.value)
    assert "连不上模型服务" in message
    assert "api.example.com" in message      # 把实际用的地址带出来，方便排查


async def test_server_error_is_retried_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, json=ok_payload("恢复了"))

    client = client_with(handler, max_retries=3)
    try:
        response = await client.chat(MESSAGES)
    finally:
        await client.aclose()

    assert response.content == "恢复了"
    assert response.attempts == 3


async def test_bad_request_is_not_retried_and_keeps_status_code() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, json={"error": {"message": "messages is required"}})

    client = client_with(handler, max_retries=2)
    try:
        with pytest.raises(LLMError) as excinfo:
            await client.chat(MESSAGES)
    finally:
        await client.aclose()

    assert calls["n"] == 1
    assert "400" in str(excinfo.value)


async def test_empty_content_is_reported() -> None:
    payload = ok_payload("   ")
    client = client_with(lambda request: httpx.Response(200, json=payload))
    try:
        with pytest.raises(LLMResponseError) as excinfo:
            await client.chat(MESSAGES)
    finally:
        await client.aclose()
    assert "空内容" in str(excinfo.value)


async def test_missing_choices_is_reported() -> None:
    client = client_with(lambda request: httpx.Response(200, json={"model": "m"}))
    try:
        with pytest.raises(LLMResponseError):
            await client.chat(MESSAGES)
    finally:
        await client.aclose()


def test_mask_keys_in_text_hides_secret_shaped_fragments() -> None:
    assert mask_keys_in_text("Incorrect API key: sk-abcdefghijklmnop") == "Incorrect API key: ***"
    assert mask_keys_in_text("普通文字不受影响") == "普通文字不受影响"


# ---------------------------------------------------------------------------
# 4. 流式输出
# ---------------------------------------------------------------------------
async def test_stream_chat_yields_deltas_in_order() -> None:
    """流式：一段段返回，顺序与内容都要对，最后一帧带结束标记与用量。"""
    client = client_with(
        lambda request: httpx.Response(
            200, headers={"content-type": "text/event-stream"},
            content=sse_body(["冒泡", "排序", "的代码"]),
        )
    )
    chunks: list[LLMStreamChunk] = []
    try:
        async for chunk in client.stream_chat(MESSAGES):
            chunks.append(chunk)
    finally:
        await client.aclose()

    assert "".join(chunk.delta for chunk in chunks) == "冒泡排序的代码"
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.total_tokens == 7
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


async def test_stream_request_asks_for_usage_and_streams() -> None:
    """流式请求体要带 stream=True 与 include_usage（否则拿不到 token 用量）。"""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              content=sse_body(["好"]))

    client = client_with(handler)
    try:
        async for _ in client.stream_chat(MESSAGES):
            pass
    finally:
        await client.aclose()

    assert captured["body"]["stream"] is True
    assert captured["body"]["stream_options"] == {"include_usage": True}


async def test_stream_error_is_mapped() -> None:
    """流式请求刚发出去就失败时，也要给中文提示（而不是英文异常）。"""
    client = client_with(lambda request: httpx.Response(401, json={"error": "bad key"}))
    try:
        with pytest.raises(LLMAuthError):
            async for _ in client.stream_chat(MESSAGES):
                pass
    finally:
        await client.aclose()


def test_generic_client_satisfies_streaming_protocol() -> None:
    """客户端要同时满足两个协议：普通调用与流式调用。"""
    client = client_with(lambda request: httpx.Response(200, json=ok_payload()))
    try:
        assert isinstance(client, StreamingLLMClient)
    finally:
        import asyncio

        asyncio.run(client.aclose())


# ---------------------------------------------------------------------------
# 5. 工厂与上下文管理器（接口层要用的那两个）
# ---------------------------------------------------------------------------
async def test_dynamic_llm_client_closes_after_use() -> None:
    """"临时用一下"的客户端：出了 with 就关掉连接池，用户 Key 不留驻内存。"""
    async with dynamic_llm_client(
        LLMClientConfig(model_name="test-model", base_url="https://api.example.com/v1",
                        api_key="sk-test-1234567890"),
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=ok_payload())),
    ) as client:
        response = await client.chat(MESSAGES)
        assert response.content == "代码在这里"
        sdk_before = client._sdk                     # noqa: SLF001 - 测试内部状态
        assert sdk_before is not None

    assert client._sdk is None                       # noqa: SLF001 - 已关闭


async def test_build_llm_client_keeps_user_key_out_of_shared_singleton() -> None:
    """按需构造的客户端**不进单例**：否则用户 Key 会跟着连接池一直留在内存里。"""
    from app.core.llm_client import _shared  # noqa: PLC0415 - 测试要看内部单例状态

    before = dict(_shared)
    client = build_llm_client(
        LLMClientConfig(model_name="m", base_url="https://api.example.com/v1",
                        api_key="sk-user-abcdefgh"),
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=ok_payload())),
    )
    try:
        assert dict(_shared) == before
        assert "sk-user-abcdefgh" not in str(_shared)
    finally:
        await client.aclose()
