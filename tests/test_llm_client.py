"""LLM 客户端测试：用 httpx.MockTransport 覆盖重试、错误映射与响应解析。

这样能在不联网、不需要 API Key 的情况下验证客户端全部关键逻辑。
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from app.core.config import LLMSettings
from app.core.llm_client import (
    LLMAuthError,
    LLMConfigError,
    LLMError,
    LLMMessage,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
    OpenAICompatibleClient,
)

MESSAGES = [LLMMessage(role="user", content="hi")]


def _settings(**overrides: Any) -> LLMSettings:
    base: dict[str, Any] = {
        "base_url": "https://api.example.com/v1",
        "api_key": SecretStr("sk-test"),
        "model": "test-model",
        "max_retries": 2,
        "retry_backoff_seconds": 0.0,  # 测试中不真的等待
    }
    base.update(overrides)
    return LLMSettings(**base)


def _ok_payload(content: str = "hello", **extra: Any) -> dict[str, Any]:
    payload = {
        "model": "test-model",
        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    payload.update(extra)
    return payload


def _client(handler: Any, **overrides: Any) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        _settings(**overrides), transport=httpx.MockTransport(handler)
    )


# ---------------------------------------------------------------------------
# 正常路径
# ---------------------------------------------------------------------------
async def test_successful_chat_parses_content_and_usage() -> None:
    client = _client(lambda request: httpx.Response(200, json=_ok_payload("代码在这里")))
    try:
        response = await client.chat(MESSAGES)
    finally:
        await client.aclose()

    assert response.content == "代码在这里"
    assert response.model == "test-model"
    assert response.usage.prompt_tokens == 10
    assert response.usage.completion_tokens == 5
    assert response.usage.total_tokens == 15
    assert response.finish_reason == "stop"
    assert response.attempts == 1
    assert response.truncated is False


async def test_finish_reason_length_marks_truncated() -> None:
    payload = _ok_payload("partial code")
    payload["choices"][0]["finish_reason"] = "length"
    client = _client(lambda request: httpx.Response(200, json=payload))
    try:
        response = await client.chat(MESSAGES)
    finally:
        await client.aclose()
    assert response.truncated is True


async def test_content_returned_as_parts_array() -> None:
    """部分网关/多模态模型把正文放在数组里。"""
    payload = _ok_payload()
    payload["choices"][0]["message"]["content"] = [
        {"type": "text", "text": "part one"},
        {"type": "text", "text": "part two"},
    ]
    client = _client(lambda request: httpx.Response(200, json=payload))
    try:
        response = await client.chat(MESSAGES)
    finally:
        await client.aclose()
    assert "part one" in response.content
    assert "part two" in response.content


async def test_reasoning_content_fallback() -> None:
    """推理模型正文为空时回退到 reasoning_content。"""
    payload = _ok_payload()
    payload["choices"][0]["message"] = {"role": "assistant", "content": "", "reasoning_content": "思考"}
    client = _client(lambda request: httpx.Response(200, json=payload))
    try:
        response = await client.chat(MESSAGES)
    finally:
        await client.aclose()
    assert response.content == "思考"


async def test_request_payload_and_headers() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_ok_payload())

    client = _client(handler)
    try:
        await client.chat(MESSAGES, temperature=0.7, max_tokens=123, stop=["```"])
    finally:
        await client.aclose()

    assert captured["url"] == "https://api.example.com/v1/chat/completions"
    assert captured["auth"] == "Bearer sk-test"
    assert captured["body"]["model"] == "test-model"
    assert captured["body"]["temperature"] == 0.7
    assert captured["body"]["max_tokens"] == 123
    assert captured["body"]["stop"] == ["```"]
    assert captured["body"]["stream"] is False
    assert captured["body"]["messages"] == [{"role": "user", "content": "hi"}]


async def test_base_url_trailing_slash_is_tolerated() -> None:
    client = _client(
        lambda request: httpx.Response(200, json=_ok_payload()),
        base_url="https://api.example.com/v1/",
    )
    assert client.endpoint == "https://api.example.com/v1/chat/completions"
    await client.aclose()


async def test_extra_headers_are_sent() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["x"] = request.headers.get("x-custom")
        return httpx.Response(200, json=_ok_payload())

    client = _client(handler, extra_headers={"X-Custom": "v"})
    try:
        await client.chat(MESSAGES)
    finally:
        await client.aclose()
    assert captured["x"] == "v"


# ---------------------------------------------------------------------------
# 错误映射
# ---------------------------------------------------------------------------
async def test_missing_api_key_raises_config_error() -> None:
    client = _client(lambda request: httpx.Response(200, json=_ok_payload()), api_key=SecretStr(""))
    try:
        with pytest.raises(LLMConfigError) as excinfo:
            await client.chat(MESSAGES)
    finally:
        await client.aclose()
    assert "LLM__API_KEY" in str(excinfo.value)
    assert client.configured is False


async def test_auth_error_is_not_retried() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, json={"error": "invalid key"})

    client = _client(handler)
    try:
        with pytest.raises(LLMAuthError):
            await client.chat(MESSAGES)
    finally:
        await client.aclose()
    # 鉴权失败重试无意义，必须只调用一次
    assert calls["n"] == 1


async def test_client_error_4xx_is_not_retried() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, json={"error": "bad request"})

    client = _client(handler)
    try:
        with pytest.raises(LLMError) as excinfo:
            await client.chat(MESSAGES)
    finally:
        await client.aclose()
    assert calls["n"] == 1
    assert "400" in str(excinfo.value)


async def test_rate_limit_is_retried_then_raises() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429, json={"error": "slow down"})

    client = _client(handler, max_retries=2)
    try:
        with pytest.raises(LLMRateLimitError):
            await client.chat(MESSAGES)
    finally:
        await client.aclose()
    assert calls["n"] == 3  # 1 次 + 2 次重试


async def test_server_error_is_retried_then_succeeds() -> None:
    """5xx 重试后成功应正常返回，并记录实际尝试次数。"""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, json=_ok_payload("recovered"))

    client = _client(handler, max_retries=3)
    try:
        response = await client.chat(MESSAGES)
    finally:
        await client.aclose()

    assert response.content == "recovered"
    assert response.attempts == 3
    assert calls["n"] == 3


async def test_timeout_is_retried_then_raises() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ReadTimeout("too slow", request=request)

    client = _client(handler, max_retries=1)
    try:
        with pytest.raises(LLMTimeoutError):
            await client.chat(MESSAGES)
    finally:
        await client.aclose()
    assert calls["n"] == 2


async def test_missing_choices_raises_response_error() -> None:
    client = _client(lambda request: httpx.Response(200, json={"model": "m"}))
    try:
        with pytest.raises(LLMResponseError):
            await client.chat(MESSAGES)
    finally:
        await client.aclose()


async def test_empty_content_raises_response_error() -> None:
    payload = _ok_payload("")
    payload["choices"][0]["message"]["content"] = "   "
    client = _client(lambda request: httpx.Response(200, json=payload))
    try:
        with pytest.raises(LLMResponseError):
            await client.chat(MESSAGES)
    finally:
        await client.aclose()


async def test_close_is_idempotent() -> None:
    client = _client(lambda request: httpx.Response(200, json=_ok_payload()))
    await client.aclose()
    await client.aclose()  # 重复关闭不应抛异常

    # 关闭后仍可再次使用（懒重建连接池）
    response = await client.chat(MESSAGES)
    assert response.content == "hello"
    await client.aclose()
