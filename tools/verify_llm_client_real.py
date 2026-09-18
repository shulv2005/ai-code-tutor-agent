"""临时验收脚本：用真实模型验证通用 LLM 客户端。

覆盖：动态配置调用、流式输出、Key 无效的提示、Key 为空的提示。
只从 .env 读配置，脚本本身不会打印 Key。
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from app.core.config import get_settings  # noqa: E402
from app.core.llm_client import (  # noqa: E402
    GenericLLMClient,
    LLMAuthError,
    LLMClientConfig,
    LLMConfigError,
    LLMMessage,
)

MESSAGES = [
    LLMMessage(role="system", content="你是一个简洁的助手，回答不超过 20 个字。"),
    LLMMessage(role="user", content="用一句话说明冒泡排序在做什么"),
]

results: list[tuple[str, bool, str]] = []


def record(label: str, ok: bool, detail: str = "") -> None:
    results.append((label, ok, detail))
    print(f"  {'OK  ' if ok else 'FAIL'} {label}" + (f"  —— {detail}" if detail else ""))


class Client:
    """极简异步上下文管理器：用完自动关连接池。"""

    def __init__(self, config: LLMClientConfig) -> None:
        self._client = GenericLLMClient(config)

    async def __aenter__(self) -> GenericLLMClient:
        return self._client

    async def __aexit__(self, *exc: object) -> None:
        await self._client.aclose()


async def main() -> int:
    settings = get_settings().llm
    config = LLMClientConfig.from_llm_settings(settings)
    print("=" * 72)
    print("通用 LLM 客户端验收（真实模型）")
    print("=" * 72)
    print(f"模型={config.model_name}  地址={config.base_url}  Key={config.masked_key()}")
    print()

    print("1. 一次性调用（动态传入 model_name / base_url / api_key）")
    async with Client(config) as client:
        started = time.perf_counter()
        response = await client.chat(MESSAGES, max_tokens=64)
        elapsed = (time.perf_counter() - started) * 1000
    record("拿到正文", bool(response.content.strip()), response.content.strip()[:40])
    record("一次成功（没走重试）", response.attempts == 1, f"attempts={response.attempts}")
    record(
        "带回 token 用量",
        response.usage.total_tokens > 0,
        f"prompt={response.usage.prompt_tokens} completion={response.usage.completion_tokens}",
    )
    record("记录了耗时", response.latency_ms > 0, f"{response.latency_ms:.0f}ms（实测 {elapsed:.0f}ms）")

    print()
    print("2. 流式输出")
    pieces: list[str] = []
    first_ms = 0.0
    finish: str | None = None
    usage = None
    started = time.perf_counter()
    async with Client(config) as client:
        async for chunk in client.stream_chat(MESSAGES, max_tokens=64):
            if chunk.delta and not pieces:
                first_ms = (time.perf_counter() - started) * 1000
            if chunk.delta:
                pieces.append(chunk.delta)
            if chunk.finish_reason:
                finish = chunk.finish_reason
                usage = chunk.usage
    text = "".join(pieces)
    record("流式拼出完整正文", bool(text.strip()), text.strip()[:40])
    record("分成多片（确实是流式）", len(pieces) > 1, f"{len(pieces)} 片")
    record("首片很快到（不必等全文）", 0 < first_ms < 5000, f"{first_ms:.0f}ms")
    record("最后一帧带结束标记", finish == "stop", f"finish_reason={finish}")
    record(
        "用量信息在最后一帧",
        usage is not None and usage.total_tokens > 0,
        f"total={getattr(usage, 'total_tokens', '-')}",
    )

    print()
    print("3. Key 无效时的提示")
    bad = LLMClientConfig(
        model_name=config.model_name,
        base_url=config.base_url,
        api_key="sk-this-key-is-wrong-000000",
        max_retries=0,
    )
    async with Client(bad) as client:
        try:
            await client.chat(MESSAGES, max_tokens=8)
            record("Key 无效应当报错", False, "居然成功了？")
        except LLMAuthError as exc:
            record("抛出 LLMAuthError", True, str(exc)[:90])
            record("提示是中文且说清原因", "API Key 无效" in str(exc))
            record("提示里没有明文 Key", "sk-this-key-is-wrong" not in str(exc))

    print()
    print("4. Key 为空时的提示")
    empty = LLMClientConfig(model_name=config.model_name, base_url=config.base_url, api_key="")
    async with Client(empty) as client:
        try:
            await client.chat(MESSAGES)
            record("Key 为空应当报错", False, "居然成功了？")
        except LLMConfigError as exc:
            record("抛出 LLMConfigError", True, str(exc))
            record("提示语含「请先在网页上输入 API Key」", "请先在网页上输入 API Key" in str(exc))

    print()
    print("=" * 72)
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"验收结果：{passed}/{len(results)} 项通过")
    print("=" * 72)
    for label, ok, _ in results:
        if not ok:
            print(f"  FAIL  {label}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
