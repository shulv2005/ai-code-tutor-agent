"""Agent Trace 单元测试：装饰器、上下文管理器、脱敏、嵌套与截断。"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator

import pytest

from app.core.trace import (
    MemoryTraceSink,
    TraceRecorder,
    current_trace_id,
    set_trace_recorder,
    trace_span,
    traced,
)


@pytest.fixture()
def recorder() -> Iterator[tuple[TraceRecorder, MemoryTraceSink]]:
    sink = MemoryTraceSink(maxlen=100)
    rec = TraceRecorder([sink])
    set_trace_recorder(rec)
    yield rec, sink
    set_trace_recorder(None)


def test_trace_span_records_output_and_duration(
    recorder: tuple[TraceRecorder, MemoryTraceSink],
) -> None:
    _, sink = recorder

    with trace_span("demo", kind="tool", payload={"city": "hangzhou"}) as span:
        time.sleep(0.02)
        span.set_output({"ok": True})

    record = sink.last()
    assert record is not None
    assert record.name == "demo"
    assert record.kind == "tool"
    assert record.status == "ok"
    # 同上：时钟粒度会让实测略短于请求值，留余量避免偶发失败
    assert record.duration_ms >= 15
    assert '"city": "hangzhou"' in (record.input or "")
    assert '"ok": true' in (record.output or "")
    assert record.trace_id == span.trace_id


def test_trace_span_records_error(recorder: tuple[TraceRecorder, MemoryTraceSink]) -> None:
    _, sink = recorder

    with pytest.raises(ValueError):
        with trace_span("boom"):
            raise ValueError("bad input")

    record = sink.last()
    assert record is not None
    assert record.status == "error"
    assert record.error == "ValueError: bad input"


def test_sensitive_fields_are_masked(recorder: tuple[TraceRecorder, MemoryTraceSink]) -> None:
    _, sink = recorder

    with trace_span("llm_call", payload={"api_key": "sk-live-123", "prompt": "hi"}):
        pass

    payload = sink.last().input or ""
    assert "sk-live-123" not in payload
    assert "***" in payload
    assert "hi" in payload


def test_nested_spans_share_trace_id(recorder: tuple[TraceRecorder, MemoryTraceSink]) -> None:
    _, sink = recorder

    with trace_span("outer") as outer:
        with trace_span("inner"):
            # 内层无需绑定变量：这里要验证的是它继承外层的 trace_id
            assert current_trace_id() == outer.trace_id

    inner_record, outer_record = sink.records()
    assert inner_record.trace_id == outer_record.trace_id
    assert inner_record.parent_span_id == outer.span_id
    assert outer_record.parent_span_id is None


def test_traced_decorator_on_sync_function(recorder: tuple[TraceRecorder, MemoryTraceSink]) -> None:
    _, sink = recorder

    @traced("add", kind="tool")
    def add(a: int, b: int) -> int:
        return a + b

    assert add(1, 2) == 3
    record = sink.last()
    assert record is not None
    assert record.name == "add"
    assert record.output == "3"
    assert '"a": 1' in (record.input or "")


async def test_traced_decorator_on_async_function(
    recorder: tuple[TraceRecorder, MemoryTraceSink],
) -> None:
    _, sink = recorder

    @traced("fetch", kind="rag")
    async def fetch(delay: float) -> str:
        await asyncio.sleep(delay)
        return "context"

    assert await fetch(0.02) == "context"
    record = sink.last()
    assert record is not None
    assert record.kind == "rag"
    # 留出余量：asyncio.sleep 在 Windows 上受时钟粒度影响，实测可能略短于请求值，
    # 因此按 >=15ms（而非 >=20ms）断言，避免满负载下偶发失败。
    assert record.duration_ms >= 15
    assert record.output == '"context"'


def test_decorator_preserves_metadata(recorder: tuple[TraceRecorder, MemoryTraceSink]) -> None:
    _, sink = recorder

    @traced("generate_patch", kind="agent", metadata={"model": "deepseek-coder"})
    def generate() -> str:
        return "diff"

    generate()
    assert sink.last().metadata["model"] == "deepseek-coder"


def test_payload_is_truncated(recorder: tuple[TraceRecorder, MemoryTraceSink]) -> None:
    _, sink = recorder
    big = "x" * 5000

    with trace_span("big", payload={"content": big}):
        pass

    assert "truncated" in (sink.last().input or "")

