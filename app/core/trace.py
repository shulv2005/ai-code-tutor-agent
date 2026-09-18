"""自建 Agent Trace：轻量级调用链追踪，覆盖 Agent / 工具 / LLM / 沙箱调用。

设计要点：
1. 用 contextvars 在同步与异步调用栈中透传 trace_id，支持跨 Agent 嵌套。
2. 两种埋点方式：`trace_span`（上下文管理器）与 `@traced`（装饰器）。
3. 记录通过可插拔 Sink 落地：默认结构化日志，内存 Sink 便于测试；
   后续接 SQLite 持久化或 OpenTelemetry 时，只需新增一个 Sink 实现。
4. 输入输出会做敏感字段脱敏与长度截断，避免 API Key 泄漏与日志爆炸。
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import json
import logging
import time
import uuid
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import TracebackType
from typing import Any, Literal, ParamSpec, Protocol, TypeVar, runtime_checkable

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.core.config import get_settings

logger = logging.getLogger("app.trace")

SpanKind = Literal["agent", "tool", "llm", "rag", "sandbox", "db", "http", "internal"]
SpanStatus = Literal["ok", "error"]

P = ParamSpec("P")
R = TypeVar("R")

# 字段名命中这些关键字（不区分大小写）时会被脱敏
SENSITIVE_KEYWORDS: frozenset[str] = frozenset(
    {"api_key", "apikey", "token", "password", "passwd", "secret", "authorization", "credential"}
)
MASK = "***"


def new_trace_id() -> str:
    """生成全局唯一的 trace_id。"""
    return uuid.uuid4().hex


def new_span_id() -> str:
    """生成 span_id。"""
    return uuid.uuid4().hex[:16]


@dataclass(frozen=True, slots=True)
class TraceContext:
    """当前执行上下文中的 trace 信息。"""

    trace_id: str
    span_id: str
    parent_span_id: str | None = None


_context: contextvars.ContextVar[TraceContext | None] = contextvars.ContextVar(
    "agent_trace_context", default=None
)


def current_context() -> TraceContext | None:
    """返回当前 TraceContext；不在追踪范围内时为 None。"""
    return _context.get()


def current_trace_id() -> str | None:
    """返回当前 trace_id。"""
    ctx = _context.get()
    return ctx.trace_id if ctx else None


# 说明：这里曾经还有一个 current_span_id()，但它全项目无人调用；
# 需要 span_id 时用 current_context().span_id 即可，因此已删除，避免留下死代码。


@dataclass(slots=True)
class TraceRecord:
    """一条完整的调用记录。"""

    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    kind: SpanKind
    started_at: datetime
    duration_ms: float
    status: SpanStatus
    input: str | None = None
    output: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转为可 JSON 序列化的字典。"""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "kind": self.kind,
            "started_at": self.started_at.isoformat(),
            "duration_ms": round(self.duration_ms, 3),
            "status": self.status,
            "input": self.input,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata,
        }


@runtime_checkable
class TraceSink(Protocol):
    """Trace 记录落地接口（日志、内存、数据库、OTel 等）。"""

    def emit(self, record: TraceRecord) -> None: ...


class LoggingTraceSink:
    """把 Trace 记录写成一行结构化日志。"""

    def __init__(self, level: int = logging.INFO) -> None:
        self._level = level

    def emit(self, record: TraceRecord) -> None:
        logger.log(
            self._level,
            "[trace] %s %s (%s) %.2fms",
            record.kind,
            record.name,
            record.status,
            record.duration_ms,
            extra={
                "trace_id": record.trace_id,
                "span_id": record.span_id,
                "parent_span_id": record.parent_span_id,
                "duration_ms": record.duration_ms,
                "trace_payload": json.dumps(record.to_dict(), ensure_ascii=False, default=str),
            },
        )


class MemoryTraceSink:
    """内存环形缓冲：便于本地调试与单元测试断言（生产可换 DB Sink）。"""

    def __init__(self, maxlen: int = 1000) -> None:
        self._records: deque[TraceRecord] = deque(maxlen=maxlen)

    def emit(self, record: TraceRecord) -> None:
        self._records.append(record)

    def records(self) -> list[TraceRecord]:
        """按写入顺序返回全部记录。"""
        return list(self._records)

    def last(self) -> TraceRecord | None:
        """返回最近一条记录。"""
        return self._records[-1] if self._records else None

    def clear(self) -> None:
        """清空缓冲。"""
        self._records.clear()


class TraceRecorder:
    """Trace 记录分发器：把记录广播给所有已注册的 Sink。"""

    def __init__(self, sinks: list[TraceSink] | None = None) -> None:
        self._sinks: list[TraceSink] = list(sinks or [])

    def add_sink(self, sink: TraceSink) -> None:
        """注册一个 Sink。"""
        self._sinks.append(sink)

    def emit(self, record: TraceRecord) -> None:
        """分发记录；单个 Sink 异常不影响主流程。"""
        for sink in self._sinks:
            try:
                sink.emit(record)
            except Exception:  # pragma: no cover - 防御性分支
                logger.warning("Trace Sink 写入失败: %s", type(sink).__name__, exc_info=True)


_recorder: TraceRecorder | None = None


def get_trace_recorder() -> TraceRecorder:
    """按配置获取 Trace 记录器（懒加载单例）。"""
    global _recorder
    if _recorder is None:
        settings = get_settings().tracing
        sinks: list[TraceSink] = []
        if settings.enabled:
            if settings.sink == "memory":
                sinks.append(MemoryTraceSink(maxlen=settings.memory_buffer_size))
            else:
                sinks.append(LoggingTraceSink())
        _recorder = TraceRecorder(sinks)
    return _recorder


def set_trace_recorder(recorder: TraceRecorder | None) -> None:
    """注入自定义记录器（测试或应用启动时使用）。"""
    global _recorder
    _recorder = recorder


def _is_sensitive(key: str) -> bool:
    lowered = key.lower()
    return any(word in lowered for word in SENSITIVE_KEYWORDS)


def _to_jsonable(value: Any, depth: int = 0) -> Any:
    """把任意对象转成可 JSON 序列化的结构，并对敏感键名脱敏。"""
    if depth > 4:
        return "<max-depth>"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): (MASK if _is_sensitive(str(key)) else _to_jsonable(item, depth + 1))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_jsonable(item, depth + 1) for item in value]
    if isinstance(value, (bytes, bytearray)):
        return f"<bytes:{len(value)}>"
    return repr(value)[:200]


def _prepare_payload(value: Any, *, enabled: bool, max_chars: int) -> str | None:
    """脱敏 + 截断，返回用于记录的字符串。"""
    if not enabled or value is None:
        return None
    try:
        encoded = json.dumps(_to_jsonable(value), ensure_ascii=False, default=str)
    except (TypeError, ValueError):  # pragma: no cover - 兜底
        encoded = repr(value)
    if len(encoded) > max_chars:
        return f"{encoded[:max_chars]}...<truncated {len(encoded) - max_chars} chars>"
    return encoded


def _extract_arguments(
    func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    """把函数入参整理成字典（去掉 self / cls）。"""
    try:
        bound = inspect.signature(func).bind_partial(*args, **kwargs)
        arguments = dict(bound.arguments)
    except TypeError:  # pragma: no cover - 无法解析签名时退化为位置参数列表
        return {"args": list(args), "kwargs": kwargs}
    return {key: value for key, value in arguments.items() if key not in ("self", "cls")}


class ActiveSpan:
    """一个活跃的 Trace span，同时支持同步与异步上下文管理器。"""

    def __init__(
        self,
        name: str,
        *,
        kind: SpanKind = "internal",
        payload: Any = None,
        metadata: dict[str, Any] | None = None,
        capture_input: bool | None = None,
        capture_output: bool | None = None,
        trace_id: str | None = None,
        recorder: TraceRecorder | None = None,
    ) -> None:
        self.name = name
        self.kind = kind
        self.metadata: dict[str, Any] = dict(metadata or {})
        self._payload = payload
        self._output: Any = None
        self._explicit_trace_id = trace_id
        self._recorder = recorder

        settings = get_settings().tracing
        self._enabled = settings.enabled
        self._capture_input = settings.capture_input if capture_input is None else capture_input
        self._capture_output = settings.capture_output if capture_output is None else capture_output
        self._max_chars = settings.max_payload_chars

        self.trace_id = trace_id or new_trace_id()
        self.span_id = new_span_id()
        self.parent_span_id: str | None = None
        self.started_at = datetime.now(UTC)
        self._started_perf = time.perf_counter()
        self._token: contextvars.Token[TraceContext | None] | None = None

    # -- 记录补充 ---------------------------------------------------------
    def set_output(self, output: Any) -> None:
        """设置本次调用的输出（通常在函数返回后立即调用）。"""
        self._output = output

    def set_metadata(self, **metadata: Any) -> None:
        """追加自定义元数据，例如 model、retry、file_path、status_code。"""
        self.metadata.update(metadata)

    def update_input(self, payload: Any) -> None:
        """补充或覆盖输入负载。"""
        self._payload = payload

    @property
    def duration_ms(self) -> float:
        """当前已耗时（毫秒）。"""
        return (time.perf_counter() - self._started_perf) * 1000

    # -- 生命周期 ---------------------------------------------------------
    def __enter__(self) -> ActiveSpan:
        self._begin()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        self._end(exc)
        return False  # 不吞异常

    async def __aenter__(self) -> ActiveSpan:
        self._begin()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        self._end(exc)
        return False

    def _begin(self) -> None:
        parent = current_context()
        if parent is not None:
            self.parent_span_id = parent.span_id
            if not self._explicit_trace_id:
                self.trace_id = parent.trace_id
        self._token = _context.set(
            TraceContext(
                trace_id=self.trace_id,
                span_id=self.span_id,
                parent_span_id=self.parent_span_id,
            )
        )

    def _end(self, exc: BaseException | None) -> None:
        if self._token is not None:
            _context.reset(self._token)
            self._token = None

        record = TraceRecord(
            trace_id=self.trace_id,
            span_id=self.span_id,
            parent_span_id=self.parent_span_id,
            name=self.name,
            kind=self.kind,
            started_at=self.started_at,
            duration_ms=self.duration_ms,
            status="error" if exc else "ok",
            input=_prepare_payload(
                self._payload, enabled=self._capture_input, max_chars=self._max_chars
            ),
            output=_prepare_payload(
                self._output, enabled=self._capture_output, max_chars=self._max_chars
            ),
            error=f"{type(exc).__name__}: {exc}" if exc else None,
            metadata=dict(self.metadata),
        )

        if self._enabled:
            (self._recorder or get_trace_recorder()).emit(record)


def trace_span(
    name: str,
    *,
    kind: SpanKind = "internal",
    payload: Any = None,
    metadata: dict[str, Any] | None = None,
    capture_input: bool | None = None,
    capture_output: bool | None = None,
    trace_id: str | None = None,
    recorder: TraceRecorder | None = None,
) -> ActiveSpan:
    """创建 span：`with trace_span("retrieve_code", kind="rag") as span: ...`。"""
    return ActiveSpan(
        name,
        kind=kind,
        payload=payload,
        metadata=metadata,
        capture_input=capture_input,
        capture_output=capture_output,
        trace_id=trace_id,
        recorder=recorder,
    )


def traced(
    name: str | None = None,
    *,
    kind: SpanKind = "agent",
    capture_input: bool | None = None,
    capture_output: bool | None = None,
    metadata: dict[str, Any] | None = None,
    recorder: TraceRecorder | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """装饰器：自动记录函数入参、返回值、异常与耗时，支持 async / sync 函数。"""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        span_name = name or f"{func.__module__}.{func.__qualname__}"
        extra = {
            "kind": kind,
            "metadata": metadata,
            "capture_input": capture_input,
            "capture_output": capture_output,
            "recorder": recorder,
        }

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                span = ActiveSpan(
                    span_name, payload=_extract_arguments(func, args, kwargs), **extra
                )
                async with span:
                    result = await func(*args, **kwargs)
                    span.set_output(result)
                    return result

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            span = ActiveSpan(span_name, payload=_extract_arguments(func, args, kwargs), **extra)
            with span:
                result = func(*args, **kwargs)
                span.set_output(result)
                return result

        return sync_wrapper  # type: ignore[return-value]

    return decorator


class TraceIdMiddleware(BaseHTTPMiddleware):
    """HTTP 入口埋点：分配 trace_id、记录请求耗时，并回写响应头。"""

    def __init__(self, app: Any, header_name: str = "X-Trace-Id") -> None:
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        incoming = request.headers.get(self.header_name) or request.headers.get("traceparent")
        trace_id = incoming.strip() if incoming else new_trace_id()

        span = trace_span(
            f"{request.method} {request.url.path}",
            kind="http",
            payload={"path": request.url.path, "method": request.method},
            metadata={"client": request.client.host if request.client else None},
            trace_id=trace_id,
            capture_input=False,
            capture_output=False,
        )
        with span:
            response = await call_next(request)
            span.set_metadata(status_code=response.status_code)

        response.headers[self.header_name] = trace_id
        return response

