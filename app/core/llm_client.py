"""通用 LLM 客户端：基于 openai 官方 SDK 的 OpenAI 兼容异步封装。

## 它解决什么问题

项目原来是"一个 .env 配一个模型"，客户端启动时就把 base_url / model / api_key
固定下来。现在要支持**在网页上选模型、填自己的 Key**，所以这里做成**通用客户端**：
每次调用都可以动态传入模型配置——用哪个模型、请求哪个地址、带哪把 Key。

## 三种构造方式（挑一种用）

```python
# 1) 直接给参数（最直观，网页传回来什么就用什么）
client = GenericLLMClient.for_model(
    model_name="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    api_key="sk-xxx",
)

# 2) 先用一份配置对象（想复用/改参数时方便）
config = LLMClientConfig(model_name=..., base_url=..., api_key=...)
client = GenericLLMClient(config)

# 3) 从项目配置里取（多模型 .env + 网页填的 Key，见 app/core/config.py）
config = LLMClientConfig.from_llm_settings(settings.llm, model_id="qwen", api_key="sk-web")
client = GenericLLMClient(config)
```

## 异常处理（每一类都对应一句人话）

| 情况 | 抛出的异常 | 给学生看的话 |
| --- | --- | --- |
| Key 是空的 | `LLMConfigError` | 请先在网页上输入 API Key |
| Key 无效 / 没权限（401/403） | `LLMAuthError` | 检查后重填（**不重试**，重试没用） |
| 被限流（429） | `LLMRateLimitError` | 请求太频繁，稍后再试（会先自动重试几次） |
| 超时 | `LLMTimeoutError` | 模型响应太慢（会先自动重试几次） |
| 连不上（断网 / 地址写错） | `LLMConnectionError` | 请检查网络与 base_url |
| 服务端 5xx | `LLMError` | 模型服务暂时不可用（会先自动重试几次） |
| 其它 4xx | `LLMError` | 请求有问题（消息里带 HTTP 状态码与原始说明） |
| 返回内容为空 / 格式不对 | `LLMResponseError` | 模型没有返回内容，请重试 |

所有异常都是 `LLMError` 的子类，调用方既可以统一 `except LLMError` 兜住，
也可以按类型分别提示。异常信息里**不会包含 API Key**（用 `mask()` 打码）。

## 为什么要用 openai 官方 SDK

- 它自带 SSE 解析，流式输出只要 `stream=True` 就能用（自己写要处理分包、`[DONE]`、异常中断）；
- 错误类型明确（`AuthenticationError` / `RateLimitError` / `APITimeoutError` …），
  可以精确映射成中文提示，而不是靠解析 HTTP 状态码猜；
- 仍然是 OpenAI 兼容协议：DeepSeek / 通义 / 硅基流动 / vLLM / Ollama / LM Studio
  都是换个 `base_url` 的事。

**重试仍由我们自己控制**：SDK 自带的重试被显式关掉（`max_retries=0`），
沿用项目原有的"指数退避 + 只重试可重试错误"策略，行为可测、可预测。
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import httpx
import openai
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from app.core.api_key_manager import mask
from app.core.config import LLMSettings, get_settings
from app.core.trace import trace_span

logger = logging.getLogger(__name__)

Role = Literal["system", "user", "assistant"]

# 可重试的 HTTP 状态码：429 限流 + 5xx 服务端错误 + 少数网关的"忙"
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

# Key 为空时统一用这句话提示（前端可以直接展示，需求里点名的文案）
MISSING_API_KEY_HINT = "请先在网页上输入 API Key（或在后端 .env 里配置 LLM__API_KEY）"

# 服务层"降级返回本地结论"时的完整提示：把"少了哪部分结论"说清楚。
# `{extra}` 由各服务填，例如「检测」「改错」「注释生成」。
MISSING_API_KEY_NOTE = (
    "AI {extra}未启用：请先在网页上输入 API Key"
    "（或在后端 .env 里配置 LLM__API_KEY）。以上是本地静态分析的结果；"
    "填好 Key 后再点一次，就能拿到模型给出的结论与评分。"
)

# 本地模型（Ollama / vLLM 等）不校验 Key，但 SDK 要求 api_key 非空，于是塞这个占位串。
# 它不会被任何服务端当成真 Key，也不会出现在日志里（日志只打 mask）。
LOCAL_PLACEHOLDER_KEY = "not-needed"


# ---------------------------------------------------------------------------
# 异常体系：让上层能区分"该重试"与"该让学生改配置"
# ---------------------------------------------------------------------------
class LLMError(RuntimeError):
    """LLM 调用基类异常。调用方 `except LLMError` 即可兜住全部情况。"""


class LLMConfigError(LLMError):
    """配置不完整：没填 API Key、base_url 为空、模型名为空等。"""


class LLMAuthError(LLMError):
    """鉴权失败（401/403）：Key 无效、过期或没有该模型的权限。"""


class LLMRateLimitError(LLMError):
    """限流（429）：已重试仍失败。"""


class LLMTimeoutError(LLMError):
    """请求超时。"""


class LLMConnectionError(LLMError):
    """连不上服务（断网、地址写错、代理问题）。"""


class LLMResponseError(LLMError):
    """响应格式不符合预期（缺少 choices、内容为空等）。"""


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class LLMMessage:
    """一条对话消息。

    Attributes:
        role: 角色，`system` 设定人设与规则 / `user` 用户输入 / `assistant` 模型回复。
        content: 文本内容。
    """

    role: Role
    content: str

    def to_payload(self) -> dict[str, str]:
        """转成 OpenAI 接口要的字典格式。"""
        return {"role": self.role, "content": self.content}


@dataclass(slots=True)
class LLMUsage:
    """token 用量（用于成本估算与 Trace 记录）。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> LLMUsage:
        """从接口返回的 `usage` 字段构造。"""
        payload = payload or {}
        return cls(
            prompt_tokens=int(payload.get("prompt_tokens", 0) or 0),
            completion_tokens=int(payload.get("completion_tokens", 0) or 0),
            total_tokens=int(payload.get("total_tokens", 0) or 0),
        )


@dataclass(slots=True)
class LLMResponse:
    """一次完整调用的结果。"""

    content: str                       # 模型输出的正文
    model: str = ""                    # 实际使用的模型名（服务端可能回不同名字）
    finish_reason: str | None = None   # stop=正常结束 / length=被 max_tokens 截断
    usage: LLMUsage = field(default_factory=LLMUsage)
    latency_ms: float = 0.0            # 本次调用耗时（含重试）
    attempts: int = 1                  # 实际请求了几次（>1 说明重试过）
    raw: dict[str, Any] = field(default_factory=dict)   # 原始响应，便于排查

    @property
    def truncated(self) -> bool:
        """是否因为达到 max_tokens 被截断（调用方据此提示"内容可能不完整"）。"""
        return self.finish_reason == "length"


@dataclass(frozen=True, slots=True)
class LLMStreamChunk:
    """流式输出的一小段。

    Attributes:
        delta: 这一次新增的文本（可能为空串，例如只带 finish_reason 的最后一帧）。
        finish_reason: 非空表示流结束（`stop` / `length`）。
        usage: 只在最后一帧出现（需要请求时带 `stream_options.include_usage`）。
        index: 第几段（从 0 开始），便于前端做进度展示。
    """

    delta: str
    finish_reason: str | None = None
    usage: LLMUsage | None = None
    index: int = 0


# ---------------------------------------------------------------------------
# 客户端配置：一次调用要用的全部参数
# ---------------------------------------------------------------------------
class LLMClientConfig(BaseModel):
    """一个 LLM 客户端的配置（**全部可以动态传入**）。

    这是"网页选模型 + 填 Key"落到客户端的载体：

    | 参数 | 作用 | 从哪来 |
    | --- | --- | --- |
    | `model_name` | 发给服务商的模型名，如 `deepseek-chat` | `.env` 的 `MODEL_NAME` |
    | `base_url` | API 地址，如 `https://api.deepseek.com/v1` | 同上（或按 provider 自动补） |
    | `api_key` | 用户自己的 Key | 网页输入（内存会话）或 `.env` |
    | `timeout_seconds` | 单次请求超时 | 全局默认，模型可覆盖 |
    | `max_tokens` | 输出上限，太小会把代码截断 | 全局默认，模型可覆盖 |
    | `temperature` | 随机性，写代码建议 0.2 左右 | 全局默认，模型可覆盖 |
    | `max_retries` | 可重试错误的重试次数 | 全局默认 |
    | `retry_backoff_seconds` | 退避基数（1s、2s、4s…） | 全局默认 |
    | `extra_headers` | 额外请求头（某些网关需要） | 全局默认 |
    | `requires_api_key` | 这个模型是否必须带 Key | 本地模型（Ollama）为 false |
    | `provider` | 提供商标识，只用于提示与排查 | `.env` 的 `PROVIDER` |
    """

    model_config = ConfigDict(extra="ignore")

    model_name: str = Field(min_length=1, description="模型名，如 deepseek-chat")
    base_url: str = Field(min_length=1, description="API 地址，如 https://api.deepseek.com/v1")
    api_key: SecretStr = Field(default=SecretStr(""), description="用户的 API Key（可空）")
    timeout_seconds: float = Field(default=120.0, gt=0, description="单次请求超时（秒）")
    max_tokens: int = Field(default=4096, gt=0, description="单次输出上限")
    temperature: float = Field(default=0.2, ge=0, le=2, description="采样温度")
    max_retries: int = Field(default=2, ge=0, description="可重试错误的重试次数")
    retry_backoff_seconds: float = Field(default=1.0, ge=0, description="退避基数（秒）")
    extra_headers: dict[str, str] = Field(default_factory=dict, description="额外请求头")
    requires_api_key: bool = Field(default=True, description="是否必须提供 Key")
    provider: str = Field(default="openai-compatible", description="提供商标识（仅用于提示）")

    @field_validator("model_name", "base_url", "provider")
    @classmethod
    def _strip(cls, value: str) -> str:
        """去掉首尾空格：网页/环境变量里的多余空格是最常见的低级错误。"""
        return (value or "").strip()

    @field_validator("base_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        """去掉末尾斜杠，否则会拼出 `//chat/completions`。"""
        return value.rstrip("/")

    # ---- 便捷属性 ----
    @property
    def has_api_key(self) -> bool:
        """用户/配置里到底有没有一把非空的 Key。"""
        return bool(self.api_key.get_secret_value().strip())

    @property
    def is_local(self) -> bool:
        """是不是本地模型（地址指向本机）；本地模型不需要真 Key。"""
        host = self.base_url.lower()
        return any(mark in host for mark in ("127.0.0.1", "localhost", "0.0.0.0"))

    @property
    def endpoint(self) -> str:
        """完整的对话补全地址（只用于日志与排查，SDK 自己会拼）。"""
        return f"{self.base_url}/chat/completions"

    def client_api_key(self) -> str:
        """交给 SDK 的 Key。

        - 正常情况：就是用户的 Key；
        - `requires_api_key=False`（本地模型）且 Key 为空：给一个占位串，
          因为 SDK 不允许 api_key 为空，而本地服务根本不校验它。
        """
        value = self.api_key.get_secret_value().strip()
        if value:
            return value
        if not self.requires_api_key:
            return LOCAL_PLACEHOLDER_KEY
        # 需要 Key 却没给：把决定权交给调用方（客户端会在发请求前抛出 LLMConfigError）
        return ""

    def masked_key(self) -> str:
        """给日志用的掩码形式（永远不打印明文）。"""
        return mask(self.api_key.get_secret_value())

    # ---- 与项目其它两部分的桥接 ----
    @classmethod
    def from_llm_settings(
        cls,
        settings: LLMSettings,
        model_id: str | None = None,
        api_key: str = "",
    ) -> LLMClientConfig:
        """从项目的 `LLMSettings` 生成客户端配置。

        这一步把前面两步的成果接起来：
        - `settings.get_model(model_id)` 取的是**多模型表**里的模型（第一步）；
        - `settings.effective_api_key(...)` 按"网页输入的 > 模型自带 > 指定环境变量 > 全局兜底"
          的优先级算出要用的 Key（第二步的规则）。

        Args:
            settings: 项目配置里的 `llm` 段。
            model_id: 用哪个模型；为空则用默认模型。
            api_key: 网页上传来的 Key（可空）。
        """
        model = settings.get_model(model_id)
        return cls(
            model_name=model.model_name,
            base_url=model.base_url,
            api_key=SecretStr(settings.effective_api_key(model, api_key)),
            timeout_seconds=float(model.timeout_seconds or settings.timeout_seconds),
            max_tokens=int(model.max_tokens or settings.max_tokens),
            temperature=(
                settings.temperature if model.temperature is None else model.temperature
            ),
            max_retries=settings.max_retries,
            retry_backoff_seconds=settings.retry_backoff_seconds,
            extra_headers=dict(settings.extra_headers),
            requires_api_key=bool(model.requires_api_key),
            provider=model.provider,
        )


# ---------------------------------------------------------------------------
# 协议：让上层（Agent / 服务）只依赖接口，测试可注入假客户端
# ---------------------------------------------------------------------------
@runtime_checkable
class LLMClient(Protocol):
    """对话补全客户端协议（项目里的 Agent 都按这个接口调用）。

    只要实现下面四个成员，就能顶替真实客户端——测试里的 `FakeLLMClient`
    就是这么做的，因此整条 Agent 链路可以完全离线跑。
    """

    @property
    def model(self) -> str:
        """模型名。"""
        ...

    @property
    def configured(self) -> bool:
        """是否已配置好（有 Key，或本身就是不需要 Key 的本地模型）。"""
        ...

    async def chat(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        """发起一次对话补全。"""
        ...

    async def aclose(self) -> None:
        """关闭底层连接池。"""
        ...


@runtime_checkable
class StreamingLLMClient(Protocol):
    """支持流式输出的客户端协议（可选能力，单独一个协议）。

    为什么不把 `stream_chat` 塞进 `LLMClient`：那样测试里的假客户端
    也必须实现它，会平白增加测试负担。需要流式的地方按这个协议判断即可。
    """

    def stream_chat(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        """流式返回一段段文本。"""
        ...


# ---------------------------------------------------------------------------
# 通用客户端
# ---------------------------------------------------------------------------
class GenericLLMClient:
    """通用 LLM 客户端：动态配置 + 异步调用 + 可选流式。

    它是无状态的（除了 SDK 连接池），所以既可以做成单例复用，
    也可以"每个请求 new 一个"——后者正是"网页填 Key"场景需要的：
    连接池用完即关，用户 Key 不留驻内存。
    """

    def __init__(
        self,
        config: LLMClientConfig | LLMSettings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """
        Args:
            config: 客户端配置。两种都接受：
                - `LLMClientConfig`（推荐，动态传入）；
                - `LLMSettings`（旧写法，会按单模型字段自动转换，保证老代码不改也能跑）。
            transport: httpx 传输层，**只给测试用**：传 `httpx.MockTransport`
                就能在没有网络、没有 Key 的情况下覆盖全部请求/重试/错误分支。
        """
        self._config = (
            LLMClientConfig.from_llm_settings(config)
            if isinstance(config, LLMSettings)
            else config
        )
        self._sdk: AsyncOpenAI | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._lock = asyncio.Lock()
        self._transport = transport

    # ---- 构造糖：直接给 model_name / base_url / api_key ----
    @classmethod
    def for_model(
        cls,
        model_name: str,
        base_url: str,
        api_key: str = "",
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        **options: Any,
    ) -> GenericLLMClient:
        """按"模型名 + 地址 + Key"直接建客户端（需求里点名的三个参数）。

        Args:
            model_name: 模型名，如 `deepseek-chat`。
            base_url: API 地址，如 `https://api.deepseek.com/v1`。
            api_key: 用户的 Key；本地模型可不填。
            transport: 测试用的 httpx 传输层。
            **options: 其余配置项（超时 / max_tokens / temperature / 重试次数 /
                extra_headers / requires_api_key / provider）。
        """
        return cls(
            LLMClientConfig(model_name=model_name, base_url=base_url,
                            api_key=SecretStr(api_key), **options),
            transport=transport,
        )

    # ---- 只读属性 ----
    @property
    def config(self) -> LLMClientConfig:
        """当前配置（`api_key` 是 SecretStr，直接打印只会看到星号）。"""
        return self._config

    @property
    def model(self) -> str:
        """模型名（Agent 会把它写进 Trace 与响应里）。"""
        return self._config.model_name

    @property
    def configured(self) -> bool:
        """是否可用：有 Key，或这是不需要 Key 的本地模型。"""
        return self._config.has_api_key or not self._config.requires_api_key

    @property
    def endpoint(self) -> str:
        """对话补全地址（排查问题时用）。"""
        return self._config.endpoint

    # ---- SDK 客户端：懒加载 + 复用 + 可关闭 ----
    async def _get_sdk(self) -> AsyncOpenAI:
        """懒加载 SDK 客户端（同一个客户端复用连接池，避免每次握三次手）。"""
        if self._sdk is None:
            async with self._lock:
                if self._sdk is None:
                    headers = {"Content-Type": "application/json"}
                    headers.update(self._config.extra_headers)
                    # 自己管重试，所以把 SDK 的重试关掉，避免"重试套重试"
                    self._sdk = AsyncOpenAI(
                        api_key=self._config.client_api_key() or LOCAL_PLACEHOLDER_KEY,
                        base_url=self._config.base_url,
                        timeout=self._config.timeout_seconds,
                        max_retries=0,
                        default_headers=headers,
                    )
                    if self._transport is not None:
                        # 测试注入点：把 SDK 的底层 HTTP 客户端换成 MockTransport
                        self._http_client = httpx.AsyncClient(transport=self._transport)
                        self._sdk = AsyncOpenAI(
                            api_key=self._config.client_api_key() or LOCAL_PLACEHOLDER_KEY,
                            base_url=self._config.base_url,
                            timeout=self._config.timeout_seconds,
                            max_retries=0,
                            default_headers=headers,
                            http_client=self._http_client,  # type: ignore[arg-type]
                        )
        return self._sdk

    async def aclose(self) -> None:
        """关闭连接池（可重复调用；关闭后再次使用会自动重建）。"""
        if self._sdk is not None:
            try:
                await self._sdk.close()
            except Exception:  # noqa: BLE001 - 关闭失败不该影响主流程
                logger.debug("关闭 openai 客户端失败", exc_info=True)
            self._sdk = None
        if self._http_client is not None:
            try:
                await self._http_client.aclose()
            except Exception:  # noqa: BLE001
                logger.debug("关闭底层 httpx 客户端失败", exc_info=True)
            self._http_client = None

    # ------------------------------------------------------------------
    # 一次性返回
    # ------------------------------------------------------------------
    async def chat(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        """发起一次对话补全（异步，含指数退避重试）。

        Args:
            messages: 对话消息列表，至少一条。
            model: 临时覆盖模型名（默认用配置里的）。
            temperature: 临时覆盖温度。
            max_tokens: 临时覆盖输出上限。
            stop: 停止词（例如生成代码时用 ``` 截断）。

        Returns:
            LLMResponse（正文、用量、耗时、实际尝试次数）。

        Raises:
            LLMConfigError: Key 为空（会提示"请先在网页上输入 API Key"）。
            LLMAuthError / LLMRateLimitError / LLMTimeoutError / LLMConnectionError /
            LLMResponseError / LLMError: 见模块头部那张表。
        """
        self._ensure_ready()

        payload: dict[str, Any] = {
            "model": model or self._config.model_name,
            "messages": [message.to_payload() for message in messages],
            "temperature": (
                self._config.temperature if temperature is None else temperature
            ),
            "max_tokens": (
                self._config.max_tokens if max_tokens is None else max_tokens
            ),
            "stream": False,
        }
        if stop:
            payload["stop"] = stop

        with trace_span(
            "llm.chat",
            kind="llm",
            payload={
                "model": payload["model"],
                # 提示词可能很长，截断由 Trace 层统一处理；此处只记规模
                "message_count": len(messages),
                "prompt_chars": sum(len(item.content) for item in messages),
            },
            metadata={"endpoint": self.endpoint, "provider": self._config.provider},
        ) as span:
            response = await self._request_with_retry(payload)
            span.set_metadata(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                attempts=response.attempts,
                finish_reason=response.finish_reason,
            )
            # 只记录输出规模，避免把整段代码写进 Trace 造成日志膨胀
            span.set_output({"chars": len(response.content), "model": response.model})
            return response

    async def _request_with_retry(self, payload: dict[str, Any]) -> LLMResponse:
        """带指数退避的重试：只重试"值得重试"的错误，鉴权/参数错误立刻抛出。"""
        attempts = self._config.max_retries + 1
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            started = time.perf_counter()
            try:
                sdk = await self._get_sdk()
                completion = await sdk.chat.completions.create(**payload)
                latency_ms = (time.perf_counter() - started) * 1000
                return self._parse(completion, latency_ms, attempt)

            except Exception as exc:  # noqa: BLE001 - 统一在这里翻译成中文提示
                mapped = self._map_error(exc)
                if not self._is_retryable(mapped) or attempt >= attempts:
                    raise mapped from exc
                last_error = mapped
                logger.warning(
                    "LLM 调用失败（第 %d/%d 次，模型=%s）：%s",
                    attempt,
                    attempts,
                    payload.get("model"),
                    mapped,
                )
                await self._sleep_backoff(attempt)

        raise last_error or LLMError("LLM 调用失败")

    async def _sleep_backoff(self, attempt: int) -> None:
        """指数退避：1s, 2s, 4s...（退避基数可配，测试里设为 0 就不等）"""
        await asyncio.sleep(self._config.retry_backoff_seconds * (2 ** (attempt - 1)))

    # ------------------------------------------------------------------
    # 流式输出
    # ------------------------------------------------------------------
    async def stream_chat(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        """流式发起对话补全：边生成边返回，适合"打字机"效果。

        用法：

        ```python
        async for chunk in client.stream_chat(messages):
            if chunk.delta:
                print(chunk.delta, end="", flush=True)
            if chunk.finish_reason:
                break
        ```

        Args:
            messages / model / temperature / max_tokens / stop: 同 `chat`。

        Yields:
            LLMStreamChunk：`delta` 是本段新增文本；最后一帧可能带 `finish_reason`
            与 `usage`。

        Raises:
            与 `chat` 相同的异常（含"请先在网页上输入 API Key"）。
            注意：流到一半断开时也会抛 `LLMConnectionError`，调用方应当
            保留已经收到的部分内容并提示"内容可能不完整"。
        """
        self._ensure_ready()

        payload: dict[str, Any] = {
            "model": model or self._config.model_name,
            "messages": [message.to_payload() for message in messages],
            "temperature": (
                self._config.temperature if temperature is None else temperature
            ),
            "max_tokens": (
                self._config.max_tokens if max_tokens is None else max_tokens
            ),
            "stream": True,
            # 让服务端在最后一帧带上 usage（不支持该参数的网关会忽略它）
            "stream_options": {"include_usage": True},
        }
        if stop:
            payload["stop"] = stop

        with trace_span(
            "llm.stream",
            kind="llm",
            payload={"model": payload["model"], "message_count": len(messages)},
            metadata={"endpoint": self.endpoint, "provider": self._config.provider},
        ) as span:
            chunks = 0
            chars = 0
            index = 0
            try:
                sdk = await self._get_sdk()
                stream = await sdk.chat.completions.create(**payload)
                async for event in stream:
                    chunk = self._to_chunk(event, index)
                    if chunk is None:
                        continue
                    index += 1
                    chunks += 1
                    chars += len(chunk.delta)
                    yield chunk
            except Exception as exc:  # noqa: BLE001
                raise self._map_error(exc) from exc
            finally:
                span.set_output({"chunks": chunks, "chars": chars})

    @staticmethod
    def _to_chunk(event: Any, index: int) -> LLMStreamChunk | None:
        """把 SDK 的流事件转成我们自己的 LLMStreamChunk；无内容无结束标记时返回 None。"""
        choices = getattr(event, "choices", None) or []
        delta_text = ""
        finish_reason: str | None = None
        if choices:
            choice = choices[0]
            delta = getattr(choice, "delta", None)
            if delta is not None:
                delta_text = _text_of(getattr(delta, "content", None))
            finish_reason = getattr(choice, "finish_reason", None)

        usage = None
        raw_usage = getattr(event, "usage", None)
        if raw_usage is not None:
            usage = LLMUsage(
                prompt_tokens=int(getattr(raw_usage, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(raw_usage, "completion_tokens", 0) or 0),
                total_tokens=int(getattr(raw_usage, "total_tokens", 0) or 0),
            )

        if not delta_text and not finish_reason and usage is None:
            return None      # 例如只带 role 的第一帧，前端不需要知道
        return LLMStreamChunk(
            delta=delta_text, finish_reason=finish_reason, usage=usage, index=index
        )

    # ------------------------------------------------------------------
    # 内部：前置检查、错误映射、响应解析
    # ------------------------------------------------------------------
    def _ensure_ready(self) -> None:
        """发请求前的检查：Key 有没有、地址对不对。

        这里是需求里那条"Key 为空就给出清晰提示"的落点——
        **在发请求之前**就抛错，既不浪费一次网络往返，也不会把空 Key
        发到服务商那里拿一个看不懂的 401。
        """
        if not self._config.model_name:
            raise LLMConfigError("没有指定模型名（model_name）")
        if not self._config.base_url:
            raise LLMConfigError("没有指定 API 地址（base_url）")
        if self._config.requires_api_key and not self._config.has_api_key:
            raise LLMConfigError(
                f"{MISSING_API_KEY_HINT}；模型「{self._config.model_name}」"
                f"（{self._config.provider}）需要 API Key"
            )

    def _is_retryable(self, error: Exception) -> bool:
        """判断这个错误值不值得重试。

        值得重试：限流、超时、连不上、服务端 5xx（都是"过一会儿可能就好"）。
        不值得重试：Key 无效、参数错误、内容为空（重试一百次也一样）。
        """
        return isinstance(
            error, (LLMRateLimitError, LLMTimeoutError, LLMConnectionError)
        ) or (isinstance(error, LLMError) and "HTTP 5" in str(error))

    def _map_error(self, exc: Exception) -> LLMError:
        """把 SDK / httpx 的异常翻译成带中文提示的 LLMError。

        映射表（顺序很重要：子类要排在父类前面）：
          AuthenticationError / PermissionDeniedError → LLMAuthError（不重试）
          RateLimitError                              → LLMRateLimitError（重试）
          APITimeoutError                             → LLMTimeoutError（重试）
          APIConnectionError                          → LLMConnectionError（重试）
          APIStatusError(5xx)                         → LLMError（重试）
          APIStatusError(其它 4xx)                    → LLMError（不重试，带状态码）
          APIResponseValidationError                  → LLMResponseError
        """
        # 已经是我们的异常（例如 _parse 抛的 LLMResponseError）就直接透传
        if isinstance(exc, LLMError):
            return exc

        key_state = "已填写" if self._config.has_api_key else "为空"

        if isinstance(exc, openai.AuthenticationError):
            return LLMAuthError(
                f"API Key 无效或已过期（HTTP {getattr(exc, 'status_code', 401)}）："
                f"当前 Key 状态={key_state}，请重新输入正确的 Key"
                f"（模型 {self._config.model_name}，{self._config.provider}）"
            )
        if isinstance(exc, openai.PermissionDeniedError):
            return LLMAuthError(
                f"这个 API Key 没有访问该模型的权限（HTTP {getattr(exc, 'status_code', 403)}）："
                f"请确认账号已开通「{self._config.model_name}」"
            )
        if isinstance(exc, openai.RateLimitError):
            return LLMRateLimitError(
                "请求太频繁或额度用完了（HTTP 429）：稍等一会儿再试，"
                "或换一个模型 / 换一把 Key"
            )
        if isinstance(exc, openai.APITimeoutError):
            return LLMTimeoutError(
                f"等待模型响应超时（超过 {self._config.timeout_seconds:g} 秒）："
                "网络较慢或模型太忙，可以重试，或在 .env 里调大 LLM__TIMEOUT_SECONDS"
            )
        if isinstance(exc, openai.APIConnectionError):
            return LLMConnectionError(
                f"连不上模型服务（{self._config.base_url}）："
                "请检查网络是否可用、地址是否写对；本地模型请确认服务已启动"
            )
        if isinstance(exc, openai.APIStatusError):
            status_code = getattr(exc, "status_code", 0)
            detail = _error_detail(exc)
            if status_code in _RETRYABLE_STATUS:
                return LLMError(f"HTTP {status_code}: {detail}")
            return LLMError(
                f"HTTP {status_code}: {detail}"
                + ("（请求参数有问题，改完再试）" if 400 <= status_code < 500 else "")
            )
        if isinstance(exc, openai.APIResponseValidationError):
            return LLMResponseError(f"模型返回的内容不符合预期格式：{exc}")

        # httpx 直接抛出的网络错误（例如 MockTransport / 自定义 http_client 场景）
        if isinstance(exc, httpx.TimeoutException):
            return LLMTimeoutError(f"等待模型响应超时：{type(exc).__name__}")
        if isinstance(exc, httpx.TransportError):
            return LLMConnectionError(
                f"连不上模型服务（{self._config.base_url}）：{type(exc).__name__}: {exc}"
            )
        if isinstance(exc, openai.OpenAIError):
            return LLMError(f"调用模型失败：{type(exc).__name__}: {exc}")

        # 兜底：解析响应时的 KeyError/TypeError 等
        return LLMResponseError(f"处理模型响应时出错：{type(exc).__name__}: {exc}")

    def _parse(self, completion: Any, latency_ms: float, attempt: int) -> LLMResponse:
        """把 SDK 返回的对象转成 `LLMResponse`。

        这里对返回内容**做宽容处理**，因为不同网关的实现不完全一致：
        - 正常情况 `content` 是字符串；
        - 有的网关（多模态 / 流式聚合）会返回"分段数组"，需要拼起来；
        - 推理模型（如 DeepSeek-R1）正文可能为空，推理内容在 `reasoning_content`。
        这几种在真实调用里都遇到过，所以统一在这里兜住。
        """
        choices = getattr(completion, "choices", None)
        if not choices:
            raise LLMResponseError(
                f"响应缺少 choices 字段（模型 {self._config.model_name} 没返回内容）"
            )

        first = choices[0]
        message = getattr(first, "message", None)
        content = _extract_content(message)
        if not content.strip():
            raise LLMResponseError(
                "模型返回了空内容：可能是提示词太长、被安全策略拦下，或模型临时异常，请重试"
            )

        usage_obj = getattr(completion, "usage", None)
        usage = LLMUsage(
            prompt_tokens=int(getattr(usage_obj, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage_obj, "completion_tokens", 0) or 0),
            total_tokens=int(getattr(usage_obj, "total_tokens", 0) or 0),
        )

        return LLMResponse(
            content=content,
            model=str(getattr(completion, "model", "") or self._config.model_name),
            finish_reason=getattr(first, "finish_reason", None),
            usage=usage,
            latency_ms=latency_ms,
            attempts=attempt,
            raw=_to_plain_dict(completion),
        )


# 旧名字：项目里其它模块一直按 `OpenAICompatibleClient` 导入，保留别名不破坏兼容。
# 两者是同一个类，改一处即可，不需要两套实现。
OpenAICompatibleClient = GenericLLMClient


# ---------------------------------------------------------------------------
# 解析辅助
# ---------------------------------------------------------------------------
def _text_of(value: Any) -> str:
    """把可能是"字符串 / 分段数组 / None"的内容统一成字符串。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(getattr(item, "text", "") or ""))
        return "".join(parts)
    return str(value)


def _extract_content(message: Any) -> str:
    """从响应消息里取出正文，并兼容三种真实遇到过的写法。

    1. `content` 是普通字符串（绝大多数情况）；
    2. `content` 是分段数组（部分网关/多模态）；
    3. `content` 为空但 `reasoning_content` 有内容（推理模型，DeepSeek-R1 等）。
    """
    if message is None:
        return ""
    if isinstance(message, dict):
        content = _text_of(message.get("content"))
        if content.strip():
            return content
        return _text_of(message.get("reasoning_content"))

    content = _text_of(getattr(message, "content", None))
    if content.strip():
        return content

    # SDK 把非标准字段放在 model_extra 里，推理模型的 reasoning_content 就在那儿
    extra = getattr(message, "model_extra", None) or {}
    reasoning = extra.get("reasoning_content") if isinstance(extra, dict) else None
    if reasoning:
        logger.debug("响应正文为空，回退使用 reasoning_content")
        return _text_of(reasoning)
    return ""


def _to_plain_dict(obj: Any) -> dict[str, Any]:
    """把 SDK 的响应对象转成普通 dict（放进 `LLMResponse.raw` 便于排查）。

    `warnings=False` 是必要的：有的网关把 `content` 返回成分段数组，
    与 SDK 的类型声明不符，pydantic 会在序列化时刷一堆 UserWarning。
    内容我们已经用 `_extract_content` 宽容处理过了，这里只是留个排查副本。
    """
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            data = dump(warnings=False)
            return data if isinstance(data, dict) else {"value": data}
        except TypeError:
            # 老版本 pydantic 没有 warnings 参数
            try:
                data = dump()
                return data if isinstance(data, dict) else {"value": data}
            except Exception:  # noqa: BLE001
                logger.debug("响应对象转 dict 失败", exc_info=True)
        except Exception:  # noqa: BLE001 - 转不出来也不能影响主流程
            logger.debug("响应对象转 dict 失败", exc_info=True)
    return {}


def _error_detail(exc: Exception) -> str:
    """从 SDK 异常里抠出"服务商说了什么"，并**抹掉可能回显的 Key**。

    为什么要抹：有些服务商在 401/400 的响应体里会把请求里的 Key 回显出来
    （`Incorrect API key provided: sk-...`）。这段文字我们要打日志、也要返回给
    前端看，所以统一过一遍脱敏。
    """
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        message = body.get("message") or body.get("error") or body
        detail = str(message)
    elif body is not None:
        detail = str(body)
    else:
        detail = str(exc)
    return mask_keys_in_text(detail[:300])


def mask_keys_in_text(text: str) -> str:
    """把文本里形如 `sk-xxxx` 的片段打码（用于错误信息脱敏）。

    只认"看起来像 Key"的片段：`sk-` / `sk_` / `key-` 开头、后面跟 8 个以上
    字母数字。这样既能把回显的 Key 抹掉，又不会把正常文字误伤成星号。
    """
    import re

    pattern = re.compile(r"\b(sk[-_][A-Za-z0-9._\-]{8,}|key[-_][A-Za-z0-9._\-]{8,})")
    return pattern.sub("***", text)


# ---------------------------------------------------------------------------
# 进程内单例（给 Agent 用）与按需构造（给"网页填 Key"用）
# ---------------------------------------------------------------------------
# 单例：httpx 连接池复用需要跨请求共享。
# 这里**只缓存用 .env 配置构造的客户端**，绝不缓存带用户 Key 的客户端——
# 否则用户的 Key 会跟着连接池一直留在内存里，违背"用完即弃"。
_shared: dict[str, LLMClient] = {}


def get_llm_client() -> LLMClient:
    """获取进程内共享的 LLM 客户端（按 `.env` 里的默认模型构造）。

    用途：Agent（测试生成 / 修复 / 导师）走这条路径，连接池跨请求复用。
    需要"用某个用户自己填的 Key 调一次"时，请用 `dynamic_llm_client()`。
    """
    settings = get_settings().llm
    key = f"{settings.base_url}|{settings.model}"
    client = _shared.get(key)
    if client is None:
        client = GenericLLMClient(LLMClientConfig.from_llm_settings(settings))
        _shared[key] = client
    return client


def set_llm_client(client: LLMClient | None) -> None:
    """注入自定义客户端（测试用）。传 None 清空。"""
    _shared.clear()
    if client is not None:
        settings = get_settings().llm
        _shared[f"{settings.base_url}|{settings.model}"] = client


def build_llm_client(
    config: LLMClientConfig | LLMSettings,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> GenericLLMClient:
    """按一份配置新建客户端（**不进单例**，用完请 `await client.aclose()`）。

    什么时候用它：用户在网页上选了某个模型、填了自己的 Key，
    这时候必须临时构造一个客户端，而不是复用全局单例。
    """
    return GenericLLMClient(config, transport=transport)


@asynccontextmanager
async def dynamic_llm_client(
    config: LLMClientConfig | LLMSettings,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> AsyncIterator[GenericLLMClient]:
    """「临时用一下」的客户端上下文管理器：退出时自动关闭连接池。

    推荐在接口里这样写（用户 Key 只在这段代码里存在）：

    ```python
    config = LLMClientConfig.from_llm_settings(settings.llm, model_id, api_key)
    async with dynamic_llm_client(config) as client:
        response = await client.chat(messages)
    ```
    """
    client = GenericLLMClient(config, transport=transport)
    try:
        yield client
    finally:
        await client.aclose()


@asynccontextmanager
async def llm_client_for_request(
    settings: LLMSettings,
    *,
    session_id: str | None = None,
    model_id: str | None = None,
    api_key: str | None = None,
    fallback: LLMClient | None = None,
) -> AsyncIterator[LLMClient]:
    """**服务层统一入口**：按"本次请求选的模型 + 用户自己的 Key"给出可用客户端。

    要解决的就是"服务不该只认 .env 里那把 Key"这件事：
    用户在网页上选了模型、填了自己的 Key 之后，深度检测 / 改错 / 注释生成
    也必须用**他的**配置去调模型。

    三种情况：
      1. 三个参数一个都没给 → 交回 `fallback`（默认是全局单例，即 `.env` 配置）。
         老用法与单元测试里注入的假客户端都走这条路，行为完全不变。
      2. 给了会话号 / 模型 / Key → 从内存会话里解析出凭据，现造一个客户端，
         **用完即关**（`dynamic_llm_client`），用户 Key 不驻留内存。
      3. 解析出来发现没有 Key（会话过期、用户没填）→ 返回的客户端
         `configured` 为 False；服务层据此走"只给本地结论 + 明确提示"的降级路径，
         真去调 `chat()` 则会抛 `LLMConfigError`，消息就是那句
         「请先在网页上输入 API Key…」。

    Args:
        settings: 项目配置里的 `llm` 段（提供模型表与默认值）。
        session_id: 网页带来的会话号（`X-Session-Id`）。
        model_id: 网页下拉框选中的模型 ID。
        api_key: 本次请求直接带来的 Key（优先级最高）。
        fallback: 没有动态配置时用的客户端；不传则用全局单例。
    """
    if not (session_id or model_id or api_key):
        yield fallback if fallback is not None else get_llm_client()
        return

    # 延迟导入：api_key_manager 与本模块互相引用，放在函数里避免循环导入
    from app.core.api_key_manager import get_api_key_manager

    credential = get_api_key_manager().resolve(session_id, model_id, api_key)
    config = LLMClientConfig.from_llm_settings(
        settings, credential.model_id, credential.api_key
    )
    logger.info(
        "本次请求使用模型=%s（Key 来源=%s，Key=%s）",
        config.model_name,
        credential.source,
        config.masked_key(),
    )
    async with dynamic_llm_client(config) as client:
        yield client


async def close_llm_clients() -> None:
    """关闭所有共享客户端（应用关闭时调用）。"""
    for client in list(_shared.values()):
        try:
            await client.aclose()
        except Exception:  # noqa: BLE001 - 关闭失败不应影响退出
            logger.debug("关闭 LLM 客户端失败", exc_info=True)
    _shared.clear()


__all__ = [
    "LOCAL_PLACEHOLDER_KEY",
    "MISSING_API_KEY_HINT",
    "MISSING_API_KEY_NOTE",
    "GenericLLMClient",
    "LLMAuthError",
    "LLMClient",
    "LLMClientConfig",
    "LLMConfigError",
    "LLMConnectionError",
    "LLMError",
    "LLMMessage",
    "LLMRateLimitError",
    "LLMResponse",
    "LLMResponseError",
    "LLMStreamChunk",
    "LLMTimeoutError",
    "LLMUsage",
    "OpenAICompatibleClient",
    "StreamingLLMClient",
    "build_llm_client",
    "close_llm_clients",
    "dynamic_llm_client",
    "get_llm_client",
    "llm_client_for_request",
    "mask_keys_in_text",
    "set_llm_client",
]

