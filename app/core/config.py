"""配置管理：基于 pydantic-settings，支持 .env 与环境变量覆盖。

约定：
- 嵌套配置用双下划线分隔，例如 APP__DEBUG、DOCKER__TIMEOUT_SECONDS。
- 通过 get_settings() 获取单例；测试中可调用 get_settings.cache_clear() 重置。
- 大模型支持"单模型"与"多模型"两种写法，多模型用
  `LLM__MODELS__<ID>__<字段>`，详见 LLMSettings 的类注释。
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# app/core/config.py -> app/core -> app -> 项目根目录
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


def split_csv_list(value: object) -> object:
    """把环境变量里的列表写法统一成 list[str]。

    兼容：'a,b,c' / 'a, b' / '["a","b"]' -> ['a','b','c']

    为什么必须显式处理：pydantic-settings 默认会用 JSON 解析 list 型字段的原始值，
    `FOO=a,b` 会直接抛 SettingsError 让应用起不来。因此所有 list 型字段都要
    标 `Annotated[list[str], NoDecode]` 并挂本函数作为 before 校验器。
    """
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):  # 兼容 JSON 数组写法
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        return [item.strip() for item in text.split(",") if item.strip()]
    return value


class AppSettings(BaseModel):
    """应用自身的运行参数（供 FastAPI / uvicorn 使用）。"""

    name: str = "opensource-collab-agent"
    version: str = "0.1.0"
    environment: Literal["local", "dev", "staging", "prod"] = "local"
    debug: bool = True
    host: str = "127.0.0.1"
    port: int = 8000
    api_v1_prefix: str = "/api/v1"
    log_level: str = "INFO"
    # 注意：NoDecode 必须保留。pydantic-settings 默认会把 list 型字段的原始值
    # 当作 JSON 解析，导致 `APP__CORS_ORIGINS=a,b` 这种逗号写法直接报
    # SettingsError 并使应用启动失败；NoDecode 会跳过该步，交给下面的
    # _split_origins 校验器处理，从而同时兼容 `a,b` 与 `["a","b"]` 两种写法。
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"]
    )
    trace_header: str = "X-Trace-Id"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        """兼容多种写法：'a,b,c' / 'a, b' / '["a","b"]' -> ['a','b','c']。"""
        return split_csv_list(value)

    @field_validator("api_v1_prefix")
    @classmethod
    def _normalize_prefix(cls, value: str) -> str:
        return value if value.startswith("/") else f"/{value}"


class DatabaseSettings(BaseModel):
    """数据库配置：开发期 SQLite，生产期切换 PostgreSQL。"""

    sqlite_path: Path = PROJECT_ROOT / "data" / "app.db"
    # 非空时优先生效，例如 postgresql+asyncpg://user:pwd@host:5432/agent
    url_override: str = ""
    echo: bool = False

    @property
    def url(self) -> str:
        """返回 SQLAlchemy 异步连接串。"""
        if self.url_override:
            return self.url_override
        if str(self.sqlite_path) == ":memory:":
            return "sqlite+aiosqlite:///:memory:"
        path = self.sqlite_path.expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return f"sqlite+aiosqlite:///{path.as_posix()}"

    def ensure_directory(self) -> None:
        """确保 SQLite 文件所在目录存在（内存库跳过）。"""
        if self.url_override or str(self.sqlite_path) == ":memory:":
            return
        path = self.sqlite_path.expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 多模型配置：提供商默认地址与"本地模型"名单
# ---------------------------------------------------------------------------
# 常见 OpenAI 兼容服务商的默认 API 地址。
# 作用：同学在 .env 里只写了 provider 而没写 BASE_URL 时，自动补上，
# 少填一项就少一个填错的机会。地址都带 /v1（Ollama 的兼容层也是 /v1）。
PROVIDER_DEFAULT_BASE_URLS: dict[str, str] = {
    "deepseek": "https://api.deepseek.com/v1",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "openai": "https://api.openai.com/v1",
    "moonshot": "https://api.moonshot.cn/v1",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
    "siliconflow": "https://api.siliconflow.cn/v1",
    "ollama": "http://127.0.0.1:11434/v1",
    "vllm": "http://127.0.0.1:8001/v1",
    "lmstudio": "http://127.0.0.1:1234/v1",
}

# 本地部署的提供商：跑在自己电脑上，不需要 API Key。
# 名单里的模型在网页上不会强制要求填 Key（学生没网也能用本地模型演示）。
LOCAL_PROVIDERS: frozenset[str] = frozenset({"ollama", "vllm", "lmstudio", "local"})


class LLMModelSettings(BaseModel):
    """**一个**可选模型的配置（对应 `.env` 里的一组 `LLM__MODELS__<ID>__*`）。

    设计原则：能自动推出来的就不要让人填。
    比如写了 `PROVIDER=deepseek` 就不用再写 BASE_URL；
    写了 `PROVIDER=ollama` 就不要求填 API Key。
    """

    # 网页下拉框里显示的名字，例如「DeepSeek 官方」「通义千问」。留空则用 model_name
    label: str = ""
    # 提供商标识：deepseek / qwen / openai / ollama …（大小写不敏感）
    provider: str = "openai-compatible"
    # 真正发给服务商的模型名，例如 deepseek-chat、qwen-plus、qwen2.5-coder:7b
    model_name: str = ""
    # API 地址；留空则按 provider 自动补（见 PROVIDER_DEFAULT_BASE_URLS）
    base_url: str = ""
    # 默认 API Key（可选）。留空表示"用网页上输入的"或"用全局兜底 Key"
    api_key: SecretStr = SecretStr("")
    # 也可以让 Key 从**另一个环境变量**读（例如 OPENAI_API_KEY），
    # 这样密钥不必重复写进 .env 的这一段
    api_key_env: str = ""
    # 网页上的说明文字，例如「需要自己申请 Key，注册送额度」
    description: str = ""
    # 是否必须填 Key。None = 按 provider 自动判断（本地模型不需要）
    requires_api_key: bool | None = None
    # 关掉后网页上不显示这个模型（临时不想用某个模型时用，比删配置安全）
    enabled: bool = True
    # 以下三项留空（None）表示跟随全局 LLM__* 的默认值
    timeout_seconds: int | None = None
    max_tokens: int | None = None
    temperature: float | None = None

    # ---- 规整与推导 ----
    @field_validator("provider")
    @classmethod
    def _normalize_provider(cls, value: str) -> str:
        """提供商标识统一小写去空格，避免 DeepSeek / deepseek 被当成两个。"""
        return (value or "openai-compatible").strip().lower()

    @field_validator("label", "model_name", "base_url", "api_key_env", "description")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        """顺手去掉首尾空格：`.env` 里 `= ` 后面多打一个空格是很常见的。"""
        return (value or "").strip()

    @model_validator(mode="after")
    def _fill_defaults(self) -> LLMModelSettings:
        """补齐能自动推出来的字段，并做基本校验。"""
        if not self.model_name:
            raise ValueError("模型缺少 MODEL_NAME（例如 LLM__MODELS__DEEPSEEK__MODEL_NAME）")

        # 地址：没写就按 provider 查表；查不到就要求显式填写
        if not self.base_url:
            self.base_url = PROVIDER_DEFAULT_BASE_URLS.get(self.provider, "")
        if not self.base_url:
            raise ValueError(
                f"provider={self.provider} 没有内置默认地址，请显式填写 BASE_URL"
            )
        self.base_url = self.base_url.rstrip("/")

        # 是否需要 Key：没显式指定就按 provider / 地址判断
        if self.requires_api_key is None:
            self.requires_api_key = not self.is_local
        return self

    # ---- 展示用属性 ----
    @property
    def display_name(self) -> str:
        """网页上显示的名字：优先 LABEL，其次模型名。"""
        return self.label or self.model_name

    @property
    def is_local(self) -> bool:
        """是不是本地部署的模型（本地模型不需要 Key，也不该走公网）。"""
        if self.provider in LOCAL_PROVIDERS:
            return True
        # 地址指向本机时也当成本地模型（同学常把 vLLM 跑在 127.0.0.1:其他端口）
        host = self.base_url.lower()
        return any(mark in host for mark in ("127.0.0.1", "localhost", "0.0.0.0"))

    def to_public_dict(
        self, *, id: str, is_default: bool = False, has_default_key: bool = False
    ) -> dict[str, object]:
        """转成"可以安全发给前端"的字典——**绝不包含 API Key**。

        Args:
            id: 模型 ID（前端提交回来时用它）。
            is_default: 是不是默认模型（前端可以把它预选上）。
            has_default_key: 后端是否已有可用 Key（只回答"有没有"，不给值）。
        """
        return {
            "id": id,
            "label": self.display_name,
            "provider": self.provider,
            "model_name": self.model_name,
            "base_url": self.base_url,
            "description": self.description,
            "is_local": self.is_local,
            "requires_api_key": bool(self.requires_api_key),
            "has_default_key": has_default_key,
            "is_default": is_default,
        }


class LLMSettings(BaseModel):
    """大模型配置：默认走 OpenAI 兼容协议，便于接 DeepSeek-Coder / CodeQwen。

    支持两种写法，可以同时用：

    1. **单模型（旧写法，仍然兼容）**：`LLM__BASE_URL` / `LLM__MODEL` / `LLM__API_KEY`。
    2. **多模型（新写法）**：在 `.env` 里按模型各写一组

       ```ini
       LLM__MODELS__DEEPSEEK__LABEL=DeepSeek
       LLM__MODELS__DEEPSEEK__MODEL_NAME=deepseek-chat
       LLM__MODELS__DEEPSEEK__BASE_URL=https://api.deepseek.com/v1
       LLM__MODELS__DEEPSEEK__API_KEY=
       LLM__DEFAULT_MODEL=deepseek
       ```

       `LLM__MODELS__<ID>__*` 里的 `<ID>` 是模型标识（小写字母数字，如 deepseek / qwen / ollama），
       网页上传回来的就是这个 ID。

    两者同时存在时**以多模型为准**；但上面这几个单模型字段仍然有用：
    它们是"全局默认值"，多模型里没写的项（超时、max_tokens、temperature）会从这里继承，
    而 `LLM__API_KEY` 还兼任"某个模型没单独配 Key 时的兜底 Key"（详见 `effective_api_key`）。
    """

    provider: str = "openai-compatible"
    base_url: str = "https://api.deepseek.com/v1"
    api_key: SecretStr = SecretStr("")
    # 实测（2026-02，本机网络）：DeepSeek 会把请求的模型名统一路由到当前主力模型，
    # 请求 deepseek-coder / deepseek-chat 都返回 model=deepseek-flash。
    # 所以这里用官方通用的 deepseek-chat；换其它服务商时按它的文档填。
    model: str = "deepseek-chat"
    timeout_seconds: int = 120
    max_tokens: int = 4096
    temperature: float = 0.2
    # 429 / 5xx / 超时的重试次数与退避基数（指数退避）
    max_retries: int = 2
    retry_backoff_seconds: float = 1.0
    # 单次请求附加头（部分网关需要）
    extra_headers: dict[str, str] = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # 多模型
    # ------------------------------------------------------------------
    # 模型表：键是模型 ID（`.env` 里 `LLM__MODELS__<ID>__*` 的小写形式）
    models: dict[str, LLMModelSettings] = Field(default_factory=dict)
    # 默认用哪个模型（写模型 ID）。留空 = 用列表里的第一个；
    # 一个多模型都没配时，自动用上面的单模型字段拼一个 id 为 default 的模型。
    default_model: str = ""

    # ---- 校验：把用户写的大小写、空格、末尾斜杠都规整掉 ----
    @field_validator("provider")
    @classmethod
    def _normalize_provider(cls, value: str) -> str:
        return (value or "openai-compatible").strip().lower()

    @field_validator("base_url")
    @classmethod
    def _normalize_base_url(cls, value: str) -> str:
        """去掉末尾斜杠：拼接 `/chat/completions` 时才不会出现 `//`。"""
        return (value or "").strip().rstrip("/")

    @field_validator("models")
    @classmethod
    def _normalize_model_ids(
        cls, value: dict[str, LLMModelSettings]
    ) -> dict[str, LLMModelSettings]:
        """把模型 ID 规整成小写并去空格，同时挡掉非法 ID。

        为什么要规整：`.env` 里写 `LLM__MODELS__DeepSeek__*` 或 `LLM__MODELS__deepseek__*`
        都应该能用；前端提交的 id 也必须能对上，否则"选了模型却报找不到"。
        """
        normalized: dict[str, LLMModelSettings] = {}
        for raw_id, model in (value or {}).items():
            model_id = str(raw_id).strip().lower()
            if not model_id:
                raise ValueError("模型 ID 不能为空（写成 LLM__MODELS__<ID>__* 的形式）")
            # 只允许 ASCII：这个 ID 会长在环境变量名里（LLM__MODELS__<ID>__*），
            # 也可能被前端放进 URL 查询串，用中文/符号会带来一堆编码问题。
            # 注意不能只用 isalnum()：中文、日文等字符在 Python 里也算是 alnum。
            if not all(ch.isascii() and (ch.isalnum() or ch in "-_") for ch in model_id):
                raise ValueError(
                    f"模型 ID「{raw_id}」只允许 ASCII 字母、数字、下划线和连字符"
                    "（建议用小写英文，如 deepseek / qwen / ollama）"
                )
            normalized[model_id] = model
        return normalized

    @model_validator(mode="after")
    def _check_default_model(self) -> LLMSettings:
        """默认模型的 ID 必须真实存在，否则就直接报错。

        这类配置错误如果不当场拦下，会拖到"学生点了检测却没反应"的时候才暴露，
        排查成本高得多——启动时失败反而更好定位。
        """
        if self.default_model and self.default_model.strip().lower() not in self.model_table:
            raise ValueError(
                f"LLM__DEFAULT_MODEL={self.default_model} 在模型列表里不存在，"
                f"可选：{sorted(self.model_table) or '（一个都没配，用的是单模型写法）'}"
            )
        self.default_model = self.default_model.strip().lower()
        return self

    # ---- 模型表：多模型为空时，用单模型字段兜底拼一个，保证老配置照样能跑 ----
    @property
    def model_table(self) -> dict[str, LLMModelSettings]:
        """返回"当前真正可用的模型表"（永远不会是空的）。"""
        if self.models:
            return self.models
        # 单模型写法：包装成一个 id 为 default 的模型，让上层逻辑只有一套
        return {
            "default": LLMModelSettings(
                label=self.model,
                provider=self.provider,
                model_name=self.model,
                base_url=self.base_url,
                api_key=self.api_key,
                description="来自单模型配置（LLM__MODEL / LLM__BASE_URL / LLM__API_KEY）",
            )
        }

    @property
    def default_model_id(self) -> str:
        """默认使用的模型 ID（`LLM__DEFAULT_MODEL` 留空时取第一个）。"""
        table = self.model_table
        if self.default_model in table:
            return self.default_model
        return next(iter(table))

    @property
    def is_configured(self) -> bool:
        """是否配置了可用的 API Key（旧的单模型判断，保留给已有调用方）。"""
        return bool(self.api_key.get_secret_value().strip())

    @property
    def has_local_model(self) -> bool:
        """是否有本地部署的模型（本地模型不需要 Key，没网也能演示）。"""
        return any(model.is_local for model in self.model_table.values())

    def get_model(self, model_id: str | None = None) -> LLMModelSettings:
        """按 ID 取模型配置；ID 为空时取默认模型。

        Raises:
            KeyError: 指定的模型不存在（调用方应转成 400，并提示可选值）。
        """
        table = self.model_table
        wanted = (model_id or "").strip().lower() or self.default_model_id
        if wanted not in table:
            raise KeyError(f"没有名为 {model_id} 的模型，可选：{sorted(table)}")
        return table[wanted]

    def effective_api_key(self, model: LLMModelSettings, override: str = "") -> str:
        """算出某个模型这次调用真正要用的 API Key。

        优先级（从高到低）：
          1. `override`：用户在网页上现场输入的 Key —— 只用于本次请求，不落盘；
          2. 模型自己配的 `LLM__MODELS__<ID>__API_KEY`；
          3. 模型指定的 `API_KEY_ENV`（Key 放在别的环境变量里，不必写进 .env）；
          4. 全局 `LLM__API_KEY` 兜底 —— 方便"只配了一个 Key"的同学直接用上新模型。

        第 4 条**只对云端模型生效**：本地模型（Ollama / vLLM 等）不吃全局 Key，
        免得把云端的密钥顺手发给本机的服务，也免得前端误以为"这个模型已经配好了"。

        另外提醒一句：把"全局 Key"发给任意云端模型时，若该 Key 只对某个服务商有效，
        别的模型会返回 401 —— 这时给那个模型单独配一个 KEY 即可（页面上的报错会说清）。
        """
        override = (override or "").strip()
        if override:
            return override

        own = model.api_key.get_secret_value().strip()
        if own:
            return own

        if model.api_key_env:
            from_env = os.environ.get(model.api_key_env.strip(), "").strip()
            if from_env:
                return from_env

        if model.is_local:
            return ""
        return self.api_key.get_secret_value().strip()

    def resolved_settings(self, model_id: str | None = None, api_key: str = "") -> LLMSettings:
        """按模型 ID 生成一份"可以直接交给 LLM 客户端用"的单模型配置。

        这是多模型与现有代码之间的适配层：`OpenAICompatibleClient` 只认
        `base_url / model / api_key` 这几个字段，这里把它们换成所选模型的值，
        其余（超时、重试、extra_headers）保持全局默认，于是**客户端一行都不用改**。

        Args:
            model_id: 模型 ID；为空用默认模型。
            api_key: 用户在网页上输入的 Key（可空，空则按 effective_api_key 的规则取）。
        """
        model = self.get_model(model_id)
        return self.model_copy(
            update={
                "provider": model.provider,
                "base_url": model.base_url,
                "model": model.model_name,
                "api_key": SecretStr(self.effective_api_key(model, api_key)),
                "timeout_seconds": model.timeout_seconds or self.timeout_seconds,
                "max_tokens": model.max_tokens or self.max_tokens,
                "temperature": (
                    self.temperature if model.temperature is None else model.temperature
                ),
            }
        )

    def public_models(self) -> list[dict[str, object]]:
        """给前端用的模型清单：**不含任何 API Key**，只说明"能不能直接用"。

        Returns:
            形如 `[{"id": "deepseek", "label": "DeepSeek", "has_default_key": True, ...}]`
        """
        default_id = self.default_model_id
        options: list[dict[str, object]] = []
        for model_id, model in self.model_table.items():
            if not model.enabled:
                continue
            options.append(
                model.to_public_dict(
                    id=model_id,
                    is_default=model_id == default_id,
                    # 只看"后端有没有现成的 Key"，具体值绝不外传
                    has_default_key=bool(self.effective_api_key(model)),
                )
            )
        return options



class SecuritySettings(BaseModel):
    """安全相关配置：网页上输入的 API Key 怎么临时保管。

    背景：项目原本的 Key 只能写在 `.env` 里（一个服务一个模型）。现在支持
    同学在网页上选模型、填自己的 Key，这些 Key 只放在**进程内存**里，
    不落库、不进日志（详见 `app/core/api_key_manager.py` 的说明）。
    这里配的就是那份内存保管箱的行为。
    """

    # 网页上输入的 Key 多久没用就自动清掉（秒）。默认 30 分钟。
    # 每次使用都会顺延，所以是"无操作 30 分钟后失效"。
    api_key_ttl_seconds: int = 30 * 60
    # 内存里最多同时保留多少个会话。set_key 是无需登录就能调的接口，
    # 不设上限的话有人循环调用就能把内存吃满；超过上限会淘汰最久没用的会话。
    max_sessions: int = 500
    # 是否允许前端传 Key。默认允许（这是本功能的目的）；
    # 想强制"只能用 .env 里配好的 Key"时设 false，接口会直接拒绝 set_key。
    allow_client_key: bool = True

    @field_validator("api_key_ttl_seconds", "max_sessions")
    @classmethod
    def _positive_security_values(cls, value: int) -> int:
        """有效期与会话上限都必须是正数，否则保管箱会立刻清空或直接拒绝服务。"""
        if value <= 0:
            raise ValueError("安全配置里的有效期与会话上限必须为正数")
        return value


class RetrievalSettings(BaseModel):
    """混合检索配置：BM25 关键词召回 + 向量语义召回，再用 RRF 融合。

    关于嵌入模型的现实约束（实测结论，勿轻易改动默认值）：
    - huggingface.co 在部分网络下不可达，必须走镜像 HF_ENDPOINT。
    - huggingface-hub 1.x 默认使用 Xet 协议（cas-server.xethub.hf.co），
      镜像站不代理该域名会返回 401，必须设 HF_HUB_DISABLE_XET=1 回退到普通
      HTTP 下载，否则模型权重永远拉不下来。
    - 用 fastembed（ONNX Runtime）而不是 sentence-transformers，可省掉
      ~2GB 的 PyTorch 依赖。
    """

    vector_backend: Literal["faiss", "chroma"] = "faiss"
    index_dir: Path = PROJECT_ROOT / "data" / "index"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_dim: int = 512
    # 嵌入后端：fastembed=真语义（ONNX）；hashing=零依赖确定性哈希，用于离线/测试兜底
    embedder: Literal["fastembed", "hashing"] = "fastembed"
    hf_endpoint: str = "https://hf-mirror.com"
    hf_disable_xet: bool = True
    model_cache_dir: Path = PROJECT_ROOT / "data" / "models"
    embed_batch_size: int = 64

    top_k: int = 10
    # 每路召回量 = top_k * candidate_multiplier，再融合重排，提升召回率
    candidate_multiplier: int = 3
    bm25_weight: float = 0.5
    vector_weight: float = 0.5
    # RRF 平滑常数：越大则各路排名差异越被拉平（60 为常用默认值）
    rrf_k: int = 60
    # 单个符号送入嵌入模型的文本上限，防止超大函数拖慢索引
    max_chunk_chars: int = 2000
    # 检索时把测试符号排到实现代码之后（实测测试符号可占索引 60%+）
    prefer_non_test: bool = True

    @property
    def index_path(self) -> Path:
        """向量索引目录的绝对路径。"""
        path = self.index_dir.expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def cache_path(self) -> Path:
        """嵌入模型缓存目录的绝对路径。"""
        path = self.model_cache_dir.expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path


class DockerSettings(BaseModel):
    """沙箱执行配置：隔离运行 pytest 与 coverage.py。

    关于镜像与依赖的一个现实约束：
    `python:3.10-slim` 这类纯净基础镜像**不含 pytest / coverage**，
    因此要么在容器启动时 pip 安装（需要网络，会削弱隔离强度），
    要么预先构建好带依赖的镜像（推荐，可全程断网）。
    两个路径都支持，由 image + install_dependencies 组合决定。
    """

    # auto = 优先 docker，daemon 不可用时降级 local；docker / local = 强制指定
    backend: Literal["auto", "docker", "local"] = "auto"
    # 实际运行的镜像；生产建议换成预构建镜像并关闭 install_dependencies
    image: str = "python:3.10-slim"
    # 预构建镜像的基底，供 docker/sandbox.Dockerfile 使用
    base_image: str = "python:3.10-slim"
    # 容器启动时先 pip install pytest coverage（需要网络）
    install_dependencies: bool = True
    # pip 镜像源（国内网络建议设为 https://pypi.tuna.tsinghua.edu.cn/simple 等）
    pip_index_url: str = ""
    # 沙箱内运行测试的用户；留空用镜像默认。生产建议设非 root（如 1000:1000）
    sandbox_user: str = ""

    timeout_seconds: int = 300
    memory_limit: str = "1g"
    cpu_limit: float = 2.0
    pids_limit: int = 256
    network_disabled: bool = True
    read_only_root: bool = True
    # read_only_root 为真时可写的 tmpfs 挂载点与容量
    tmpfs_size: str = "64m"
    workspace_dir: Path = PROJECT_ROOT / "data" / "workspaces"
    workspace_mount: str = "/workspace"
    max_output_bytes: int = 1_000_000
    # 生成测试所在子目录（Step 4 的落盘位置）
    test_dir: str = "tests"

    @field_validator("cpu_limit")
    @classmethod
    def _positive_cpu(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("cpu_limit 必须为正数")
        return value

    @property
    def workspace_path(self) -> Path:
        """宿主机上的沙箱工作区绝对路径。"""
        path = self.workspace_dir.expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def needs_network_for_setup(self) -> bool:
        """安装依赖时必须放行网络（否则 pip 无法工作）。"""
        return self.install_dependencies and not self.pip_index_url.startswith("file:")



class TracingSettings(BaseModel):
    """Agent Trace 配置。"""

    enabled: bool = True
    sink: Literal["log", "memory"] = "log"
    capture_input: bool = True
    capture_output: bool = True
    max_payload_chars: int = 2000
    memory_buffer_size: int = 1000


class RepositorySettings(BaseModel):
    """仓库解析模块配置：克隆策略、安全边界与解析范围。"""

    workspace_dir: Path = PROJECT_ROOT / "data" / "repos"
    # 浅克隆深度；设为 0 表示完整克隆（需要完整 git 历史时用）
    clone_depth: int = 1
    clone_timeout_seconds: int = 300
    # 只允许从这些主机的 http(s) 地址克隆，防止 file:// 本地文件读取与 SSRF
    allowed_hosts: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["github.com", "gitee.com"]
    )
    # 仓库体积上限（MB），克隆后校验，超限拒绝索引
    max_repo_size_mb: int = 500
    # 单文件解析上限，超过则只记元信息、不做符号解析（避免解析超大生成文件）
    max_file_bytes: int = 512_000
    # 单仓库最多索引文件数，防止巨型仓库打爆 SQLite
    max_files: int = 5_000
    excluded_dirs: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
            "env", ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist",
            "build", ".next", "target", "vendor", "site-packages", ".idea", ".vscode",
        ]
    )

    @field_validator("allowed_hosts", "excluded_dirs", mode="before")
    @classmethod
    def _split_lists(cls, value: object) -> object:
        """见 split_csv_list 说明（NoDecode + 自定义拆分）。"""
        return split_csv_list(value)

    @field_validator("clone_depth")
    @classmethod
    def _non_negative_depth(cls, value: int) -> int:
        if value < 0:
            raise ValueError("clone_depth 不能为负数")
        return value

    @property
    def workspace_path(self) -> Path:
        """宿主机上的仓库克隆根目录（绝对路径）。"""
        path = self.workspace_dir.expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path


class LibrarySettings(BaseModel):
    """本地项目库配置：扫描学生本地文件夹并自动按语言分类。

    安全边界（很重要）：
    可扫描的目录**只能来自本配置文件**，接口里没有任何参数能指定任意路径。
    这样即使页面被同学随手改坏、或有人构造恶意请求，也无法读到配置之外的
    任何文件；配合 `resolve()` 后的一致性校验，`../` 之类的路径穿越也会被拒。

    写入类操作（在线编辑保存 / 替换 / 删除）遵守同一条边界：
    目标路径同样要 `resolve()` 后确认落在某个根目录之内，而且
    - 只能改**已存在**的源码文件（新建文件请走「拖进项目库」的入库流程）；
    - 删除默认是移到根目录下的 `.trash/`，不是抹掉；
    - 不想要这些能力时，用 `allow_write=False` 一次性关掉。
    """

    # 要扫描的根目录列表。为空时只用下面的默认目录（data/library）。
    # 想扫描自己的作业文件夹，在 .env 里写：
    #   LIBRARY__ROOTS=D:\我的作业, E:\课程代码
    roots: Annotated[list[str], NoDecode] = Field(default_factory=list)
    # 默认目录：学生把代码丢进这里就能一键看到，开箱即用
    default_dir: Path = PROJECT_ROOT / "data" / "library"
    # 随项目附带的示例作业目录（里面是故意写错的代码，用于第一次体验三个 AI 功能）
    example_dir: Path = PROJECT_ROOT / "examples"
    # 是否把示例目录也当作一个项目库来源。
    # 只在「没有显式配置 roots」时生效：老师配置了自己的作业目录后，
    # 就不该再混进项目自带的示例，否则每次都要在列表里翻找自己的文件。
    include_examples: bool = True
    # 目录递归深度上限，防止符号链接成环或目录层级过深导致扫描卡死
    max_depth: int = 6
    # 单次扫描最多收录多少个代码文件
    max_files: int = 500
    # 单个文件超过这个大小就只登记元信息、不读内容（避免误选了几十 MB 的数据文件）
    max_file_bytes: int = 512_000
    # 目录名黑名单：依赖目录、构建产物、版本库内部文件都不是学生的作业代码
    excluded_dirs: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
            "env", ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist",
            "build", ".next", "target", "vendor", "site-packages", ".idea", ".vscode",
            "cmake-build-debug", ".vs", "Debug", "Release", "x64",
            # 回收站：删除功能把文件移到这里。不排除的话，删掉的文件又会出现在列表里
            ".trash",
        ]
    )

    # ------------------------------------------------------------------
    # 写入类操作（在线编辑保存 / 替换 / 删除）
    # ------------------------------------------------------------------
    # 是否允许在页面上改文件。默认开启：这是给学生用的练习台，
    # 写错了还能改回来（删除是移到 .trash，不是抹掉）。
    # 如果项目库指向的是老师共享的公共目录，在 .env 里设
    # LIBRARY__ALLOW_WRITE=false，页面上的编辑/替换/删除按钮会自动禁用。
    allow_write: bool = True
    # 删除文件时先移到这个目录（相对每个项目库根目录），而不是直接抹掉：
    # 学生手滑点错了还能自己去把文件找回来。
    # 设成空字符串表示「直接彻底删除」，不保留回收站。
    # 注意：这个目录名同时被上面的 excluded_dirs 排除，不会出现在文件列表里。
    trash_dir_name: str = ".trash"

    @field_validator("roots", "excluded_dirs", mode="before")
    @classmethod
    def _split_lists(cls, value: object) -> object:
        """见 split_csv_list 说明（NoDecode + 自定义拆分）。"""
        return split_csv_list(value)

    @field_validator("max_depth", "max_files", "max_file_bytes")
    @classmethod
    def _positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("项目库的深度/文件数/大小上限必须为正数")
        return value

    @property
    def default_path(self) -> Path:
        """默认项目库目录的绝对路径。"""
        path = self.default_dir.expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def example_path(self) -> Path:
        """示例作业目录的绝对路径。"""
        path = self.example_dir.expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def root_paths(self) -> list[Path]:
        """实际要扫描的根目录（绝对路径、去重、保序）。

        相对路径按项目根目录解析，这样 `.env` 里写 `LIBRARY__ROOTS=我的作业`
        也能用；写绝对路径则原样使用。列表为空时回落到默认目录，
        并额外带上随项目附带的示例目录，保证第一次打开页面就有东西可看。
        """
        explicit = bool(self.roots)
        raw = self.roots or [str(self.default_path)]
        resolved: list[Path] = []
        for item in raw:
            path = Path(item).expanduser()
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            if path not in resolved:
                resolved.append(path)

        example = self.example_path
        if not explicit and self.include_examples and example.is_dir() and example not in resolved:
            resolved.append(example)
        return resolved


class ClassifierSettings(BaseModel):
    """本地代码文件自动分类（归档）配置。

    与 `LibrarySettings` 的区别，一句话说清：
      * 项目库（library）默认只浏览：原地扫描、只登记
        （编辑 / 替换 / 删除要用户主动点，见 LibrarySettings 的说明）；
      * 分类器（classifier）会**真的搬动文件**：把 .c/.h/.java/.py 移到
        `data/library/c`、`/java`、`/python` 下，并按语言归档。

    两者**默认指向同一个目录**（理由见下面 root 的注释）。

    因为会动文件，下面每一条限制都是为了"别把同学的作业弄丢"：
    不跟随软链接、不覆盖同名文件、目标目录必须落在根目录之内。
    """

    # 待分类的根目录。学生把散放的代码丢进这里，扫描后会被归到子目录。
    #
    # **刻意与「本地项目库」用同一个目录**（data/library），原因很实际：
    # 前端「拖进项目库」的流程是 上传 → /files/scan 按语言归档 → 刷新左侧列表，
    # 而左侧列表来自 /library/scan。如果这两个功能指向不同文件夹，
    # 学生刚拖进去的文件就**不会出现在列表里**，看起来像是没交上。
    # 用一个目录后：上传落到 data/library/，归档到 data/library/python/ 等子目录，
    # 而 /library/scan 是递归扫描的，归档后的文件依然在列表里（显示为 python/xxx.py），
    # 点一下还能直接把内容读回编辑器。一个文件夹，两条功能都通。
    root: Path = PROJECT_ROOT / "data" / "library"
    # 递归扫描子目录。关掉后只处理根目录下的文件（适合"我就放了一层"的情况）
    recursive: bool = True
    # 未知类型的文件是否也移到 unknown/。
    # 默认 False：只登记不移动。未知文件里可能混着别的重要资料，乱动风险太高。
    move_unknown: bool = False
    # 单次扫描最多处理多少个文件，防止误把整个磁盘拖进来
    max_files: int = 2_000
    # 单文件大小上限（超过则跳过并说明原因）。这里只是搬家、不读内容，所以给得比较宽松
    max_file_bytes: int = 5 * 1024 * 1024
    # 遍历时跳过的目录名：这些目录里的代码不是学生手写的作业
    excluded_dirs: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
            "env", ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist",
            "build", "target", "site-packages", ".idea", ".vscode", ".vs",
            # 回收站：项目库的「删除」把文件移到这里。**必须排除**，
            # 否则学生一往项目库拖文件，这次归档就会顺路走进 .trash，
            # 把刚删掉的文件又搬回 python/ 里（名字还带着回收站的时间戳前缀）。
            # 这个坑是真实使用中踩到的：删掉的作业又出现在列表里，学生一头雾水。
            ".trash",
        ]
    )

    @field_validator("excluded_dirs", mode="before")
    @classmethod
    def _split_lists(cls, value: object) -> object:
        """见 split_csv_list 说明（NoDecode + 自定义拆分）。"""
        return split_csv_list(value)

    @field_validator("max_files", "max_file_bytes")
    @classmethod
    def _positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("分类器的文件数与大小上限必须为正数")
        return value

    @property
    def root_path(self) -> Path:
        """分类根目录的绝对路径（相对路径按项目根目录解析）。"""
        path = self.root.expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path


class Settings(BaseSettings):
    """全局配置根对象。"""

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    app: AppSettings = Field(default_factory=AppSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    docker: DockerSettings = Field(default_factory=DockerSettings)
    tracing: TracingSettings = Field(default_factory=TracingSettings)
    repository: RepositorySettings = Field(default_factory=RepositorySettings)
    library: LibrarySettings = Field(default_factory=LibrarySettings)
    classifier: ClassifierSettings = Field(default_factory=ClassifierSettings)

    @model_validator(mode="after")
    def _protect_library_trash_from_classifier(self) -> Settings:
        """把项目库的回收站目录名补进分类器的遍历黑名单。

        为什么需要这一步：回收站目录名是可配置的（LIBRARY__TRASH_DIR_NAME），
        而分类器的黑名单是它自己的一份清单。两者一旦对不上，
        "往项目库拖一个文件"就会顺路走进回收站，把学生刚删掉的文件
        又搬回 c/ java/ python/ 里——名字还带着回收站的时间戳前缀。
        与其在两处各写一份、靠人记得同步，不如在这里由配置自己兜住。
        """
        trash = (self.library.trash_dir_name or "").strip()
        if trash and trash not in self.classifier.excluded_dirs:
            self.classifier.excluded_dirs.append(trash)
        return self

    def ensure_directories(self) -> None:
        """一次性创建运行期需要的本地目录。"""
        self.database.ensure_directory()
        self.retrieval.index_path.mkdir(parents=True, exist_ok=True)
        self.retrieval.cache_path.mkdir(parents=True, exist_ok=True)
        self.docker.workspace_path.mkdir(parents=True, exist_ok=True)
        self.repository.workspace_path.mkdir(parents=True, exist_ok=True)
        # 默认项目库目录：建出来学生才知道代码该放哪儿
        self.library.default_path.mkdir(parents=True, exist_ok=True)
        # 分类根目录 + 三个归档目录：先把"收纳盒"摆好，学生一看就明白会归到哪。
        # 目录名与 app/services/file_classifier.py 的 ARCHIVE_DIRS 保持一致；
        # 那边新增语言时，这里也要加上对应的目录名。
        self.classifier.root_path.mkdir(parents=True, exist_ok=True)
        for name in ("c", "java", "python"):
            (self.classifier.root_path / name).mkdir(exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取全局配置单例（带缓存）。"""
    return Settings()


def get_available_models(settings: Settings | None = None) -> list[dict[str, object]]:
    """列出当前配置里所有"可选用"的模型。

    这是给前端下拉框用的接口数据，因此有一条硬规矩：
    **返回值里绝不能出现 API Key**，只回答"后端有没有现成的 Key"（`has_default_key`）。
    用户在网页上输入的 Key 属于"这次请求"的临时数据，不参与这份清单。

    Args:
        settings: 可传入一份自定义配置（测试用）；不传则用全局单例。

    Returns:
        每个模型一个字典，字段如下：

        | 字段 | 说明 |
        | --- | --- |
        | `id` | 模型 ID（前端选完把它传回来，如 `deepseek`） |
        | `label` | 网页上显示的名字，如 `DeepSeek 官方` |
        | `provider` | 提供商标识，如 `deepseek` / `qwen` / `ollama` |
        | `model_name` | 发给服务商的模型名，如 `deepseek-chat` |
        | `base_url` | API 地址 |
        | `description` | 给学生的说明文字 |
        | `is_local` | 是否本地模型（本地模型不需要 Key） |
        | `requires_api_key` | 是否必须填 Key |
        | `has_default_key` | 后端是否已配好 Key（只给布尔值，不给 Key 本身） |
        | `is_default` | 是否是默认模型（前端可以预选它） |
    """
    current = settings or get_settings()
    return current.llm.public_models()

