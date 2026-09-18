"""多模型与 API Key 相关的 API 契约。

安全约定（**改这个文件时务必守住**）：
所有响应模型里都**不能出现 api_key 字段**。用户在网页上填的 Key 只进不出：
进来时是请求体里的字符串，出去时只剩 `has_key: true/false` 这种布尔值。

换句话说：哪怕前端把自己的 Key 又原样读回去，也只能从它自己的输入框里读，
后端永远不会把 Key 回传。
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# 模型清单
# ---------------------------------------------------------------------------
class ModelOptionRead(BaseModel):
    """一个可选模型（给网页下拉框用），不含任何 Key。"""

    id: str = Field(description="模型 ID，选完把它传回来，如 deepseek")
    label: str = Field(description="网页上显示的名字，如「DeepSeek 官方」")
    provider: str = Field(description="提供商标识，如 deepseek / qwen / ollama")
    model_name: str = Field(description="发给服务商的模型名，如 deepseek-chat")
    base_url: str = Field(description="API 地址（只用于展示，前端不需要自己拼）")
    description: str = Field(default="", description="给学生的说明，例如去哪申请 Key")
    is_local: bool = Field(default=False, description="是否本地模型（不需要 Key）")
    requires_api_key: bool = Field(default=True, description="是否必须填 Key")
    has_default_key: bool = Field(
        default=False, description="后端是否已配好 Key（只回布尔值，不给 Key）"
    )
    is_default: bool = Field(default=False, description="是否是默认模型，前端可预选它")


class ModelListResponse(BaseModel):
    """`GET /api/v1/models/list` 的响应。

    额外带上"当前会话"的情况，前端据此决定要不要提示"请输入 Key"。
    """

    models: list[ModelOptionRead] = Field(default_factory=list)
    default_model_id: str = Field(default="", description="默认模型 ID")
    allow_client_key: bool = Field(
        default=True, description="后端是否允许前端传 Key（SECURITY__ALLOW_CLIENT_KEY）"
    )
    session: SessionStateRead | None = Field(
        default=None, description="当前会话的状态（没有会话时为 null）"
    )
    total: int = Field(default=0, description="可用模型个数")


# ---------------------------------------------------------------------------
# 会话状态（永远不含 Key）
# ---------------------------------------------------------------------------
class SessionStateRead(BaseModel):
    """某个会话当前的状态。**没有 api_key 字段，这是刻意的。**"""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(description="会话号，前端记住它（放 X-Session-Id 请求头）")
    models: list[str] = Field(default_factory=list, description="这个会话里存了哪些模型的 Key")
    default_model_id: str = Field(default="", description="最后一次设置的模型")
    has_key: bool = Field(default=False, description="会话里是否真的存着 Key")
    expires_at: datetime = Field(description="什么时候过期（每次使用会顺延）")
    last_used_at: datetime = Field(description="最后一次使用时间")
    use_count: int = Field(default=0, description="被用过多少次")


# ---------------------------------------------------------------------------
# 写入 / 清除 Key
# ---------------------------------------------------------------------------
class SetKeyRequest(BaseModel):
    """`POST /api/v1/auth/set_key` 的请求体。

    为什么 Key 走请求体而不是 URL 查询串：
    查询串会被 uvicorn 的访问日志原样打出来（`GET /x?api_key=sk-...`），
    而请求体不会。这是"Key 绝不进日志"的第一道保障。
    """

    api_key: str = Field(
        default="",
        max_length=512,
        description="用户自己的 API Key。本地模型（Ollama 等）可以留空",
    )
    model_id: str = Field(
        min_length=1,
        max_length=64,
        description="要用哪个模型，取 /models/list 里的 id",
    )
    session_id: str | None = Field(
        default=None,
        description="已有的会话号；留空表示新建一个会话（也可以放在 X-Session-Id 头里）",
    )


class SetKeyResponse(BaseModel):
    """存好之后的回应：只说"存下了"，不回 Key。"""

    session: SessionStateRead
    message: str = Field(default="", description="给用户看的中文提示")
    ttl_seconds: int = Field(default=0, description="有效期（秒），前端可以据此做倒计时")


class ClearKeyRequest(BaseModel):
    """`POST /api/v1/auth/clear_key` 的请求体。"""

    session_id: str | None = Field(
        default=None, description="要清哪个会话；也可以用 X-Session-Id 头带过来"
    )
    model_id: str | None = Field(
        default=None,
        description="只清某个模型的 Key；留空表示清掉整个会话（「退出」时用这个）",
    )


class ClearKeyResponse(BaseModel):
    """清除结果。"""

    cleared: bool = Field(description="是否真的清掉了（本来就是空的会返回 false）")
    session_id: str = Field(default="", description="被清的会话号（可能已经被删除）")
    message: str = Field(default="")
