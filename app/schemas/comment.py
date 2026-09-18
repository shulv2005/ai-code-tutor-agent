"""代码注释生成模块的 API 契约。"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# 请求
# ---------------------------------------------------------------------------
class CommentGenerateRequest(BaseModel):
    """注释生成请求：一段代码 + 语言类型。"""

    code: str = Field(..., min_length=1, description="要加注释的代码内容")
    language: str | None = Field(
        default=None,
        description=(
            "语言类型：c / java / python（大小写不敏感，也接受 py、python3 等写法）。"
            "留空时会按 filename 的后缀自动判断。"
        ),
    )
    filename: str | None = Field(
        default=None, max_length=255, description="文件名（可选），用于展示与语言推断"
    )
    # ---- 模型与 Key：网页「模型设置」里填的东西从这里进来 ----
    # 两个都不传 → 用后端 .env 里配好的默认模型（老用法完全不变）
    model_id: str | None = Field(
        default=None,
        max_length=64,
        description="用哪个模型（取自 /api/v1/models/list 的 id）；留空用默认模型",
    )
    api_key: str | None = Field(
        default=None,
        max_length=512,
        description=(
            "用户自己的 API Key（可选）。只用于本次请求，不落库、不进日志；"
            "推荐先用 /api/v1/auth/set_key 存进内存会话，再用 X-Session-Id 头带会话号"
        ),
    )


# ---------------------------------------------------------------------------
# 复检结论
# ---------------------------------------------------------------------------
class CommentVerificationRead(BaseModel):
    """本地复检结论：注释有没有把代码改坏、有没有漏注释。

    `code_unchanged` 是三个值而非布尔：true=确认未改动，false=确认被改动，
    null=无法判定（例如原代码本身有语法错误）。前端应按三态展示，
    不要把 null 当成"没问题"。
    """

    syntax_ok: bool = Field(description="加了注释后代码是否仍能解析")
    code_unchanged: bool | None = Field(
        default=None, description="代码逻辑是否未被改动；null 表示无法判定"
    )
    unchanged_note: str = Field(default="", description="代码比对的依据与结论")
    functions_total: int = Field(default=0, description="原代码里的函数/方法总数")
    functions_covered: int = Field(default=0, description="拿到注释的函数数")
    coverage_ratio: float = Field(default=0.0, description="函数注释覆盖率，0-1")
    file_comment: bool = Field(default=False, description="是否有文件级注释")
    comment_lines_before: int = Field(default=0, description="原代码的注释行数")
    comment_lines_after: int = Field(default=0, description="生成后的注释行数")
    added_comment_lines: int = Field(default=0, description="新增的注释行数")
    inline_comment_lines: int = Field(default=0, description="行内注释（代码后跟注释）的行数")
    verified: bool = Field(description="综合结论：语法通过 + 代码未被改动 + 确有新增注释")
    note: str = Field(default="", description="人话说明，含验证边界")


# ---------------------------------------------------------------------------
# 响应
# ---------------------------------------------------------------------------
class CommentGenerateResponse(BaseModel):
    """一次注释生成的完整结果。"""

    filename: str
    language: str
    language_label: str = Field(description="展示名：C / Java / Python")

    original_code: str = Field(description="学生原来的代码（原样返回，便于左右对比）")
    commented_code: str = Field(
        description="带注释的完整代码。任何失败路径下都会回退为原代码，不会返回空串"
    )
    summary: str = Field(default="", description="一句话说明加了哪些注释")

    verification: CommentVerificationRead

    ai_available: bool = Field(description="AI 是否真的参与了本次生成")
    model: str = ""
    note: str = Field(default="", description="需要额外告诉学生的说明")
    warnings: list[str] = Field(default_factory=list, description="降级/异常/风险提示")
    record_id: int | None = Field(default=None, description="落库后的记录 ID；未落库为 null")
    duration_ms: float = 0.0
    ai_duration_ms: float | None = None
    generated_at: datetime
    trace_id: str | None = None


# ---------------------------------------------------------------------------
# 历史记录
# ---------------------------------------------------------------------------
class CommentRecordRead(BaseModel):
    """历史列表里的一条（不含代码全文，避免列表过重）。"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    language: str
    original_lines: int
    commented_lines: int
    added_comment_lines: int
    functions_total: int
    functions_covered: int
    coverage_ratio: float
    file_comment: bool
    verified: bool
    code_unchanged: bool | None = None
    ai_available: bool = False
    model: str | None = None
    summary: str | None = None
    created_at: datetime


class CommentRecordDetail(CommentRecordRead):
    """历史详情：带上原代码与带注释的代码，用于左右对比。"""

    original_code: str = ""
    commented_code: str = ""
    verification_note: str | None = None
    syntax_ok: bool = False
    inline_comment_lines: int = 0
    duration_ms: float | None = None


class CommentHistoryResponse(BaseModel):
    """注释生成历史列表。"""

    total: int
    items: list[CommentRecordRead] = Field(default_factory=list)
