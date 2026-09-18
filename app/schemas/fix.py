"""代码改错模块的 API 契约。"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ChangeCategoryLiteral = Literal["syntax", "logic", "risk", "style"]
SeverityLiteral = Literal["error", "warning", "info"]


# ---------------------------------------------------------------------------
# 请求
# ---------------------------------------------------------------------------
class CodeFixRequest(BaseModel):
    """改错请求：一段有错误的代码 + 语言类型。"""

    code: str = Field(..., min_length=1, description="要修正的代码内容")
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
# 修改说明（四问结构 —— 需求里点名的四项）
# ---------------------------------------------------------------------------
class CodeChangeRead(BaseModel):
    """一处修改的说明。"""

    line: int | None = Field(default=None, description="所在行号；不确定时为 null")
    category: ChangeCategoryLiteral = Field(
        description="syntax=语法 / logic=逻辑 / risk=潜在风险 / style=风格"
    )
    what: str = Field(description="原来错在哪里")
    why: str = Field(default="", description="为什么错（会造成什么后果）")
    how: str = Field(default="", description="怎么改")
    avoid: str = Field(default="", description="以后如何避免")
    original: str = Field(default="", description="改前的代码（原文摘抄）")
    fixed: str = Field(default="", description="改后的代码")


# ---------------------------------------------------------------------------
# 本地复检
# ---------------------------------------------------------------------------
class FixSyntaxErrorRead(BaseModel):
    """语法错误的位置与说明。

    名字带 Fix 前缀是有原因的：检测模块里也有一个同结构的 `SyntaxErrorRead`，
    而 OpenAPI 的组件名是**按类名**生成的，重名会让两份 schema 一起被改写成
    `app__schemas__check__SyntaxErrorRead` 这种带模块路径的丑名字，
    直接展示在 /docs 页面上。改名比让学生看那种名字强。
    """

    line: int | None = None
    column: int | None = None
    message: str = ""
    tool: str = Field(default="", description="ast=Python 内置解析器，tree-sitter=C/Java")
    raw: str = ""


class DiffStatsRead(BaseModel):
    """修改前后的行数差异统计。"""

    added: int = 0
    removed: int = 0
    changed: int = 0
    unchanged: int = 0
    total_changed: int = Field(default=0, description="增+删+改的行数合计")


class VerificationRead(BaseModel):
    """本地复检结论。

    **`verified` 只代表语法层面**：逻辑正确性本地无法自动验证，
    具体边界写在 `note` 里，前端应把它一并展示，避免学生理解成"逻辑也对"。
    """

    verified: bool = Field(description="修正后的代码是否通过了本地语法复检")
    note: str = Field(default="", description="复检结论的人话说明（含验证边界）")
    syntax_before: FixSyntaxErrorRead | None = None
    syntax_after: FixSyntaxErrorRead | None = None


class LocalIssueRead(BaseModel):
    """本地静态检查发现的问题（改错时作为 AI 的输入）。"""

    line: int | None = None
    severity: SeverityLiteral = "info"
    category: str = ""
    title: str = ""
    detail: str = ""
    suggestion: str = ""
    source: str = "local"


# ---------------------------------------------------------------------------
# 响应
# ---------------------------------------------------------------------------
class CodeFixResponse(BaseModel):
    """一次改错的完整结果。"""

    filename: str
    language: str
    language_label: str = Field(description="展示名：C / Java / Python")

    # ---- 核心结果 ----
    had_error: bool = Field(description="是否真的发现了需要修改的错误")
    fixed_code: str = Field(description="修正后的完整代码（可直接复制运行）")
    changes: list[CodeChangeRead] = Field(
        default_factory=list, description="逐条修改说明（错在哪/为什么错/怎么改/以后如何避免）"
    )
    summary: str = Field(default="", description="一句话总体说明")
    categories: dict[str, int] = Field(
        default_factory=dict, description="各类错误的条数，例如 {\"logic\": 2}"
    )

    # ---- 对比学习用的差异 ----
    diff: str = Field(default="", description="统一 diff 文本，可逐行对比")
    diff_stats: DiffStatsRead = Field(default_factory=DiffStatsRead)

    # ---- 本地复检 ----
    verification: VerificationRead

    # ---- 本地分析（AI 的输入，也一并返回给学生看）----
    local_issues: list[LocalIssueRead] = Field(default_factory=list)
    local_duration_ms: float = 0.0

    # ---- 运行信息 ----
    ai_available: bool = Field(description="AI 是否真的参与了本次改错")
    model: str = ""
    note: str = Field(default="", description="需要额外告诉学生的说明")
    warnings: list[str] = Field(default_factory=list, description="降级/异常/矛盾提示")
    record_id: int | None = Field(default=None, description="落库后的记录 ID；未落库为 null")
    duration_ms: float = 0.0
    ai_duration_ms: float | None = None
    fixed_at: datetime
    trace_id: str | None = None


# ---------------------------------------------------------------------------
# 历史记录（对比学习）
# ---------------------------------------------------------------------------
class CodeFixRecordRead(BaseModel):
    """历史列表里的一条（不含代码全文，避免列表过重）。"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    language: str
    line_count: int
    had_error: bool
    change_count: int
    summary: str | None = None
    verified: bool
    ai_available: bool = False
    model: str | None = None
    added_lines: int = 0
    removed_lines: int = 0
    changed_lines: int = 0
    created_at: datetime


class CodeFixRecordDetail(CodeFixRecordRead):
    """历史详情：带上原代码、新代码、diff 与逐条说明，用于左右对比。"""

    original_code: str = ""
    fixed_code: str = ""
    changes: list[CodeChangeRead] = Field(default_factory=list)
    categories: dict[str, int] = Field(default_factory=dict)
    diff: str = ""
    verification_note: str | None = None
    syntax_error: str | None = None


class CodeFixHistoryResponse(BaseModel):
    """改错历史列表。"""

    total: int
    items: list[CodeFixRecordRead] = Field(default_factory=list)
