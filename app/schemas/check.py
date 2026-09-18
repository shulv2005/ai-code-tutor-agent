"""AI 自动检测模块的 API 契约。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

SeverityLiteral = Literal["error", "warning", "info"]
CategoryLiteral = Literal["syntax", "logic", "style", "risk"]


# ---------------------------------------------------------------------------
# 请求
# ---------------------------------------------------------------------------
class CodeCheckRequest(BaseModel):
    """检测请求：一段代码 + 语言类型。"""

    code: str = Field(..., min_length=1, description="要检测的代码内容")
    language: str | None = Field(
        default=None,
        description=(
            "语言类型：c / java / python（大小写不敏感，也接受 py、python3 等写法）。"
            "留空时会按 filename 的后缀自动判断。"
        ),
    )
    filename: str | None = Field(
        default=None,
        max_length=255,
        description="文件名（可选）。用于展示，也用于在 language 为空时推断语言。",
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
            "用户自己的 API Key（可选）。**只用于本次请求，不落库、不进日志**；"
            "一般不直接传它，而是先用 /api/v1/auth/set_key 存进内存会话，"
            "再用 X-Session-Id 请求头把会话号带过来"
        ),
    )


# ---------------------------------------------------------------------------
# 本地静态检查
# ---------------------------------------------------------------------------
class SyntaxErrorRead(BaseModel):
    """语法错误的精确位置。"""

    line: int | None = Field(default=None, description="出错行号（从 1 开始）")
    column: int | None = Field(default=None, description="出错列号（从 1 开始）")
    message: str = Field(description="错误说明")
    tool: str = Field(description="结论来源：ast=Python 内置解析器，tree-sitter=C/Java 语法树")
    raw: str = Field(description="原始错误信息，便于排查工具本身的问题")


class CodeMetricsRead(BaseModel):
    """代码统计指标。"""

    total_lines: int = 0
    code_lines: int = 0
    comment_lines: int = 0
    blank_lines: int = 0
    comment_ratio: float = Field(default=0.0, description="注释行占比，0-1")
    max_line_length: int = 0
    long_lines: int = 0
    trailing_whitespace_lines: int = 0
    mixed_indent: bool = Field(default=False, description="是否混用了 Tab 与空格缩进")
    uses_tabs: bool = False
    function_count: int = 0
    class_count: int = 0
    max_complexity: int = Field(default=0, description="最高圈复杂度（分支越多越大）")
    longest_function: int = Field(default=0, description="最长函数的行数")
    longest_function_name: str = ""
    symbols: list[dict[str, Any]] = Field(
        default_factory=list, description="函数/类清单（名称、起止行、复杂度）"
    )


class LocalCheckRead(BaseModel):
    """阶段 1（本地静态检查）的结果。**这一部分永远有值，不依赖大模型**。"""

    language: str
    syntax_ok: bool = Field(description="语法是否通过")
    syntax_error: SyntaxErrorRead | None = None
    metrics: CodeMetricsRead = Field(default_factory=CodeMetricsRead)
    duration_ms: float = Field(default=0.0, description="本地检查耗时（毫秒）")


# ---------------------------------------------------------------------------
# 问题条目
# ---------------------------------------------------------------------------
class CodeIssueRead(BaseModel):
    """一条检测出问题。本地规则与 AI 结论共用同一结构，前端可统一渲染。"""

    line: int | None = Field(default=None, description="所在行号；不确定时为 null")
    severity: SeverityLiteral = Field(description="error=会出错 / warning=有隐患 / info=可以更好")
    category: CategoryLiteral = Field(
        description="syntax=语法 / logic=逻辑 / style=风格 / risk=潜在风险"
    )
    title: str = Field(description="一句话说明问题")
    detail: str = Field(default="", description="为什么这是问题")
    suggestion: str = Field(default="", description="怎么改")
    source: Literal["local", "ai"] = Field(
        description="结论来源：local=本地解析/规则，ai=大模型。便于学生分辨依据"
    )


# ---------------------------------------------------------------------------
# 响应
# ---------------------------------------------------------------------------
class CodeCheckResponse(BaseModel):
    """一次自动检测的完整结果。"""

    filename: str
    language: str
    language_label: str = Field(description="展示名：C / Java / Python")

    # ---- 评分 ----
    score: float = Field(ge=0, le=100, description="综合评分 0-100")
    level: str = Field(description="等级：优秀 / 良好 / 及格 / 待改进")
    score_reason: str = Field(description="为什么是这个分（评分规则的说明）")
    ai_score: float | None = Field(
        default=None, description="AI 给出的原始分数；未启用 AI 时为 null"
    )

    # ---- 语法 ----
    syntax_ok: bool

    # ---- 四类检测结果（对应需求里的四项输出）----
    errors: list[CodeIssueRead] = Field(
        default_factory=list, description="语法错误与逻辑错误"
    )
    style: list[CodeIssueRead] = Field(
        default_factory=list, description="代码风格建议（命名、缩进、注释）"
    )
    risks: list[CodeIssueRead] = Field(
        default_factory=list, description="潜在的 Bug 或风险点"
    )
    advice: list[str] = Field(
        default_factory=list, description="面向学生的学习建议（通俗语言）"
    )
    highlights: list[str] = Field(default_factory=list, description="做得好的地方")

    summary: str = Field(default="", description="一句话总体评价")

    # ---- 本地检查明细 ----
    local: LocalCheckRead

    # ---- 运行信息 ----
    ai_available: bool = Field(description="AI 深度检测是否真的参与了本次检测")
    model: str = ""
    note: str = Field(default="", description="需要额外告诉学生的说明，例如 AI 未启用")
    warnings: list[str] = Field(default_factory=list, description="降级/异常提示")
    record_id: int | None = Field(default=None, description="落库后的记录 ID；未落库为 null")
    duration_ms: float = 0.0
    ai_duration_ms: float | None = None
    checked_at: datetime
    trace_id: str | None = None
