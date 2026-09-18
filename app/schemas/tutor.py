"""AI 代码导师模块的 API 契约。"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# 前端可触发的三种 AI 操作
TutorActionLiteral = Literal["check", "comment", "fix"]
SeverityLiteral = Literal["error", "warning", "info"]


# ---------------------------------------------------------------------------
# 代码分析（不依赖大模型，纯本地解析）
# ---------------------------------------------------------------------------
class CodeSymbolRead(BaseModel):
    """代码里的一个函数/类。"""

    model_config = ConfigDict(from_attributes=True)

    name: str
    qualified_name: str
    kind: str
    start_line: int
    end_line: int
    signature: str = ""
    complexity: int = 1


class AnalyzeResponse(BaseModel):
    """上传代码后的分析结果：自动识别语言 + 解析结构。"""

    filename: str
    language: str = Field(description="识别出的语言标识，如 c / java / python")
    language_label: str = Field(description="面向展示的语言名称，如 C / Java / Python")
    line_count: int
    char_count: int
    size_bytes: int
    parse_error: str | None = Field(default=None, description="解析失败原因，正常时为 null")
    symbols: list[CodeSymbolRead] = Field(default_factory=list)
    imports: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    trace_id: str | None = None


# ---------------------------------------------------------------------------
# AI 检测
# ---------------------------------------------------------------------------
class CheckIssue(BaseModel):
    """一条检测出的问题。"""

    line: int | None = Field(default=None, description="所在行号，不确定时为 null")
    severity: SeverityLiteral = "info"
    title: str = Field(description="一句话说明问题")
    detail: str = Field(default="", description="为什么这是问题")
    suggestion: str = Field(default="", description="怎么改")


class CheckResponse(BaseModel):
    """AI 检测结果。"""

    filename: str
    language: str
    score: float = Field(ge=0, le=100, description="代码质量评分 0-100")
    level: str = Field(description="等级：优秀 / 良好 / 及格 / 待改进")
    summary: str = Field(default="", description="总体评价")
    issues: list[CheckIssue] = Field(default_factory=list)
    highlights: list[str] = Field(default_factory=list, description="做得好的地方")
    model: str = ""
    record_id: int | None = None
    trace_id: str | None = None


# ---------------------------------------------------------------------------
# 生成注释
# ---------------------------------------------------------------------------
class CommentResponse(BaseModel):
    """AI 注释生成结果。"""

    filename: str
    language: str
    commented_code: str = Field(description="带中文注释的完整代码")
    summary: str = Field(default="", description="注释了哪些地方")
    model: str = ""
    record_id: int | None = None
    trace_id: str | None = None


# ---------------------------------------------------------------------------
# 自动改错
# ---------------------------------------------------------------------------
class CodeChange(BaseModel):
    """一处修改说明。"""

    line: int | None = None
    original: str = Field(default="", description="原来的写法")
    fixed: str = Field(default="", description="修改后的写法")
    reason: str = Field(default="", description="为什么要这样改")


class FixResponse(BaseModel):
    """AI 改错结果。"""

    filename: str
    language: str
    fixed_code: str = Field(description="修正后的完整代码")
    changes: list[CodeChange] = Field(default_factory=list)
    summary: str = Field(default="", description="总体说明")
    had_error: bool = Field(default=True, description="是否真的发现了需要修的错误")
    model: str = ""
    record_id: int | None = None
    trace_id: str | None = None


# ---------------------------------------------------------------------------
# 历史记录
# ---------------------------------------------------------------------------
class TutorRecordRead(BaseModel):
    """一条历史记录。"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    language: str
    action: str
    line_count: int
    score: float | None = None
    summary: str | None = None
    model: str | None = None
    duration_ms: float | None = None
    created_at: datetime


class TutorHistoryResponse(BaseModel):
    """历史记录列表。"""

    total: int
    items: list[TutorRecordRead] = Field(default_factory=list)


class TutorRecordDetail(TutorRecordRead):
    """历史记录详情：额外带上原始代码与完整结果。"""

    code: str = ""
    result_json: str | None = None


class TutorStatusResponse(BaseModel):
    """AI 导师模块的可用状态，供前端提示学生。"""

    ai_available: bool = Field(description="是否已配置大模型；False 时 AI 按钮不可用")
    model: str = ""
    supported_languages: dict[str, str] = Field(
        default_factory=dict, description="语言标识 -> 展示名"
    )
    max_upload_bytes: int = 0
    history_count: int = 0
