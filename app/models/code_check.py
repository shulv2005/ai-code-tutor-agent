"""AI 自动检测记录的 ORM 模型。

每次调用 `POST /api/v1/check/code` 都会写一条记录：谁（文件名/语言）、
检查了哪段代码、本地检查的结论、AI 给出的问题与学习建议、最终评分。

为什么把代码全文也存下来：
学生回头看到"上次这份作业 62 分"时，最想知道的是"当时错在哪"，
只存分数和问题列表就答不上来；代码片段通常只有几十行，存储代价可接受。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.models.repository import utc_now


class CodeCheckRecord(Base):
    """一次 AI 自动检测的结果。"""

    __tablename__ = "code_check_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # ---------- 被检查的代码 ----------
    filename: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    language: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    code: Mapped[str] = mapped_column(Text, nullable=False)
    line_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # ---------- 本地静态检查（不依赖大模型，永远有值）----------
    # 语法是否通过（Python 走 ast，C/Java 走 tree-sitter 的错误节点检测）
    syntax_ok: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    # 语法错误的描述，例如 "SyntaxError: invalid syntax (line 3)"
    syntax_error: Mapped[str | None] = mapped_column(String(512), default=None)

    # ---------- AI 检测结论 ----------
    score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    level: Mapped[str] = mapped_column(String(16), default="", nullable=False)
    summary: Mapped[str | None] = mapped_column(Text, default=None)

    # 三类问题的条数：语法/逻辑错误、风格建议、潜在风险
    error_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    style_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    risk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # 完整问题列表与学习建议，存 JSON 字符串：
    # 这是展示型数据、结构随 Prompt 演进会变，拆成多张表维护成本远大于收益。
    issues_json: Mapped[str | None] = mapped_column(Text, default=None)
    advice_json: Mapped[str | None] = mapped_column(Text, default=None)
    highlights_json: Mapped[str | None] = mapped_column(Text, default=None)

    # ---------- 运行信息 ----------
    # AI 是否真的参与了本次检测。没配模型时接口仍然能用（只出本地结论），
    # 这个字段让"哪些记录是纯本地结果"一目了然，避免统计时把两种情况混在一起。
    ai_available: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    model: Mapped[str | None] = mapped_column(String(128), default=None)
    duration_ms: Mapped[float | None] = mapped_column(Float, default=None)
    local_duration_ms: Mapped[float | None] = mapped_column(Float, default=None)
    ai_duration_ms: Mapped[float | None] = mapped_column(Float, default=None)
    trace_id: Mapped[str | None] = mapped_column(String(64), default=None)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, index=True, nullable=False
    )

    __table_args__ = (
        # 典型查询：某语言的检测历史 / 某个文件被检查过几次，均按时间倒序
        Index("ix_code_check_lang_created", "language", "created_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return f"<CodeCheckRecord id={self.id} {self.filename} {self.score}>"
