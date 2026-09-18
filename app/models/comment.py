"""代码注释生成记录的 ORM 模型。

需求第 6 条要求"生成历史保存到 SQLite"。这里同时存下三样东西：
  1. 原代码（original_code）——学生对比"我原来写的"和"加了注释的"；
  2. 带注释的代码（commented_code）——可直接复制回去；
  3. 复检结论（code_unchanged / coverage / verified）——说明这次生成值不值得信。

第 3 点尤其重要：如果某次生成"改动了原有代码"，学生回头翻历史时应该能一眼看到，
而不是把一份被动过手脚的代码当成自己的代码继续用。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.models.repository import utc_now


class CommentRecord(Base):
    """一次注释生成的记录。"""

    __tablename__ = "comment_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # ---------- 代码 ----------
    filename: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    language: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    original_code: Mapped[str] = mapped_column(Text, nullable=False)
    commented_code: Mapped[str] = mapped_column(Text, nullable=False)
    original_lines: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    commented_lines: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # ---------- 复检结论 ----------
    # 注释有没有把代码改坏
    syntax_ok: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    # 代码逻辑有没有被改动。null 表示无法判定（例如原代码本身有语法错误），
    # 这与 False（确实改了）是两回事，必须分开存，否则统计时会混淆。
    code_unchanged: Mapped[bool | None] = mapped_column(Boolean, default=None)
    verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    verification_note: Mapped[str | None] = mapped_column(Text, default=None)

    # ---------- 注释覆盖情况 ----------
    functions_total: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    functions_covered: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # 覆盖率不落库、由上面两个字段算得（避免同一事实存两份而对不上），
    # 但为了列表展示方便，这里存一份冗余值；写入时由服务层统一计算。
    coverage_ratio: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    file_comment: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    comment_lines_before: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    comment_lines_after: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    added_comment_lines: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    inline_comment_lines: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # ---------- 运行信息 ----------
    summary: Mapped[str | None] = mapped_column(Text, default=None)
    ai_available: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    model: Mapped[str | None] = mapped_column(String(128), default=None)
    duration_ms: Mapped[float | None] = mapped_column(Float, default=None)
    ai_duration_ms: Mapped[float | None] = mapped_column(Float, default=None)
    trace_id: Mapped[str | None] = mapped_column(String(64), default=None)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, index=True, nullable=False
    )

    __table_args__ = (
        # 典型查询：某个文件生成过几次 / 某语言的历史，均按时间倒序
        Index("ix_comment_file_created", "filename", "created_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return f"<CommentRecord id={self.id} {self.filename} +{self.added_comment_lines}行>"
