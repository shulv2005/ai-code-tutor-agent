"""代码改错记录的 ORM 模型。

需求里的第 5 条是"修改历史保存到 SQLite，**方便学生对比学习**"。
"对比学习"这个目的决定了表结构必须同时存下三样东西：
  1. 修改前的代码（original_code）——学生要看到自己原来是怎么写的；
  2. 修改后的代码（fixed_code）——要能直接复制去跑；
  3. 逐条讲解（changes_json）与差异（diff_text）——知道"错在哪、为什么、怎么避免"。

只存"改后的代码"是没用的：学生过两天回来看，根本想不起自己原来错在哪。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.models.repository import utc_now


class CodeFixRecord(Base):
    """一次代码改错的记录。"""

    __tablename__ = "code_fix_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # ---------- 被改的代码 ----------
    filename: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    language: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    # 修改前后的完整代码：对比学习的关键，两个都要存
    original_code: Mapped[str] = mapped_column(Text, nullable=False)
    fixed_code: Mapped[str] = mapped_column(Text, nullable=False)
    line_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # ---------- 修改情况 ----------
    had_error: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    change_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text, default=None)
    # 逐条讲解（四问结构：错在哪/为什么错/怎么改/以后如何避免），JSON 字符串
    changes_json: Mapped[str | None] = mapped_column(Text, default=None)
    # 各类错误的条数，例如 {"logic": 2, "syntax": 1}，用于统计"学生在哪类问题上栽跟头"
    categories_json: Mapped[str | None] = mapped_column(Text, default=None)
    # 统一 diff 文本，前端可以直接渲染成"逐行对比"
    diff_text: Mapped[str | None] = mapped_column(Text, default=None)
    added_lines: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    removed_lines: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    changed_lines: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # ---------- 本地复检 ----------
    # 修正后的代码是否通过了本地语法复检。注意它只代表"语法层面"，
    # 逻辑正确性本地验证不了（见 code_fixer.FixVerification 的说明）。
    verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    syntax_error: Mapped[str | None] = mapped_column(String(512), default=None)
    verification_note: Mapped[str | None] = mapped_column(Text, default=None)

    # ---------- 运行信息 ----------
    ai_available: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    model: Mapped[str | None] = mapped_column(String(128), default=None)
    duration_ms: Mapped[float | None] = mapped_column(Float, default=None)
    ai_duration_ms: Mapped[float | None] = mapped_column(Float, default=None)
    trace_id: Mapped[str | None] = mapped_column(String(64), default=None)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, index=True, nullable=False
    )

    __table_args__ = (
        # 典型查询：某个文件改过几次 / 某语言的改错历史，均按时间倒序
        Index("ix_code_fix_file_created", "filename", "created_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return f"<CodeFixRecord id={self.id} {self.filename} changes={self.change_count}>"
