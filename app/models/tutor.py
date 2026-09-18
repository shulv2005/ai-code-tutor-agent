"""学生代码学习记录的 ORM 模型。

面向「AI 代码导师」场景：保存学生提交过的代码、AI 检测/注释/改错的结果，
用于前端展示历史记录，也方便课堂回顾与复习。
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Literal

from sqlalchemy import DateTime, Float, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.models.repository import utc_now

if TYPE_CHECKING:
    pass

# 学生可以对同一份代码做三种操作
TutorAction = Literal["analyze", "check", "comment", "fix"]


class TutorRecord(Base):
    """一次 AI 辅导的记录（检测 / 注释 / 改错）。"""

    __tablename__ = "tutor_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # ---------- 学生代码的基本信息 ----------
    filename: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    language: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    code: Mapped[str] = mapped_column(Text, nullable=False)
    line_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # ---------- 本次操作 ----------
    # analyze=仅解析分类；check=AI 检测；comment=生成注释；fix=自动改错
    action: Mapped[str] = mapped_column(String(16), index=True, nullable=False)

    # 检测得分（0-100），仅 check 操作有值
    score: Mapped[float | None] = mapped_column(Float, default=None)
    # 一句话结论，用于历史列表展示
    summary: Mapped[str | None] = mapped_column(Text, default=None)

    # 完整结果的 JSON 字符串：检测问题列表 / 注释后的代码 / 修改说明等
    # 用 Text 存 JSON 而不是拆多张表：结果是易变的展示型数据，
    # 前端按自己的需要解析即可，避免为每种操作的字段差异建表。
    result_json: Mapped[str | None] = mapped_column(Text, default=None)

    # 使用的模型名与耗时，便于排查"为什么这次结果不一样"
    model: Mapped[str | None] = mapped_column(String(128), default=None)
    duration_ms: Mapped[float | None] = mapped_column(Float, default=None)
    trace_id: Mapped[str | None] = mapped_column(String(64), default=None)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, index=True, nullable=False
    )

    __table_args__ = (
        # 历史列表按「文件名 + 操作类型」筛选、按时间倒序，因此建组合索引
        Index("ix_tutor_records_file_action", "filename", "action"),
    )

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return f"<TutorRecord id={self.id} {self.filename} {self.action}>"
