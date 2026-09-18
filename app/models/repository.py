"""Repository ORM 模型：一个被解析的开源仓库。"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from sqlalchemy import DateTime, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.code import CodeFile

# 仓库在索引流水线中的状态机：
# pending -> cloning -> parsing -> ready
#                    \-> failed（任一阶段异常）
RepoStatus = Literal["pending", "cloning", "parsing", "ready", "failed"]


def utc_now() -> datetime:
    """统一的 UTC 时间戳工厂。"""
    return datetime.now(UTC)


class Repository(Base):
    """被克隆并索引的远程 Git 仓库。"""

    __tablename__ = "repositories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # 归一化后的克隆地址，唯一约束保证同一仓库不会重复克隆
    url: Mapped[str] = mapped_column(String(512), unique=True, index=True, nullable=False)
    host: Mapped[str] = mapped_column(String(128), nullable=False)
    owner: Mapped[str] = mapped_column(String(128), nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)

    local_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    default_branch: Mapped[str | None] = mapped_column(String(255), default=None)
    head_commit: Mapped[str | None] = mapped_column(String(64), default=None)

    status: Mapped[str] = mapped_column(String(32), default="pending", index=True, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, default=None)

    # 索引统计（冗余存储，避免每次列表查询都 count 子表）
    file_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    symbol_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    cloned_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    # async SQLAlchemy 下惰性加载会抛 MissingGreenlet；但用 selectin 又会让
    # 「列出仓库」这种查询把全部子文件一起拉出来。故用 raise_on_sql：默认禁止
    # 隐式加载，需要时显式 selectinload(Repository.files)。
    files: Mapped[list[CodeFile]] = relationship(
        back_populates="repository",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="raise_on_sql",
    )

    __table_args__ = (Index("ix_repositories_owner_name", "owner", "name"),)

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return f"<Repository id={self.id} {self.owner}/{self.name} status={self.status}>"
