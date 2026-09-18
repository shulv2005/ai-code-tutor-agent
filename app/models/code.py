"""代码结构 ORM 模型：CodeFile（文件）与 CodeSymbol（符号）。

这是 Step 3 混合检索的语料来源：BM25 吃 CodeSymbol.signature/docstring 的文本，
向量库吃代码分块（Step 3 实现），两者都锚定到 CodeSymbol 与行号区间。
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.repository import utc_now

if TYPE_CHECKING:
    from app.models.repository import Repository


class CodeFile(Base):
    """仓库中的一个源文件及其解析结果概要。"""

    __tablename__ = "code_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repository_id: Mapped[int] = mapped_column(
        ForeignKey("repositories.id", ondelete="CASCADE"), index=True, nullable=False
    )
    # 仓库内相对路径，统一用 posix 风格（'/' 分隔），保证跨平台一致
    path: Mapped[str] = mapped_column(String(1024), nullable=False)
    language: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_lines: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    symbol_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # 解析失败（语法错误、编码错误、超限）时记录原因，文件本身仍入库
    parse_error: Mapped[str | None] = mapped_column(Text, default=None)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    repository: Mapped[Repository] = relationship(back_populates="files")
    symbols: Mapped[list[CodeSymbol]] = relationship(
        back_populates="file",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="selectin",
    )

    __table_args__ = (UniqueConstraint("repository_id", "path", name="uq_code_files_repo_path"),)

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return f"<CodeFile id={self.id} {self.path} lang={self.language}>"


class CodeSymbol(Base):
    """文件内的一个可定位符号（函数 / 方法 / 类 / 结构体 / 接口）。"""

    __tablename__ = "code_symbols"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    file_id: Mapped[int] = mapped_column(
        ForeignKey("code_files.id", ondelete="CASCADE"), index=True, nullable=False
    )
    # 冗余 repository_id：Step 3 检索时按仓库过滤，避免每次 join code_files
    repository_id: Mapped[int] = mapped_column(
        ForeignKey("repositories.id", ondelete="CASCADE"), index=True, nullable=False
    )

    name: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    qualified_name: Mapped[str] = mapped_column(String(512), index=True, nullable=False)
    kind: Mapped[str] = mapped_column(String(32), index=True, nullable=False)

    # 1-based 闭区间行号，用于 Step 4/6 精确切片给 LLM
    start_line: Mapped[int] = mapped_column(Integer, nullable=False)
    end_line: Mapped[int] = mapped_column(Integer, nullable=False)

    signature: Mapped[str] = mapped_column(Text, default="", nullable=False)
    docstring: Mapped[str | None] = mapped_column(Text, default=None)
    is_async: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    # 圈复杂度近似值，Step 4 用它优先给复杂函数生成测试
    complexity: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    file: Mapped[CodeFile] = relationship(back_populates="symbols")

    __table_args__ = (
        Index("ix_code_symbols_repo_kind", "repository_id", "kind"),
        Index("ix_code_symbols_file_start", "file_id", "start_line"),
    )

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return f"<CodeSymbol {self.kind} {self.qualified_name} L{self.start_line}-{self.end_line}>"
