"""本地代码文件分类记录的 ORM 模型。

对应「本地代码文件自动分类」功能：每次扫描后，把每个被识别到的文件的元信息
写进 SQLite，供 `GET /api/v1/files/list` 按语言查询。

为什么单独建一张表而不是复用 tutor_records：
`tutor_records` 存的是"学生对某段代码做过什么 AI 操作"，代码内容全量入库；
这里存的是"这个文件在磁盘上的什么位置、多大、什么时候进来的"，是文件台账，
两者生命周期完全不同（文件删了 AI 记录仍然有意义，反之亦然）。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, Boolean, DateTime, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.models.repository import utc_now


class ClassifiedFileRecord(Base):
    """一个被分类（并可能已归档）的本地代码文件。"""

    __tablename__ = "classified_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # ---------- 识别结果 ----------
    filename: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    # c / java / python / unknown（只认这三种语言，其余一律 unknown）
    language: Mapped[str] = mapped_column(String(32), index=True, nullable=False)

    # ---------- 文件位置 ----------
    # 归档后的路径（相对分类根目录，正斜杠，如 "c/main.c"）；未知文件未移动时即原位置
    path: Mapped[str] = mapped_column(String(1024), index=True, nullable=False)
    # 归档后的绝对路径：列表接口用它判断文件是否还在（学生在外面删掉的情况）
    absolute_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    # 扫描时（移动前）的绝对路径。**唯一索引建在它上面**：
    # 同一个物理文件无论后来被移到哪个语言目录，都只对应一条记录，重复扫描只更新不新增。
    source_path: Mapped[str] = mapped_column(String(1024), nullable=False, unique=True)
    # 所属分类根目录（支持配置多个根目录时用来区分）
    root: Mapped[str] = mapped_column(String(1024), index=True, nullable=False)

    # ---------- 元信息 ----------
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    # 文件自身的修改时间（磁盘上的 mtime）
    file_modified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )
    # 是否已经归档到语言子目录里
    archived: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    # 未能处理时的原因（软链接、超大、读不到等）
    note: Mapped[str | None] = mapped_column(String(512), default=None)

    # ---------- 时间 ----------
    # 首次入库时间，也就是接口里展示的「上传时间」
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, index=True, nullable=False
    )
    # 最近一次扫描到它的时间
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    __table_args__ = (
        # 列表接口的典型查询：按语言筛选 + 按时间倒序
        Index("ix_classified_files_lang_created", "language", "created_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return f"<ClassifiedFileRecord id={self.id} {self.path} ({self.language})>"
