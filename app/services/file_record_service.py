"""分类记录的持久化服务：把扫描结果写进 SQLite，并按语言查询。

与 `FileClassifier` 的分工：
  * 分类器只负责"文件系统上的事"（扫描、识别、移动），返回结构化结果；
  * 本服务只负责"数据库里的事"（落库、查询），不碰文件系统。
这样分类器可以脱离数据库单独测试，数据库逻辑也不会被文件搬运的细节污染。
"""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.file_record import ClassifiedFileRecord
from app.services.file_classifier import ClassificationReport

logger = logging.getLogger(__name__)


def _source_absolute(root: str, relative_path: str) -> Path:
    """由「根目录 + 相对路径」拼回扫描时的绝对路径。

    单独抽成函数而不是内联：写库时要在两个地方用到它（查询条件与新增赋值），
    抽出来可以保证两处算法完全一致，不会因为改动漏掉一处而出现重复记录。
    """
    return Path(root) / relative_path


class FileRecordService:
    """分类记录的读写。"""

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    @staticmethod
    async def save_report(
        session: AsyncSession, report: ClassificationReport
    ) -> tuple[int, int]:
        """把一次扫描结果写进数据库，返回 (新增条数, 更新条数)。

        参数:
            session: 数据库会话。
            report:  `FileClassifier.classify()` 的返回值。

        返回:
            `(inserted, updated)` 二元组，接口会把它回给前端，方便学生确认"确实记下了"。

        关键逻辑:
            * **按 `source_path`（扫描时的绝对路径）去重**：同一个物理文件反复扫描
              只会更新，不会一条条堆积成历史垃圾。用 `source_path` 而不是归档后的
              路径，是因为文件移动后路径会变 —— 拿新路径去匹配就永远匹配不上。
            * 预演（dry_run）不落库：预演的全部意义就是"不留痕迹地看一眼"。
            * `skipped`（软链接/超大/读不到）的文件不入库：它们没有可用的元信息，
              记进去只会让列表出现一堆无用条目。
            * 单条写失败不影响其它文件，只记日志 —— 扫描本身已经成功，
              不该因为一条脏数据把整次结果丢掉。
        """
        if report.dry_run:
            logger.info("预演模式不写数据库：根目录=%s", report.root)
            return 0, 0

        inserted = 0
        updated = 0
        for item in report.files:
            if item.action == "skipped":
                continue
            try:
                # 扫描时的绝对路径 = 分类根目录 + 相对路径（相对路径就是相对根目录算的）
                source_absolute = str(_source_absolute(report.root, item.source_path))
                existing = (
                    await session.execute(
                        select(ClassifiedFileRecord).where(
                            ClassifiedFileRecord.source_path == source_absolute
                        )
                    )
                ).scalar_one_or_none()

                if existing is None:
                    session.add(
                        ClassifiedFileRecord(
                            filename=item.filename,
                            language=item.language,
                            path=item.target_path,
                            absolute_path=item.absolute_path,
                            source_path=source_absolute,
                            root=report.root,
                            size_bytes=item.size_bytes,
                            file_modified_at=item.modified_at,
                            archived=item.action == "moved",
                            note=item.note or None,
                        )
                    )
                    inserted += 1
                else:
                    existing.filename = item.filename
                    existing.language = item.language
                    existing.path = item.target_path
                    existing.absolute_path = item.absolute_path
                    existing.root = report.root
                    existing.size_bytes = item.size_bytes
                    existing.file_modified_at = item.modified_at
                    existing.archived = existing.archived or (item.action == "moved")
                    existing.note = item.note or None
                    updated += 1
            except Exception:  # noqa: BLE001 - 一条失败不该毁掉整次扫描
                logger.warning("写入分类记录失败: %s", item.filename, exc_info=True)

        try:
            await session.commit()
        except Exception:  # noqa: BLE001
            logger.warning("提交分类记录失败", exc_info=True)
            await session.rollback()
            return 0, 0

        logger.info("分类记录已落库: 新增=%s 更新=%s", inserted, updated)
        return inserted, updated

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    @staticmethod
    async def list_files(
        session: AsyncSession,
        *,
        language: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[ClassifiedFileRecord], int]:
        """按语言分页查询已分类的文件，按入库时间倒序。

        参数:
            session:  数据库会话。
            language: 语言标识（c / java / python / unknown）；None 表示全部。
            limit:    本页最多返回多少条。
            offset:   跳过多少条（配合 limit 做翻页）。

        返回:
            `(记录列表, 符合条件的总数)`。总数用于前端显示"共 N 个文件"。
        """
        count_stmt = select(func.count(ClassifiedFileRecord.id))
        list_stmt = select(ClassifiedFileRecord).order_by(
            ClassifiedFileRecord.created_at.desc(), ClassifiedFileRecord.id.desc()
        )
        if language:
            count_stmt = count_stmt.where(ClassifiedFileRecord.language == language)
            list_stmt = list_stmt.where(ClassifiedFileRecord.language == language)

        total = (await session.execute(count_stmt)).scalar_one()
        rows = (
            await session.execute(list_stmt.limit(limit).offset(offset))
        ).scalars().all()
        return list(rows), total

    @staticmethod
    async def count_by_language(session: AsyncSession) -> dict[str, int]:
        """统计每种语言各有多少个文件，返回 {语言: 数量}。

        用于列表接口顶部的汇总，学生一眼能看出自己哪门课的作业最多。
        """
        rows = await session.execute(
            select(ClassifiedFileRecord.language, func.count(ClassifiedFileRecord.id))
            .group_by(ClassifiedFileRecord.language)
        )
        return {language: count for language, count in rows.all()}


__all__ = ["FileRecordService"]
