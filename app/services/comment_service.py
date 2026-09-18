"""注释生成记录的持久化服务。

分工与其它模块一致：`CommentGenerator` 只负责"生成与复检"，本服务只负责"存"与"读"。
"""

from __future__ import annotations

import logging

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.comment import CommentRecord
from app.services.comment_generator import CommentOutcome

logger = logging.getLogger(__name__)


class CommentService:
    """注释生成记录的读写。"""

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    @staticmethod
    async def save(
        session: AsyncSession, outcome: CommentOutcome, *, trace_id: str | None = None
    ) -> CommentRecord:
        """把一次注释生成的结果写入数据库。

        参数:
            session:  数据库会话。
            outcome:  `CommentGenerator.generate()` 的返回值。
            trace_id: 本次请求的 Trace ID。

        返回:
            落库后的 `CommentRecord`（带自增 id）。

        关键逻辑:
            刻意不让异常冒泡：生成结果已经算出来了，存库失败不该让学生
            连注释都拿不到。失败时只告警并返回未落库对象（id 为 None）。
        """
        check = outcome.verification
        item = CommentRecord(
            filename=outcome.filename,
            language=outcome.language,
            original_code=outcome.original_code,
            commented_code=outcome.commented_code,
            original_lines=len(outcome.original_code.splitlines()),
            commented_lines=len(outcome.commented_code.splitlines()),
            syntax_ok=check.syntax_ok,
            code_unchanged=check.code_unchanged,
            verified=check.verified,
            verification_note=check.note or None,
            functions_total=check.functions_total,
            functions_covered=check.functions_covered,
            coverage_ratio=round(check.coverage_ratio, 4),
            file_comment=check.file_comment,
            comment_lines_before=check.comment_lines_before,
            comment_lines_after=check.comment_lines_after,
            added_comment_lines=check.added_comment_lines,
            inline_comment_lines=check.inline_comment_lines,
            summary=outcome.summary or None,
            ai_available=outcome.ai_available,
            model=outcome.model or None,
            duration_ms=outcome.duration_ms,
            ai_duration_ms=outcome.ai_duration_ms,
            trace_id=trace_id,
        )

        try:
            session.add(item)
            await session.commit()
            await session.refresh(item)
            logger.info(
                "注释记录已落库: id=%s 文件=%s 新增注释=%s 行",
                item.id, item.filename, item.added_comment_lines,
            )
        except Exception:  # noqa: BLE001 - 存不上不该影响返回结果
            logger.warning("写入注释记录失败", exc_info=True)
            await session.rollback()
        return item

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    @staticmethod
    async def list_records(
        session: AsyncSession,
        *,
        language: str | None = None,
        filename: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[CommentRecord], int]:
        """分页查询生成历史，按时间倒序。返回 (记录列表, 总数)。"""
        conditions = []
        if language:
            conditions.append(CommentRecord.language == language)
        if filename:
            conditions.append(CommentRecord.filename == filename)

        count_stmt = select(func.count(CommentRecord.id))
        list_stmt = select(CommentRecord).order_by(
            CommentRecord.created_at.desc(), CommentRecord.id.desc()
        )
        for condition in conditions:
            count_stmt = count_stmt.where(condition)
            list_stmt = list_stmt.where(condition)

        total = (await session.execute(count_stmt)).scalar_one()
        rows = (
            await session.execute(list_stmt.limit(limit).offset(offset))
        ).scalars().all()
        return list(rows), total

    @staticmethod
    async def get_record(session: AsyncSession, record_id: int) -> CommentRecord | None:
        """按主键取一条记录（详情页要拿它做左右对比）。"""
        return await session.get(CommentRecord, record_id)


__all__ = ["CommentService"]
