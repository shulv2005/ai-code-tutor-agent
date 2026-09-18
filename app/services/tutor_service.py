"""AI 代码导师的记录服务：历史记录的读写与评分等级换算。

与 Agent 分开的原因：Agent 只负责「问模型、拿结果」，
持久化与查询属于服务层职责，分开后 Agent 可以脱离数据库单独测试。
"""

from __future__ import annotations

import logging

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.tutor import TutorRecord
from app.services.text_utils import count_text_lines

logger = logging.getLogger(__name__)


class TutorService:
    """学生代码学习记录的服务层。"""

    # -- 评分等级 ---------------------------------------------------------
    @staticmethod
    def score_level(score: float) -> str:
        """把 0-100 的分数换算成学生看得懂的等级。"""
        if score >= 90:
            return "优秀"
        if score >= 75:
            return "良好"
        if score >= 60:
            return "及格"
        return "待改进"

    # -- 写入 -------------------------------------------------------------
    @staticmethod
    async def record(
        session: AsyncSession,
        *,
        filename: str,
        language: str,
        code: str,
        action: str,
        summary: str | None = None,
        result_json: str | None = None,
        score: float | None = None,
        model: str | None = None,
        duration_ms: float | None = None,
    ) -> TutorRecord:
        """写入一条学习记录。

        刻意不让异常冒泡：记录失败不应该让学生的检测结果丢失，
        因此写失败时只告警，仍返回一个未落库的对象。
        """
        from app.core.trace import current_trace_id

        item = TutorRecord(
            filename=filename,
            language=language,
            code=code,
            line_count=count_text_lines(code),
            action=action,
            score=score,
            summary=summary,
            result_json=result_json,
            model=model,
            duration_ms=duration_ms,
            trace_id=current_trace_id(),
        )
        try:
            session.add(item)
            await session.commit()
            await session.refresh(item)
        except Exception:  # noqa: BLE001 - 记不上不该影响主流程
            logger.warning("写入导师记录失败", exc_info=True)
            await session.rollback()
        return item

    # -- 查询 -------------------------------------------------------------
    @staticmethod
    async def list_records(
        session: AsyncSession,
        *,
        limit: int = 20,
        offset: int = 0,
        filename: str | None = None,
        action: str | None = None,
    ) -> tuple[list[TutorRecord], int]:
        """分页查询历史记录，按时间倒序。返回 (记录列表, 总数)。"""
        conditions = []
        if filename:
            conditions.append(TutorRecord.filename == filename)
        if action:
            conditions.append(TutorRecord.action == action)

        count_stmt = select(func.count(TutorRecord.id))
        list_stmt = select(TutorRecord).order_by(TutorRecord.created_at.desc())
        for condition in conditions:
            count_stmt = count_stmt.where(condition)
            list_stmt = list_stmt.where(condition)

        total = (await session.execute(count_stmt)).scalar_one()
        rows = (
            await session.execute(list_stmt.limit(limit).offset(offset))
        ).scalars().all()
        return list(rows), total

    @staticmethod
    async def get_record(session: AsyncSession, record_id: int) -> TutorRecord | None:
        """按主键取一条记录。"""
        return await session.get(TutorRecord, record_id)

    @staticmethod
    async def delete_record(session: AsyncSession, record_id: int) -> bool:
        """删除一条记录；不存在时返回 False。"""
        result = await session.execute(
            delete(TutorRecord).where(TutorRecord.id == record_id)
        )
        await session.commit()
        return bool(result.rowcount)


__all__ = ["TutorService"]
