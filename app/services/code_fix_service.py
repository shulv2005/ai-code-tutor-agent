"""代码改错记录的持久化服务。

分工与检测模块一致：`CodeFixer` 只负责"改"，本服务只负责"存"与"读"。
"""

from __future__ import annotations

import json
import logging

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.code_fix import CodeFixRecord
from app.services.code_fixer import FixOutcome

logger = logging.getLogger(__name__)


class CodeFixService:
    """改错记录的读写。"""

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    @staticmethod
    async def save(
        session: AsyncSession, outcome: FixOutcome, *, trace_id: str | None = None
    ) -> CodeFixRecord:
        """把一次改错结果写入数据库。

        参数:
            session:  数据库会话。
            outcome:  `CodeFixer.fix()` 的返回值。
            trace_id: 本次请求的 Trace ID，便于把记录与日志对上。

        返回:
            落库后的 `CodeFixRecord`（带自增 id）。

        关键逻辑:
            与其它模块同样的原则：**不让存库失败影响学生的结果**。
            异常只告警不抛出，返回未落库的对象（id 为 None，接口据此返回 null）。
        """
        verification = outcome.verification
        stats = outcome.diff_stats
        item = CodeFixRecord(
            filename=outcome.filename,
            language=outcome.language,
            original_code=outcome.original_code,
            fixed_code=outcome.fixed_code,
            line_count=len(outcome.original_code.splitlines()),
            had_error=outcome.had_error,
            change_count=outcome.change_count,
            summary=outcome.summary or None,
            changes_json=_dump([change.to_dict() for change in outcome.changes]),
            categories_json=_dump(outcome.categories),
            diff_text=outcome.diff or None,
            added_lines=stats.added,
            removed_lines=stats.removed,
            changed_lines=stats.changed,
            verified=verification.verified,
            syntax_error=verification.syntax_before.raw if verification.syntax_before else None,
            verification_note=verification.note or None,
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
                "改错记录已落库: id=%s 文件=%s 修改=%s 处",
                item.id, item.filename, item.change_count,
            )
        except Exception:  # noqa: BLE001 - 存不上不该影响返回结果
            logger.warning("写入改错记录失败", exc_info=True)
            await session.rollback()
        return item

    # ------------------------------------------------------------------
    # 查询（需求 5 的"方便对比学习"就靠这两个方法）
    # ------------------------------------------------------------------
    @staticmethod
    async def list_records(
        session: AsyncSession,
        *,
        language: str | None = None,
        filename: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[CodeFixRecord], int]:
        """分页查询改错历史，按时间倒序。返回 (记录列表, 总数)。"""
        conditions = []
        if language:
            conditions.append(CodeFixRecord.language == language)
        if filename:
            conditions.append(CodeFixRecord.filename == filename)

        count_stmt = select(func.count(CodeFixRecord.id))
        list_stmt = select(CodeFixRecord).order_by(
            CodeFixRecord.created_at.desc(), CodeFixRecord.id.desc()
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
    async def get_record(session: AsyncSession, record_id: int) -> CodeFixRecord | None:
        """按主键取一条记录（详情页要拿它做左右对比）。"""
        return await session.get(CodeFixRecord, record_id)


def _dump(value: object) -> str:
    """序列化成 JSON 字符串；中文不转义，便于直接在数据库里查看。"""
    return json.dumps(value, ensure_ascii=False)


__all__ = ["CodeFixService"]
