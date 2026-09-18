"""代码检测记录的持久化服务。

与 `CodeChecker` 的分工（和导师模块同一套分层）：
  * `CodeChecker` 只负责"检查"，产出 `CheckOutcome`，不碰数据库；
  * 本服务只负责"存"，把结论落进 SQLite 供复习与统计。
这样检测逻辑可以脱离数据库测试，数据库改动也不会碰到检测逻辑。
"""

from __future__ import annotations

import json
import logging

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.code_check import CodeCheckRecord
from app.services.code_checker import CheckOutcome

logger = logging.getLogger(__name__)


class CodeCheckService:
    """检测记录的读写。"""

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    @staticmethod
    async def save(
        session: AsyncSession, outcome: CheckOutcome, *, code: str, trace_id: str | None = None
    ) -> CodeCheckRecord:
        """把一次检测结果写入数据库。

        参数:
            session:  数据库会话。
            outcome:  `CodeChecker.check()` 的返回值。
            code:     被检测的原始代码（存全文，方便学生回看"当时错在哪"）。
            trace_id: 本次请求的 Trace ID，便于把记录和日志对上。

        返回:
            落库后的 `CodeCheckRecord`（带自增 id）。

        关键逻辑:
            刻意不让异常冒泡：**学生的检测结果已经算出来了，不能因为存库失败
            就让整个请求报错**。失败时只告警，并返回一个未落库的对象
            （其 id 为 None，接口据此把 record_id 返回为 null，前端不会误以为存上了）。
        """
        syntax_error = outcome.local.syntax_error
        item = CodeCheckRecord(
            filename=outcome.filename,
            language=outcome.language,
            code=code,
            line_count=outcome.local.metrics.total_lines,
            syntax_ok=outcome.local.syntax_ok,
            syntax_error=syntax_error.raw if syntax_error else None,
            score=outcome.score,
            level=outcome.level,
            summary=outcome.summary or outcome.score_reason,
            error_count=len(outcome.errors),
            style_count=len(outcome.style),
            risk_count=len(outcome.risks),
            issues_json=_dump([item.to_dict() for item in outcome.all_issues]),
            advice_json=_dump(outcome.advice),
            highlights_json=_dump(outcome.highlights),
            ai_available=outcome.ai_available,
            model=outcome.model or None,
            duration_ms=outcome.duration_ms,
            local_duration_ms=outcome.local.duration_ms,
            ai_duration_ms=outcome.ai_duration_ms,
            trace_id=trace_id,
        )

        try:
            session.add(item)
            await session.commit()
            await session.refresh(item)
            logger.info(
                "检测记录已落库: id=%s 文件=%s 评分=%s", item.id, item.filename, item.score
            )
        except Exception:  # noqa: BLE001 - 存不上不该影响检测结果
            logger.warning("写入检测记录失败", exc_info=True)
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
    ) -> tuple[list[CodeCheckRecord], int]:
        """分页查询检测历史，按时间倒序。返回 (记录列表, 总数)。

        参数:
            session:  数据库会话。
            language: 按语言筛选（c / java / python），None 表示全部。
            filename: 按文件名精确匹配，用于看"同一份作业改了几版"。
            limit:    本页最多返回多少条。
            offset:   跳过多少条。
        """
        conditions = []
        if language:
            conditions.append(CodeCheckRecord.language == language)
        if filename:
            conditions.append(CodeCheckRecord.filename == filename)

        count_stmt = select(func.count(CodeCheckRecord.id))
        list_stmt = select(CodeCheckRecord).order_by(
            CodeCheckRecord.created_at.desc(), CodeCheckRecord.id.desc()
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
    async def get_record(session: AsyncSession, record_id: int) -> CodeCheckRecord | None:
        """按主键取一条记录（详情用）。"""
        return await session.get(CodeCheckRecord, record_id)


def _dump(value: object) -> str:
    """序列化成 JSON 字符串；中文不转义，便于直接在数据库里查看。"""
    return json.dumps(value, ensure_ascii=False)


__all__ = ["CodeCheckService"]
