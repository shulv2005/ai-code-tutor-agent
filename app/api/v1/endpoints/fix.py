"""代码改错接口：/api/v1/fix

  POST /code              传入代码 + 语言，返回修正后的完整代码 + 逐条修改说明，并落库
  GET  /history           改错历史列表（需求 5「方便学生对比学习」）
  GET  /history/{id}      历史详情：原代码 / 新代码 / diff / 四问说明，用于左右对比

流程（细节见 `app/services/code_fixer.py` 的模块注释）：
  1. 本地分析：ast / tree-sitter 定位语法错误（精确到行列）+ 风格与常见坑
  2. AI 修正：把本地结论一起喂给模型，输出修正后的完整代码与四问式讲解
  3. 本地复检：把 AI 给的代码**再跑一遍语法检查**，确认语法错误真的消除了

为什么要把「复检」单独做成一个接口字段：
  "AI 说改好了"和"真的能跑"是两件事。`verification.verified` 由本地解析器给出，
  并且 `note` 里明确写了验证的边界（语法能验证、逻辑不能），前端把它一起展示，
  学生就不会误以为"verified=true 就等于逻辑也对、可以直接交作业"。
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Header, HTTPException, Query, status

from app.api.deps import CodeFixerDep, DbSession
from app.core.api_key_manager import SESSION_HEADER
from app.core.trace import current_trace_id
from app.schemas.fix import (
    CodeChangeRead,
    CodeFixHistoryResponse,
    CodeFixRecordDetail,
    CodeFixRecordRead,
    CodeFixRequest,
    CodeFixResponse,
    DiffStatsRead,
    FixSyntaxErrorRead,
    LocalIssueRead,
    VerificationRead,
)
from app.services.code_checker import UnsupportedLanguageError
from app.services.code_fix_service import CodeFixService
from app.services.code_fixer import (
    CodeChange,
    CodeFixerError,
    FixOutcome,
    SyntaxErrorInfo,
    language_label,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# 改错
# ---------------------------------------------------------------------------
@router.post(
    "/code",
    response_model=CodeFixResponse,
    summary="AI 代码改错",
    description=(
        "先本地分析（Python 用 AST、C/Java 用 Tree-sitter 定位语法错误），"
        "再让大模型分析错误并给出**修正后的完整代码**，同时逐条说明：\n"
        "- `what` 原来错在哪里\n- `why` 为什么错\n"
        "- `how` 怎么改\n- `avoid` 以后如何避免\n\n"
        "返回还包含：`diff`（逐行对比）、`verification`（本地复检结论）、"
        "`local_issues`（本地静态检查结果）。结果写入 SQLite，可随时回看对比。"
    ),
)
async def fix_code(
    payload: CodeFixRequest,
    session: DbSession,
    fixer: CodeFixerDep,
    x_session_id: str | None = Header(
        default=None,
        alias=SESSION_HEADER,
        description="网页上填过 API Key 时会带这个会话号，用来取用户自己的 Key",
    ),
) -> CodeFixResponse:
    """修正代码并给出逐条讲解，最后落库。

    模型的选取：请求体里带了 `model_id` / `api_key`，或带了 `X-Session-Id`，
    就用**用户自己选的那个模型 + 他那把 Key**；一个都没带才回落到 `.env` 配置。
    """
    try:
        outcome = await fixer.fix_text(
            payload.code,
            language=payload.language,
            filename=payload.filename,
            session_id=x_session_id,
            model_id=payload.model_id,
            api_key=payload.api_key,
        )
    except UnsupportedLanguageError as exc:
        # 语言不支持属于"请求不合法"，用 400 让前端能区分于服务端故障
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except CodeFixerError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    record = await CodeFixService.save(session, outcome, trace_id=current_trace_id())
    return _to_response(outcome, record_id=record.id)


# ---------------------------------------------------------------------------
# 历史（对比学习）
# ---------------------------------------------------------------------------
@router.get(
    "/history",
    response_model=CodeFixHistoryResponse,
    summary="改错历史列表",
    description="按时间倒序列出历次改错记录；详情接口里能拿到原代码与新代码做左右对比。",
)
async def list_history(
    session: DbSession,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    language: str | None = Query(default=None, description="按语言筛选：c / java / python"),
    filename: str | None = Query(default=None, description="按文件名精确匹配"),
) -> CodeFixHistoryResponse:
    """分页列出改错历史。"""
    items, total = await CodeFixService.list_records(
        session,
        language=(language or "").strip().lower() or None,
        filename=filename,
        limit=limit,
        offset=offset,
    )
    return CodeFixHistoryResponse(
        total=total, items=[CodeFixRecordRead.model_validate(item) for item in items]
    )


@router.get(
    "/history/{record_id}",
    response_model=CodeFixRecordDetail,
    summary="改错历史详情（用于对比学习）",
    description="返回当时的原代码、修正后的代码、diff 与逐条讲解，供学生左右对比复习。",
)
async def get_history_detail(record_id: int, session: DbSession) -> CodeFixRecordDetail:
    """取一条改错记录的完整内容。"""
    record = await CodeFixService.get_record(session, record_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="记录不存在")

    return CodeFixRecordDetail(
        id=record.id,
        filename=record.filename,
        language=record.language,
        line_count=record.line_count,
        had_error=record.had_error,
        change_count=record.change_count,
        summary=record.summary,
        verified=record.verified,
        ai_available=record.ai_available,
        model=record.model,
        added_lines=record.added_lines,
        removed_lines=record.removed_lines,
        changed_lines=record.changed_lines,
        created_at=record.created_at,
        original_code=record.original_code,
        fixed_code=record.fixed_code,
        changes=_load_changes(record.changes_json),
        categories=_load_json_dict(record.categories_json),
        diff=record.diff_text or "",
        verification_note=record.verification_note,
        syntax_error=record.syntax_error,
    )


# ---------------------------------------------------------------------------
# 结果转换
# ---------------------------------------------------------------------------
def _to_response(outcome: FixOutcome, *, record_id: int | None) -> CodeFixResponse:
    """把服务层的 FixOutcome 转成响应模型。"""
    verification = outcome.verification
    return CodeFixResponse(
        filename=outcome.filename,
        language=outcome.language,
        language_label=language_label(outcome.language),
        had_error=outcome.had_error,
        fixed_code=outcome.fixed_code,
        changes=[_change_to_read(change) for change in outcome.changes],
        summary=outcome.summary,
        categories=outcome.categories,
        diff=outcome.diff,
        diff_stats=DiffStatsRead(**outcome.diff_stats.to_dict()),
        verification=VerificationRead(
            verified=verification.verified,
            note=verification.note,
            syntax_before=_syntax_to_read(verification.syntax_before),
            syntax_after=_syntax_to_read(verification.syntax_after),
        ),
        local_issues=[
            _issue_to_read(item) for item in (outcome.local.issues if outcome.local else [])
        ],
        local_duration_ms=round(outcome.local.duration_ms, 3) if outcome.local else 0.0,
        ai_available=outcome.ai_available,
        model=outcome.model,
        note=outcome.note,
        warnings=outcome.warnings,
        record_id=record_id,
        duration_ms=round(outcome.duration_ms, 3),
        ai_duration_ms=outcome.ai_duration_ms,
        fixed_at=outcome.fixed_at,
        trace_id=current_trace_id(),
    )


def _change_to_read(change: CodeChange) -> CodeChangeRead:
    """修改说明 -> 响应模型。"""
    return CodeChangeRead(
        line=change.line,
        category=change.category,
        what=change.what,
        why=change.why,
        how=change.how,
        avoid=change.avoid,
        original=change.original,
        fixed=change.fixed,
    )


def _syntax_to_read(info: SyntaxErrorInfo | None) -> FixSyntaxErrorRead | None:
    """语法错误信息 -> 响应模型。"""
    if info is None:
        return None
    return FixSyntaxErrorRead(
        line=info.line,
        column=info.column,
        message=info.message,
        tool=info.tool,
        raw=info.raw,
    )


def _issue_to_read(item) -> LocalIssueRead:  # noqa: ANN001 - Issue 类型在服务层定义
    """本地问题 -> 响应模型。"""
    return LocalIssueRead(
        line=item.line,
        severity=item.severity,
        category=item.category,
        title=item.title,
        detail=item.detail,
        suggestion=item.suggestion,
        source=item.source,
    )


def _load_changes(raw: str | None) -> list[CodeChangeRead]:
    """把存储的 JSON 还原成修改说明列表；数据损坏时返回空列表而不是报错。"""
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("改错记录的 changes_json 解析失败，已按空列表处理")
        return []
    if not isinstance(data, list):
        return []

    changes: list[CodeChangeRead] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        category = str(item.get("category") or "logic")
        if category not in ("syntax", "logic", "risk", "style"):
            category = "logic"
        changes.append(
            CodeChangeRead(
                line=item.get("line") if isinstance(item.get("line"), int) else None,
                category=category,  # type: ignore[arg-type]
                what=str(item.get("what") or ""),
                why=str(item.get("why") or ""),
                how=str(item.get("how") or ""),
                avoid=str(item.get("avoid") or ""),
                original=str(item.get("original") or ""),
                fixed=str(item.get("fixed") or ""),
            )
        )
    return changes


def _load_json_dict(raw: str | None) -> dict[str, int]:
    """把存储的 JSON 对象还原成 {类型: 条数}；损坏时返回空字典。"""
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): int(value) for key, value in data.items() if isinstance(value, int)}


__all__ = ["router"]
