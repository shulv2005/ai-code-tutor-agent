"""代码注释生成接口：/api/v1/comment

  POST /generate          传入代码 + 语言，返回带三层中文注释的完整代码，并落库
  GET  /history           生成历史列表
  GET  /history/{id}      历史详情：原代码 / 带注释代码 / 复检结论，用于左右对比

流程（细节见 `app/services/comment_generator.py` 的模块注释）：
  1. 本地分析：解析出函数/方法清单与签名（告诉模型该给谁写注释、参数有哪些）
  2. AI 生成：文件级 / 函数级 / 行内三层注释，按语言规范（docstring / Javadoc / C 块注释）
  3. 本地复检：**代码有没有被改动**、语法有没有被注释破坏、函数有没有漏注释

为什么复检要单独做成一组字段：
  注释类任务最危险的不是"注释写得不好"，而是"模型顺手把代码改了"。
  `verification.code_unchanged` 由本地 AST / 去注释文本比对给出，
  一旦为 false 就说明返回的代码不能直接拿去交作业，必须让学生看见。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Header, HTTPException, Query, status

from app.api.deps import CommentGeneratorDep, DbSession
from app.core.api_key_manager import SESSION_HEADER
from app.core.trace import current_trace_id
from app.schemas.comment import (
    CommentGenerateRequest,
    CommentGenerateResponse,
    CommentHistoryResponse,
    CommentRecordDetail,
    CommentRecordRead,
    CommentVerificationRead,
)
from app.services.code_checker import UnsupportedLanguageError, language_label
from app.services.comment_generator import CommentGeneratorError, CommentOutcome
from app.services.comment_service import CommentService

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# 生成
# ---------------------------------------------------------------------------
@router.post(
    "/generate",
    response_model=CommentGenerateResponse,
    summary="AI 生成中文注释",
    description=(
        "为代码生成三层中文注释：\n"
        "- **文件级**：说明这个文件做什么\n"
        "- **函数/方法级**：功能、参数、返回值，按语言规范书写"
        "（Python 用 docstring、Java 用 Javadoc、C 用块注释）\n"
        "- **关键逻辑行内注释**：只解释难懂的代码，不逐行堆砌\n\n"
        "返回带注释的完整代码，并附**本地复检结论**："
        "`code_unchanged` 表示代码逻辑有没有被改动（null=无法判定）、"
        "`coverage_ratio` 是函数注释覆盖率、`verified` 是综合结论。"
        "结果写入 SQLite，可随时回看对比。"
    ),
)
async def generate_comments(
    payload: CommentGenerateRequest,
    session: DbSession,
    generator: CommentGeneratorDep,
    x_session_id: str | None = Header(
        default=None,
        alias=SESSION_HEADER,
        description="网页上填过 API Key 时会带这个会话号，用来取用户自己的 Key",
    ),
) -> CommentGenerateResponse:
    """为代码生成注释并落库。

    模型的选取：请求体里带了 `model_id` / `api_key`，或带了 `X-Session-Id`，
    就用**用户自己选的那个模型 + 他那把 Key**；一个都没带才回落到 `.env` 配置。
    """
    try:
        outcome = await generator.generate_text(
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
    except CommentGeneratorError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    record = await CommentService.save(session, outcome, trace_id=current_trace_id())
    return _to_response(outcome, record_id=record.id)


# ---------------------------------------------------------------------------
# 历史
# ---------------------------------------------------------------------------
@router.get(
    "/history",
    response_model=CommentHistoryResponse,
    summary="注释生成历史列表",
    description="按时间倒序列出历次生成记录；详情接口里能拿到原代码与带注释代码做对比。",
)
async def list_history(
    session: DbSession,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    language: str | None = Query(default=None, description="按语言筛选：c / java / python"),
    filename: str | None = Query(default=None, description="按文件名精确匹配"),
) -> CommentHistoryResponse:
    """分页列出注释生成历史。"""
    items, total = await CommentService.list_records(
        session,
        language=(language or "").strip().lower() or None,
        filename=filename,
        limit=limit,
        offset=offset,
    )
    return CommentHistoryResponse(
        total=total, items=[CommentRecordRead.model_validate(item) for item in items]
    )


@router.get(
    "/history/{record_id}",
    response_model=CommentRecordDetail,
    summary="注释历史详情（用于对比学习）",
    description="返回当时的原代码与带注释的代码，以及复检结论，供学生左右对比复习。",
)
async def get_history_detail(record_id: int, session: DbSession) -> CommentRecordDetail:
    """取一条注释生成记录的完整内容。"""
    record = await CommentService.get_record(session, record_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="记录不存在")

    return CommentRecordDetail(
        id=record.id,
        filename=record.filename,
        language=record.language,
        original_lines=record.original_lines,
        commented_lines=record.commented_lines,
        added_comment_lines=record.added_comment_lines,
        functions_total=record.functions_total,
        functions_covered=record.functions_covered,
        coverage_ratio=record.coverage_ratio,
        file_comment=record.file_comment,
        verified=record.verified,
        code_unchanged=record.code_unchanged,
        ai_available=record.ai_available,
        model=record.model,
        summary=record.summary,
        created_at=record.created_at,
        original_code=record.original_code,
        commented_code=record.commented_code,
        verification_note=record.verification_note,
        syntax_ok=record.syntax_ok,
        inline_comment_lines=record.inline_comment_lines,
        duration_ms=record.duration_ms,
    )


# ---------------------------------------------------------------------------
# 结果转换
# ---------------------------------------------------------------------------
def _to_response(outcome: CommentOutcome, *, record_id: int | None) -> CommentGenerateResponse:
    """把服务层的 CommentOutcome 转成响应模型。"""
    check = outcome.verification
    return CommentGenerateResponse(
        filename=outcome.filename,
        language=outcome.language,
        language_label=language_label(outcome.language),
        original_code=outcome.original_code,
        commented_code=outcome.commented_code,
        summary=outcome.summary,
        verification=CommentVerificationRead(
            syntax_ok=check.syntax_ok,
            code_unchanged=check.code_unchanged,
            unchanged_note=check.unchanged_note,
            functions_total=check.functions_total,
            functions_covered=check.functions_covered,
            coverage_ratio=round(check.coverage_ratio, 4),
            file_comment=check.file_comment,
            comment_lines_before=check.comment_lines_before,
            comment_lines_after=check.comment_lines_after,
            added_comment_lines=check.added_comment_lines,
            inline_comment_lines=check.inline_comment_lines,
            verified=check.verified,
            note=check.note,
        ),
        ai_available=outcome.ai_available,
        model=outcome.model,
        note=outcome.note,
        warnings=outcome.warnings,
        record_id=record_id,
        duration_ms=round(outcome.duration_ms, 3),
        ai_duration_ms=outcome.ai_duration_ms,
        generated_at=outcome.generated_at,
        trace_id=current_trace_id(),
    )


__all__ = ["router"]
