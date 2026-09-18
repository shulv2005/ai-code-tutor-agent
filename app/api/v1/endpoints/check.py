"""AI 自动检测接口：/api/v1/check

按需求提供一个接口：
  POST /code   接收代码 + 语言，返回检测结果，并把结果写入 SQLite

处理流程（两阶段的细节见 `app/services/code_checker.py` 的模块注释）：
  1. 本地静态检查：Python 用 ast、C/Java 用 tree-sitter 找语法错误，
     附带缩进/命名/注释比例等风格规则与几个经典风险模式；
  2. AI 深度检测：把本地结论一起喂给模型，让它专注于逻辑、边界与算法，
     输出「错误 / 风格 / 风险 / 学习建议 + 评分」。

两个刻意的设计（都写在这里，方便前端同学理解为什么字段是这么给的）：
  * **没配大模型时返回 200 而不是 503**。阶段 1 的结论（语法错误位置、风格问题）
    本身就是学生最需要的东西，因为缺 Key 就把已经算出来的结果丢掉太浪费；
    这时 `ai_available=false`、`note` 里说明原因，前端据此提示"配好模型后有更深的建议"。
  * **AI 调用失败也返回 200**，同理：降级为"只有本地结论 + 一条 warning"。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Header, HTTPException, status

from app.api.deps import CodeCheckerDep, DbSession
from app.core.api_key_manager import SESSION_HEADER
from app.core.trace import current_trace_id
from app.schemas.check import (
    CodeCheckRequest,
    CodeCheckResponse,
    CodeIssueRead,
    CodeMetricsRead,
    LocalCheckRead,
    SyntaxErrorRead,
)
from app.services.code_check_service import CodeCheckService
from app.services.code_checker import (
    CodeCheckError,
    UnsupportedLanguageError,
    language_label,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/code",
    response_model=CodeCheckResponse,
    summary="AI 自动检测代码",
    description=(
        "先做本地语法检查（Python 用 AST，C/Java 用 Tree-sitter 语法树），"
        "再调用大模型做深度检测。返回：\n"
        "- `errors`：语法错误与逻辑错误\n"
        "- `style`：代码风格建议（命名、缩进、注释）\n"
        "- `risks`：潜在的 Bug 与风险点\n"
        "- `advice`：面向学生的学习建议（通俗语言）\n"
        "- `score` / `level`：0-100 评分与等级\n\n"
        "每条问题都带 `source` 字段标明来自本地规则还是 AI，"
        "并给出具体行号与可执行的修改建议。检测结果会写入 SQLite。"
    ),
)
async def check_code(
    payload: CodeCheckRequest,
    session: DbSession,
    checker: CodeCheckerDep,
    x_session_id: str | None = Header(
        default=None,
        alias=SESSION_HEADER,
        description="网页上填过 API Key 时会带这个会话号，用来取用户自己的 Key",
    ),
) -> CodeCheckResponse:
    """检测一段代码，返回四类问题 + 评分，并落库。

    模型的选取：请求体里带了 `model_id` / `api_key`，或带了 `X-Session-Id`，
    就用**用户自己选的那个模型 + 他那把 Key**；一个都没带才回落到 `.env` 配置。
    """
    try:
        outcome = await checker.check_text(
            payload.code,
            language=payload.language,
            filename=payload.filename,
            session_id=x_session_id,
            model_id=payload.model_id,
            api_key=payload.api_key,
        )
    except UnsupportedLanguageError as exc:
        # 语言不支持是"请求不合法"，用 400 让前端能区分于服务端故障
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except CodeCheckError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    record = await CodeCheckService.save(
        session, outcome, code=payload.code, trace_id=current_trace_id()
    )

    return CodeCheckResponse(
        filename=outcome.filename,
        language=outcome.language,
        language_label=language_label(outcome.language),
        score=outcome.score,
        level=outcome.level,
        score_reason=outcome.score_reason,
        ai_score=outcome.ai_score,
        syntax_ok=outcome.local.syntax_ok,
        errors=[_to_read(item) for item in outcome.errors],
        style=[_to_read(item) for item in outcome.style],
        risks=[_to_read(item) for item in outcome.risks],
        advice=outcome.advice,
        highlights=outcome.highlights,
        summary=outcome.summary,
        local=LocalCheckRead(
            language=outcome.local.language,
            syntax_ok=outcome.local.syntax_ok,
            syntax_error=(
                SyntaxErrorRead(
                    line=outcome.local.syntax_error.line,
                    column=outcome.local.syntax_error.column,
                    message=outcome.local.syntax_error.message,
                    tool=outcome.local.syntax_error.tool,
                    raw=outcome.local.syntax_error.raw,
                )
                if outcome.local.syntax_error
                else None
            ),
            metrics=CodeMetricsRead(**outcome.local.metrics.to_dict()),
            duration_ms=round(outcome.local.duration_ms, 3),
        ),
        ai_available=outcome.ai_available,
        model=outcome.model,
        note=outcome.note,
        warnings=outcome.warnings,
        record_id=record.id,
        duration_ms=round(outcome.duration_ms, 3),
        ai_duration_ms=outcome.ai_duration_ms,
        checked_at=outcome.checked_at,
        trace_id=current_trace_id(),
    )


def _to_read(item) -> CodeIssueRead:  # noqa: ANN001 - Issue 类型在服务层定义
    """把服务层的 Issue 转成响应模型。"""
    return CodeIssueRead(
        line=item.line,
        severity=item.severity,
        category=item.category,
        title=item.title,
        detail=item.detail,
        suggestion=item.suggestion,
        source=item.source,
    )
