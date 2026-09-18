"""AI 代码导师接口：/api/v1/tutor

面向前端「学生代码学习」页面的四个能力：
  POST /analyze   上传文件或贴代码 -> 自动识别语言 + 解析结构（不依赖大模型）
  POST /check     AI 检测：评分 + 问题清单 + 做得好的地方
  POST /comment   AI 生成通俗中文注释
  POST /fix       AI 自动改错：修正后代码 + 逐条修改说明
  GET  /history   历史记录列表 / 详情

设计说明：
- `/analyze` 完全不依赖大模型，因此**没配置模型时前端依然可用**，
  能演示文件上传、语言分类、代码结构展示；三个 AI 按钮会给出明确提示。
- 三个 AI 接口的结果都会写入历史记录，供课堂回顾。
- **模型可以在页面上选**：三个 AI 接口都接受可选的两个表单字段
  `model_id`（用哪个模型）与 `api_key`（用户自己填的 Key），
  另外认 `X-Session-Id` 请求头（Key 存在服务端内存会话里的时候用）。
  一个都不传时，就用后端 `.env` 里配好的默认模型——保证老用法完全不受影响。
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import (
    APIRouter,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    UploadFile,
    status,
)

from app.agents.tutor_agent import TutorAgent
from app.api.deps import ApiKeyManagerDep, AppSettingsDep, DbSession, TutorAgentDep
from app.core.api_key_manager import SESSION_HEADER, ApiKeyManager
from app.core.config import Settings
from app.core.llm_client import (
    LLMClientConfig,
    LLMConfigError,
    LLMError,
    LLMTimeoutError,
    dynamic_llm_client,
)
from app.core.trace import current_trace_id
from app.models.tutor import TutorRecord
from app.schemas.tutor import (
    AnalyzeResponse,
    CheckIssue,
    CheckResponse,
    CodeChange,
    CodeSymbolRead,
    CommentResponse,
    FixResponse,
    TutorHistoryResponse,
    TutorRecordDetail,
    TutorRecordRead,
    TutorStatusResponse,
)
from app.services.repo.language import (
    LANGUAGE_LABELS,
    SUFFIX_TO_LANGUAGE,
    supported_languages,
)
from app.services.repo.parser import parse_source
from app.services.text_utils import count_text_lines
from app.services.tutor_service import TutorService

logger = logging.getLogger(__name__)

router = APIRouter()

# 单次上传的代码大小上限：学生作业不会很大，超过基本是误传了文件
MAX_CODE_BYTES = 512 * 1024


# ---------------------------------------------------------------------------
# 公共逻辑
# ---------------------------------------------------------------------------
def _resolve_language(filename: str) -> str:
    """按文件名后缀识别语言；不支持的扩展名返回 unknown。"""
    detected = None
    lowered = filename.lower()
    for suffix, language in SUFFIX_TO_LANGUAGE.items():
        if lowered.endswith(suffix):
            detected = language
            break
    return detected or "unknown"


def _analyze(filename: str, code: str) -> AnalyzeResponse:
    """本地解析代码：识别语言、提取函数与类。不调用大模型。"""
    language = _resolve_language(filename)
    raw = code.encode("utf-8", errors="replace")

    notes: list[str] = []
    if language == "unknown":
        notes.append(
            "无法从文件名识别语言，目前支持：" + "、".join(sorted(SUFFIX_TO_LANGUAGE))
        )
        parsed_symbols: list[CodeSymbolRead] = []
        parse_error = None
        imports: list[str] = []
    else:
        parsed = parse_source(raw, filename, language)
        parsed_symbols = [
            CodeSymbolRead(
                name=symbol.name,
                qualified_name=symbol.qualified_name,
                kind=symbol.kind,
                start_line=symbol.start_line,
                end_line=symbol.end_line,
                signature=symbol.signature,
                complexity=symbol.complexity,
            )
            for symbol in parsed.symbols
        ]
        parse_error = parsed.parse_error
        imports = [item.module for item in parsed.imports]
        if parse_error:
            notes.append("代码存在语法错误，符号解析可能不完整，但仍可用于 AI 检测")

    return AnalyzeResponse(
        filename=filename,
        language=language,
        language_label=LANGUAGE_LABELS.get(language, language),
        line_count=count_text_lines(code),
        char_count=len(code),
        size_bytes=len(raw),
        parse_error=parse_error,
        symbols=parsed_symbols,
        imports=imports,
        notes=notes,
        trace_id=current_trace_id(),
    )


async def _read_code(
    session_file: UploadFile | None, code: str | None, filename: str | None
) -> tuple[str, str]:
    """从「上传文件」或「表单里的代码文本」中取出代码与文件名。"""
    if session_file is not None and session_file.filename:
        raw = await session_file.read()
        if len(raw) > MAX_CODE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"文件过大（{len(raw) // 1024} KB），上限为 {MAX_CODE_BYTES // 1024} KB",
            )
        if not raw.strip():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="上传的文件是空的")
        # 优先 utf-8，失败再退化，避免一个 GBK 编码的作业直接报错
        for encoding in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
            try:
                return raw.decode(encoding), session_file.filename
            except UnicodeDecodeError:
                continue
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="无法识别文件编码"
        )

    if code and code.strip():
        return code, (filename or "untitled.py")

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="请上传代码文件，或在表单中提供 code 字段",
    )


def _ai_error(exc: Exception) -> HTTPException:
    """把 LLM 异常映射成前端能看懂的中文提示。"""
    if isinstance(exc, LLMConfigError):
        # 两类情况要分开说：
        #   1. 学生在页面上选了模型但没填 Key —— 用异常自带的那句话（更具体）；
        #   2. 后端 .env 一个 Key 都没配 —— 用下面这句（告诉老师/同学去哪配）。
        message = str(exc).strip()
        if "网页" in message:
            return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=message)
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI 功能尚未启用：请在页面「模型设置」里填入 API Key，"
            "或在后端 .env 中配置 LLM__API_KEY 后重启服务。",
        )
    if isinstance(exc, LLMTimeoutError):
        return HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="AI 模型响应超时，请稍后重试或换一段更短的代码。",
        )
    if isinstance(exc, LLMError):
        return HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=f"AI 调用失败：{exc}"
        )
    return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@asynccontextmanager
async def _active_agent(
    default_agent: TutorAgent,
    settings: Settings,
    manager: ApiKeyManager,
    session_id: str | None,
    model_id: str | None,
    api_key: str | None,
) -> AsyncIterator[TutorAgent]:
    """挑出这次请求真正要用的 Agent。

    三种情况：
      1. 三个参数一个都没传 → 直接用注入进来的默认 Agent（= `.env` 里配的模型）。
         这条路径同时也是**测试的注入点**（`dependency_overrides`）。
      2. 传了 `api_key`（页面本次提交带的）→ 现造一个客户端，用它。
      3. 只传了 `X-Session-Id` / `model_id` → 从内存会话里取出对应的 Key 再造。

    为什么用 `dynamic_llm_client`（用完即关）：用户的 Key 只该在这次请求里活着，
    不能跟着全局单例的连接池一直留在内存中。
    """
    if not (session_id or model_id or api_key):
        yield default_agent
        return

    # 会话 Key / 页面 Key / 后端配置 Key 的优先级由 ApiKeyManager 统一决定
    credential = manager.resolve(session_id, model_id, api_key)
    config = LLMClientConfig.from_llm_settings(
        settings.llm, credential.model_id, credential.api_key
    )
    logger.info(
        "本次请求使用模型=%s（Key 来源=%s，Key=%s）",
        config.model_name,
        credential.source,
        config.masked_key(),
    )
    async with dynamic_llm_client(config) as client:
        yield TutorAgent(settings, client)


# ---------------------------------------------------------------------------
# 状态
# ---------------------------------------------------------------------------
@router.get(
    "/status",
    response_model=TutorStatusResponse,
    summary="AI 导师模块状态",
    description="前端据此决定三个 AI 按钮是否可点击，以及提示学生配置模型。",
)
async def tutor_status(agent: TutorAgentDep, session: DbSession) -> TutorStatusResponse:
    from sqlalchemy import func, select

    count = (await session.execute(select(func.count(TutorRecord.id)))).scalar_one()
    languages = supported_languages()
    return TutorStatusResponse(
        ai_available=agent.llm.configured,
        model=agent.llm.model,
        supported_languages={name: LANGUAGE_LABELS.get(name, name) for name in languages},
        max_upload_bytes=MAX_CODE_BYTES,
        history_count=count,
    )


# ---------------------------------------------------------------------------
# 分析（不依赖大模型）
# ---------------------------------------------------------------------------
@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="分析代码（识别语言 + 解析结构）",
    description=(
        "支持两种提交方式：上传文件（multipart/form-data）或直接提交代码文本。"
        "本接口不调用大模型，因此在未配置 AI 时依然可用。"
    ),
)
async def analyze_code(
    session: DbSession,
    file: UploadFile | None = File(default=None, description="代码文件（.c/.java/.py 等）"),
    code: str | None = Form(default=None, description="也可以直接提交代码文本"),
    filename: str | None = Form(default=None, description="代码文本对应的文件名"),
) -> AnalyzeResponse:
    content, name = await _read_code(file, code, filename)
    result = _analyze(name, content)
    # 记录一次「分析」历史，让学生看到自己看过哪些文件
    await TutorService.record(
        session,
        filename=name,
        language=result.language,
        code=content,
        action="analyze",
        summary=f"分析了 {result.line_count} 行代码，识别为 {result.language_label}",
        result_json=None,
    )
    return result


# ---------------------------------------------------------------------------
# AI 检测
# ---------------------------------------------------------------------------
@router.post(
    "/check",
    response_model=CheckResponse,
    summary="AI 检测代码",
    description="返回评分、问题清单（含修改建议）与做得好的地方。",
)
async def check_code(
    session: DbSession,
    agent: TutorAgentDep,
    settings: AppSettingsDep,
    manager: ApiKeyManagerDep,
    file: UploadFile | None = File(default=None),
    code: str | None = Form(default=None),
    filename: str | None = Form(default=None),
    model_id: str | None = Form(default=None, description="用哪个模型（页面下拉框选的）"),
    api_key: str | None = Form(default=None, description="用户自己填的 API Key（可空）"),
    x_session_id: str | None = Header(default=None, alias=SESSION_HEADER),
) -> CheckResponse:
    content, name = await _read_code(file, code, filename)
    language = _resolve_language(name)
    if language == "unknown":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="无法识别代码语言，请使用 .c / .java / .py 等受支持的扩展名。",
        )

    try:
        async with _active_agent(
            agent, settings, manager, x_session_id, model_id, api_key
        ) as active:
            result = await active.check(content, filename=name, language=language)
    except Exception as exc:  # noqa: BLE001 - 统一映射为友好错误
        raise _ai_error(exc) from exc

    payload = result.payload
    issues = [
        CheckIssue(
            line=item.get("line") if isinstance(item.get("line"), int) else None,
            severity=item.get("severity", "info"),
            title=str(item.get("title", "")).strip() or "（未命名问题）",
            detail=str(item.get("detail", "")),
            suggestion=str(item.get("suggestion", "")),
        )
        for item in payload.get("issues", [])
    ]
    score = float(payload.get("score", 0.0))
    record = await TutorService.record(
        session,
        filename=name,
        language=language,
        code=content,
        action="check",
        score=score,
        summary=str(payload.get("summary", ""))[:500],
        result_json=json.dumps(
            {"issues": [i.model_dump() for i in issues],
             "highlights": payload.get("highlights", [])},
            ensure_ascii=False,
        ),
        model=result.model,
        duration_ms=result.duration_ms,
    )

    return CheckResponse(
        filename=name,
        language=language,
        score=score,
        level=TutorService.score_level(score),
        summary=str(payload.get("summary", "")),
        issues=issues,
        highlights=payload.get("highlights", []),
        model=result.model,
        record_id=record.id,
        trace_id=current_trace_id(),
    )


# ---------------------------------------------------------------------------
# 生成注释
# ---------------------------------------------------------------------------
@router.post(
    "/comment",
    response_model=CommentResponse,
    summary="生成中文注释",
    description="给学生代码加上通俗易懂的中文注释，帮助理解代码逻辑。",
)
async def comment_code(
    session: DbSession,
    agent: TutorAgentDep,
    settings: AppSettingsDep,
    manager: ApiKeyManagerDep,
    file: UploadFile | None = File(default=None),
    code: str | None = Form(default=None),
    filename: str | None = Form(default=None),
    model_id: str | None = Form(default=None, description="用哪个模型（页面下拉框选的）"),
    api_key: str | None = Form(default=None, description="用户自己填的 API Key（可空）"),
    x_session_id: str | None = Header(default=None, alias=SESSION_HEADER),
) -> CommentResponse:
    content, name = await _read_code(file, code, filename)
    language = _resolve_language(name)

    try:
        async with _active_agent(
            agent, settings, manager, x_session_id, model_id, api_key
        ) as active:
            result = await active.comment(content, filename=name, language=language)
    except Exception as exc:  # noqa: BLE001
        raise _ai_error(exc) from exc

    commented = result.payload.get("commented_code", content)
    summary = str(result.payload.get("summary", ""))
    record = await TutorService.record(
        session,
        filename=name,
        language=language,
        code=content,
        action="comment",
        summary=summary[:500],
        result_json=json.dumps({"commented_code": commented}, ensure_ascii=False),
        model=result.model,
        duration_ms=result.duration_ms,
    )

    return CommentResponse(
        filename=name,
        language=language,
        commented_code=commented,
        summary=summary,
        model=result.model,
        record_id=record.id,
        trace_id=current_trace_id(),
    )


# ---------------------------------------------------------------------------
# 自动改错
# ---------------------------------------------------------------------------
@router.post(
    "/fix",
    response_model=FixResponse,
    summary="AI 自动改错",
    description="找出代码错误，返回修正后的完整代码与逐条修改说明。",
)
async def fix_code(
    session: DbSession,
    agent: TutorAgentDep,
    settings: AppSettingsDep,
    manager: ApiKeyManagerDep,
    file: UploadFile | None = File(default=None),
    code: str | None = Form(default=None),
    filename: str | None = Form(default=None),
    model_id: str | None = Form(default=None, description="用哪个模型（页面下拉框选的）"),
    api_key: str | None = Form(default=None, description="用户自己填的 API Key（可空）"),
    x_session_id: str | None = Header(default=None, alias=SESSION_HEADER),
) -> FixResponse:
    content, name = await _read_code(file, code, filename)
    language = _resolve_language(name)

    try:
        async with _active_agent(
            agent, settings, manager, x_session_id, model_id, api_key
        ) as active:
            result = await active.fix(content, filename=name, language=language)
    except Exception as exc:  # noqa: BLE001
        raise _ai_error(exc) from exc

    payload = result.payload
    changes = [
        CodeChange(
            line=item.get("line") if isinstance(item.get("line"), int) else None,
            original=str(item.get("original", "")),
            fixed=str(item.get("fixed", "")),
            reason=str(item.get("reason", "")),
        )
        for item in payload.get("changes", [])
    ]
    fixed_code = payload.get("fixed_code", content)
    summary = str(payload.get("summary", ""))
    had_error = bool(payload.get("had_error", False))

    record = await TutorService.record(
        session,
        filename=name,
        language=language,
        code=content,
        action="fix",
        summary=summary[:500],
        result_json=json.dumps(
            {"fixed_code": fixed_code, "changes": [c.model_dump() for c in changes],
             "had_error": had_error},
            ensure_ascii=False,
        ),
        model=result.model,
        duration_ms=result.duration_ms,
    )

    return FixResponse(
        filename=name,
        language=language,
        fixed_code=fixed_code,
        changes=changes,
        summary=summary,
        had_error=had_error,
        model=result.model,
        record_id=record.id,
        trace_id=current_trace_id(),
    )


# ---------------------------------------------------------------------------
# 历史记录
# ---------------------------------------------------------------------------
@router.get(
    "/history",
    response_model=TutorHistoryResponse,
    summary="历史记录列表",
    description="按时间倒序列出学生之前的检测/注释/改错记录。",
)
async def list_history(
    session: DbSession,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    filename: str | None = Query(default=None, description="按文件名筛选"),
    action: str | None = Query(default=None, description="按操作类型筛选：check/comment/fix"),
) -> TutorHistoryResponse:
    items, total = await TutorService.list_records(
        session, limit=limit, offset=offset, filename=filename, action=action
    )
    return TutorHistoryResponse(
        total=total, items=[TutorRecordRead.model_validate(item) for item in items]
    )


@router.get(
    "/history/{record_id}",
    response_model=TutorRecordDetail,
    summary="历史记录详情",
    description="返回某次记录的原始代码与完整结果，前端可用于回看。",
)
async def get_history_detail(record_id: int, session: DbSession) -> TutorRecordDetail:
    record = await TutorService.get_record(session, record_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="记录不存在")
    return TutorRecordDetail.model_validate(record)


@router.delete(
    "/history/{record_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除一条历史记录",
)
async def delete_history(record_id: int, session: DbSession) -> None:
    if not await TutorService.delete_record(session, record_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="记录不存在")
