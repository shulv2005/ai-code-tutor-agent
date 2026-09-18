"""Agent 接口：/api/v1/agent

当前提供测试生成（Step 4）。后续修复 Agent（Step 6）与编排（Step 7）挂在这里。
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from app.agents.context import (
    ContextError,
    context_from_query,
    context_from_snippet,
    context_from_symbol,
    enrich_context,
)
from app.agents.dto import CodeContext
from app.agents.patch import parse_patch_files, prepare_worktree
from app.agents.prompts import PROMPT_VERSION
from app.agents.test_agent import TestAgentError, TestGenerationFailed
from app.api.deps import (
    AutoFixPlannerDep,
    DbSession,
    RepoServiceDep,
    RetrievalServiceDep,
    TestAgentDep,
)
from app.core.config import Settings, get_settings
from app.core.llm_client import LLMConfigError, LLMError, LLMTimeoutError
from app.core.trace import current_trace_id
from app.models.repository import Repository
from app.schemas.agent import (
    AutoFixRequest,
    AutoFixResponse,
    CodeContextRead,
    LLMStatusResponse,
    LLMUsageRead,
    LoopIterationRead,
    PrDraftRead,
    TestGenerationRequest,
    TestGenerationResponse,
)
from app.services.pr_draft import build_issue_comment, build_pr_draft
from app.services.repo.git_service import (
    InvalidRepoUrlError,
    RepoCloneError,
)
from app.services.retrieval.service import (
    IndexEmptyError,
    IndexMissingError,
    RetrievalError,
)
from app.services.sandbox import SandboxUnavailable

logger = logging.getLogger(__name__)

router = APIRouter()

# 422 常量在较新 starlette 中已更名（UNPROCESSABLE_ENTITY -> UNPROCESSABLE_CONTENT），
# 直接用数字避免版本间命名差异带来的弃用告警。
HTTP_422_UNPROCESSABLE = 422


@router.get(
    "/status",
    response_model=LLMStatusResponse,
    summary="LLM 配置状态",
    description="前端可用它提示用户先配置模型；未配置时生成接口会返回 503。",
)
async def llm_status(agent: TestAgentDep, settings: object = None) -> LLMStatusResponse:
    from app.core.config import get_settings

    config: Settings = get_settings()
    return LLMStatusResponse(
        configured=agent.llm.configured,
        model=agent.llm.model,
        base_url=config.llm.base_url,
        provider=config.llm.provider,
        max_retries=config.llm.max_retries,
        timeout_seconds=config.llm.timeout_seconds,
        prompt_version=PROMPT_VERSION,
    )


async def _resolve_context(
    payload: TestGenerationRequest,
    session: DbSession,
    retrieval: RetrievalServiceDep,
) -> tuple[CodeContext, Repository | None]:
    """把请求归一成 CodeContext。"""
    repository: Repository | None = None
    if payload.repository_id is not None:
        repository = await session.get(Repository, payload.repository_id)
        if repository is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"仓库不存在: {payload.repository_id}",
            )

    if payload.code and payload.code.strip():
        context = context_from_snippet(
            payload.code,
            file_path=payload.file_path,
            symbol_name=payload.symbol_name,
            language=payload.language,
        )
    elif payload.symbol_id is not None:
        assert repository is not None  # 已由 model_validator 保证
        context = await context_from_symbol(session, repository, payload.symbol_id)
    else:
        assert repository is not None and payload.query
        context = await context_from_query(session, repository, retrieval, payload.query)

    context = await enrich_context(
        session,
        repository,
        retrieval if repository is not None else None,
        context,
        include_related=payload.include_related,
        include_existing_tests=payload.include_existing_tests,
    )
    return context, repository


@router.post(
    "/generate_test",
    response_model=TestGenerationResponse,
    summary="生成 pytest 测试",
    description=(
        "接收代码上下文（直接片段 / 仓库符号 / 自然语言检索），调用 LLM 生成 pytest 测试，"
        "提取纯 Python 代码并写入沙箱目录 sandbox_repo/tests/test_generated.py。\n\n"
        "指定 repository_id 时会把仓库副本一并放入沙箱，使生成的测试能够导入真实模块；"
        "这样产出的目录可直接交给 POST /api/v1/sandbox/run 执行。"
    ),
)
async def generate_test(
    payload: TestGenerationRequest,
    session: DbSession,
    retrieval: RetrievalServiceDep,
    agent: TestAgentDep,
) -> TestGenerationResponse:
    try:
        context, repository = await _resolve_context(payload, session, retrieval)
    except ContextError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except IndexMissingError as exc:
        # 查询模式依赖 Step 3 检索索引，未建索引时给出可操作的提示
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except RetrievalError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    # 先定 run_id：下面要按它在沙箱里铺好仓库副本，再让 Agent 往里写测试
    run_id = uuid.uuid4().hex[:16]

    # 关键整合点：若上下文来自仓库，就把仓库副本放进沙箱目录。
    # 否则生成的测试会 `from <module> import ...`，而沙箱里没有该模块，
    # pytest 直接以 collection error（exit=2）失败——Step 4 与 Step 5 无法串联。
    if repository is not None and payload.save_to_sandbox:
        try:
            await asyncio.to_thread(
                prepare_worktree,
                Path(repository.local_path),
                agent.sandbox_root(run_id),
                exclude=set(get_settings().repository.excluded_dirs),
            )
        except Exception as exc:  # noqa: BLE001 - 副本准备失败不阻断生成
            logger.warning("准备沙箱仓库副本失败：%s", exc, exc_info=True)
        else:
            # 仓库可用时，提示词切换为"导入真实模块"，避免内联替身
            context.module_available = True

    try:
        result = await agent.generate(
            context,
            max_attempts=payload.max_attempts,
            save_to_sandbox=payload.save_to_sandbox,
            run_id=run_id,
        )
    except LLMConfigError as exc:
        # 未配置模型属于服务端未就绪，用 503 让前端明确区分于参数错误
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except LLMTimeoutError as exc:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(exc)) from exc
    except TestGenerationFailed as exc:
        raise HTTPException(status_code=HTTP_422_UNPROCESSABLE, detail=str(exc)) from exc
    except (LLMError, TestAgentError) as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    target = result.target
    return TestGenerationResponse(
        run_id=result.run_id,
        target=CodeContextRead(
            path=target.path,
            qualified_name=target.qualified_name,
            kind=target.kind,
            language=target.language,
            signature=target.signature,
            docstring=target.docstring,
            start_line=target.start_line,
            end_line=target.end_line,
            repository=target.repository,
            symbol_id=target.symbol_id,
            related_count=len(target.related),
            existing_test_count=len(target.existing_tests),
        ),
        test_code=result.test.code,
        test_functions=result.test.test_functions,
        model=result.test.model,
        prompt_version=PROMPT_VERSION,
        attempts=result.test.attempts,
        warnings=result.test.warnings,
        usage=LLMUsageRead(
            prompt_tokens=result.test.usage.prompt_tokens,
            completion_tokens=result.test.usage.completion_tokens,
            total_tokens=result.test.usage.total_tokens,
        ),
        saved_path=str(result.saved_path) if result.saved_path else None,
        sandbox_dir=(
            str(agent.sandbox_root(result.run_id)) if result.saved_path else None
        ),
        timings_ms=result.timings_ms,
        trace_id=current_trace_id(),
    )


# ---------------------------------------------------------------------------
# 自动修复（Step 6/7）
# ---------------------------------------------------------------------------
async def _resolve_autofix_target(
    payload: AutoFixRequest,
    session: DbSession,
    repo_service: RepoServiceDep,
    retrieval: RetrievalServiceDep,
) -> tuple[Repository, CodeContext]:
    """把 auto_fix 请求归一成（仓库, 待修复上下文）。"""
    # 1) 定位仓库：给 URL 就克隆 + 解析 + 建索引
    if payload.repository_id is not None:
        repository = await session.get(Repository, payload.repository_id)
        if repository is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"仓库不存在: {payload.repository_id}",
            )
    else:
        assert payload.repository_url
        try:
            repository, _stats = await repo_service.register_and_index(
                session, payload.repository_url
            )
        except InvalidRepoUrlError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except RepoCloneError as exc:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    # 2) 确保检索索引存在（Issue 描述需要用它定位代码）
    if payload.symbol_id is None:
        try:
            await retrieval.build_index(session, repository)
        except IndexEmptyError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        except RetrievalError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    # 3) 定位待修复代码
    try:
        if payload.symbol_id is not None:
            context = await context_from_symbol(session, repository, payload.symbol_id)
        else:
            assert payload.issue
            context = await context_from_query(session, repository, retrieval, payload.issue)
    except ContextError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except IndexMissingError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except RetrievalError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return repository, context


@router.post(
    "/auto_fix",
    response_model=AutoFixResponse,
    summary="自动修复（完整链路）",
    description=(
        "接收仓库 URL 与 Issue 描述，依次执行：克隆 → 解析 → 建检索索引 → 定位代码 → "
        "生成测试 → 沙箱执行 → 失败则生成补丁并重跑，最多 max_attempts 轮。\n\n"
        "所有补丁只应用到隔离工作副本，**原始克隆仓库全程只读**。"
    ),
)
async def auto_fix(
    payload: AutoFixRequest,
    session: DbSession,
    repo_service: RepoServiceDep,
    retrieval: RetrievalServiceDep,
    planner: AutoFixPlannerDep,
) -> AutoFixResponse:
    repository, context = await _resolve_autofix_target(
        payload, session, repo_service, retrieval
    )

    local_path = Path(repository.local_path)
    if not local_path.exists():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"本地克隆不存在（{local_path}），请重新克隆后重试",
        )

    try:
        result = await planner.run(
            context,
            local_path,
            max_attempts=payload.max_attempts,
            keep_worktree=payload.keep_worktree,
        )
    except LLMConfigError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except LLMTimeoutError as exc:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except SandboxUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except Exception as exc:  # noqa: BLE001 - 兜底，避免 500 无上下文
        logger.exception("自动修复失败")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"自动修复失败：{type(exc).__name__}: {exc}",
        ) from exc

    target = result.target
    final = result.final_test
    iteration_payloads = [item.to_dict() for item in result.iterations]

    # 链路终点：根据真实执行结果生成 PR 草稿与 Issue 评论
    draft = build_pr_draft(
        issue=payload.issue or f"自动修复 {target.display_name if target else ''}".strip(),
        repository=f"{repository.owner}/{repository.name}",
        target_path=target.path if target else "",
        target_name=target.display_name if target else "",
        success=result.success,
        status=result.status,
        message=result.message,
        diff=result.final_diff,
        files_changed=result.changed_files,
        iterations=iteration_payloads,
        passed=final.passed if final else 0,
        failed=final.failed if final else 0,
        errors=final.errors if final else 0,
        coverage_percent=(
            round(final.coverage.percent_covered, 2) if final and final.coverage else None
        ),
        generated_test=result.generated_test,
        run_id=result.run_id,
        trace_id=current_trace_id(),
        base_branch=repository.default_branch or "main",
    )
    comment = build_issue_comment(
        status=result.status,
        success=result.success,
        message=result.message,
        target_path=target.path if target else "",
        target_name=target.display_name if target else "",
        passed=final.passed if final else 0,
        failed=final.failed if final else 0,
        errors=final.errors if final else 0,
        coverage_percent=(
            round(final.coverage.percent_covered, 2) if final and final.coverage else None
        ),
        run_id=result.run_id,
    )

    return AutoFixResponse(
        run_id=result.run_id,
        status=result.status,
        success=result.success,
        attempts=len(result.iterations),
        message=result.message,
        target=(
            CodeContextRead(
                path=target.path,
                qualified_name=target.qualified_name,
                kind=target.kind,
                language=target.language,
                signature=target.signature,
                docstring=target.docstring,
                start_line=target.start_line,
                end_line=target.end_line,
                repository=target.repository,
                symbol_id=target.symbol_id,
                related_count=len(target.related),
                existing_test_count=len(target.existing_tests),
            )
            if target
            else None
        ),
        generated_test=result.generated_test,
        final_diff=result.final_diff,
        changed_files=result.changed_files,
        worktree=result.worktree,
        passed=final.passed if final else 0,
        failed=final.failed if final else 0,
        errors=final.errors if final else 0,
        coverage_percent=(
            round(final.coverage.percent_covered, 2)
            if final and final.coverage
            else None
        ),
        iterations=[
            LoopIterationRead(
                index=item.index,
                passed=item.test_result.passed if item.test_result else 0,
                failed=item.test_result.failed if item.test_result else 0,
                errors=item.test_result.errors if item.test_result else 0,
                exit_code=item.test_result.exit_code if item.test_result else None,
                category=item.proposal.category if item.proposal else None,
                analysis=item.proposal.analysis if item.proposal else None,
                patch_files=(
                    [change.path for change in parse_patch_files(item.proposal.patch)]
                    if item.proposal and item.proposal.patch
                    else []
                ),
                patch_applied=item.patch_applied,
                patch_error=item.patch_error,
                note=item.note,
            )
            for item in result.iterations
        ],
        pr_draft=PrDraftRead(
            title=draft.title,
            body=draft.body,
            branch_name=draft.branch_name,
            commit_message=draft.commit_message,
            base_branch=draft.base_branch,
            files_changed=draft.files_changed,
            additions=draft.additions,
            deletions=draft.deletions,
            is_draft=draft.is_draft,
            labels=draft.labels,
            verified=draft.verified,
            warnings=draft.warnings,
        ),
        issue_comment=comment,
        timings_ms=result.timings_ms,
        trace_id=current_trace_id(),
    )
