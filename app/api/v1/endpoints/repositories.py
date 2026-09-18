"""仓库解析接口：/api/v1/repositories

对应工作流的「输入层 + 解析层」：接收仓库地址，完成克隆与代码结构解析。

说明：当前克隆/索引在请求内同步完成（受 repository.clone_timeout_seconds 约束）。
Step 7 引入任务队列后会改为「先返回 202 + task_id，后台执行」。
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from app.api.deps import DbSession, RepoServiceDep, RetrievalServiceDep
from app.core.trace import current_trace_id, trace_span
from app.models.code import CodeFile
from app.models.repository import Repository
from app.schemas.repository import (
    CodeFileRead,
    IndexStatsRead,
    RepositoryCreateRequest,
    RepositoryIndexResponse,
    RepositoryListResponse,
    RepositoryRead,
    RepositoryStructureResponse,
)
from app.schemas.retrieval import IndexBuildResponse
from app.services.repo.git_service import (
    InvalidRepoUrlError,
    RepoCloneError,
    RepositoryError,
)
from app.services.repo.language import supported_languages
from app.services.retrieval.service import IndexEmptyError, RetrievalError

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/languages",
    summary="当前支持的解析语言",
    description="返回本环境真正可用的语言（对应语法包已安装）。",
)
async def list_languages() -> dict[str, list[str]]:
    return {"languages": supported_languages()}


@router.post(
    "",
    response_model=RepositoryIndexResponse,
    status_code=status.HTTP_201_CREATED,
    summary="注册并索引仓库",
    description="克隆远程仓库（GitPython），用 AST / Tree-sitter 解析代码结构并落库。",
)
async def create_repository(
    payload: RepositoryCreateRequest,
    session: DbSession,
    service: RepoServiceDep,
) -> RepositoryIndexResponse:
    try:
        repository, stats = await service.register_and_index(
            session, payload.url, branch=payload.branch, force=payload.force
        )
    except InvalidRepoUrlError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RepoCloneError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except RepositoryError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return RepositoryIndexResponse(
        repository=RepositoryRead.model_validate(repository),
        stats=IndexStatsRead.model_validate(stats, from_attributes=True),
        trace_id=current_trace_id(),
    )


@router.get("", response_model=RepositoryListResponse, summary="仓库列表")
async def list_repositories(
    session: DbSession,
    service: RepoServiceDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> RepositoryListResponse:
    total = (await session.execute(select(func.count(Repository.id)))).scalar_one()
    items = await service.list_repositories(session, limit=limit, offset=offset)
    return RepositoryListResponse(
        total=total,
        items=[RepositoryRead.model_validate(item) for item in items],
    )


@router.get("/{repository_id}", response_model=RepositoryRead, summary="仓库详情")
async def get_repository(repository_id: int, session: DbSession) -> RepositoryRead:
    repository = await session.get(Repository, repository_id)
    if repository is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="仓库不存在")
    return RepositoryRead.model_validate(repository)


@router.get(
    "/{repository_id}/structure",
    response_model=RepositoryStructureResponse,
    summary="仓库代码结构",
    description="返回按文件聚合的符号结构，供前端展示与后续检索/Agent 使用。",
)
async def get_repository_structure(
    repository_id: int,
    session: DbSession,
    language: Annotated[str | None, Query(description="按语言过滤，如 python")] = None,
    path_prefix: Annotated[str | None, Query(description="按路径前缀过滤，如 src/")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> RepositoryStructureResponse:
    repository = await session.get(Repository, repository_id)
    if repository is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="仓库不存在")

    with trace_span(
        "api.repository_structure",
        kind="http",
        payload={"repository_id": repository_id, "language": language},
    ):
        stmt = select(CodeFile).where(CodeFile.repository_id == repository_id)
        if language:
            stmt = stmt.where(CodeFile.language == language)
        if path_prefix:
            stmt = stmt.where(CodeFile.path.startswith(path_prefix))
        stmt = (
            stmt.options(selectinload(CodeFile.symbols))
            .order_by(CodeFile.path)
            .limit(limit)
            .offset(offset)
        )
        files = list((await session.execute(stmt)).scalars().all())

        # 语言分布按整仓库统计，不受分页影响
        language_rows = await session.execute(
            select(CodeFile.language, func.count(CodeFile.id))
            .where(CodeFile.repository_id == repository_id)
            .group_by(CodeFile.language)
        )

        return RepositoryStructureResponse(
            repository_id=repository_id,
            total_files=repository.file_count,
            total_symbols=repository.symbol_count,
            languages={name: count for name, count in language_rows.all()},
            files=[CodeFileRead.model_validate(item) for item in files],
        )


@router.post(
    "/{repository_id}/reindex",
    response_model=RepositoryIndexResponse,
    summary="重新解析仓库",
    description="不重新克隆，仅对本地已有副本重新遍历解析（改了配置或解析器后使用）。",
)
async def reindex_repository(
    repository_id: int,
    session: DbSession,
    service: RepoServiceDep,
) -> RepositoryIndexResponse:
    repository = await session.get(Repository, repository_id)
    if repository is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="仓库不存在")

    try:
        stats = await service.reindex(session, repository)
    except FileNotFoundError as exc:
        # 本地副本丢失：提示调用方用 force 重新克隆
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"{exc}（可调用 POST /repositories 并传 force=true 重新克隆）",
        ) from exc

    return RepositoryIndexResponse(
        repository=RepositoryRead.model_validate(repository),
        stats=IndexStatsRead.model_validate(stats, from_attributes=True),
        trace_id=current_trace_id(),
    )


@router.post(
    "/{repository_id}/index",
    response_model=IndexBuildResponse,
    summary="构建检索索引",
    description="对仓库符号建立 BM25 + FAISS 向量索引，供 POST /api/v1/search 使用。",
)
async def build_search_index(
    repository_id: int,
    session: DbSession,
    service: RetrievalServiceDep,
    force: Annotated[bool, Query(description="为 true 时忽略已有索引强制重建")] = False,
) -> IndexBuildResponse:
    repository = await session.get(Repository, repository_id)
    if repository is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="仓库不存在")

    try:
        stats = await service.build_index(session, repository, force=force)
    except IndexEmptyError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except RetrievalError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return IndexBuildResponse(
        repository_id=stats.repository_id,
        chunk_count=stats.chunk_count,
        vector_count=stats.vector_count,
        dimension=stats.dimension,
        embedder=stats.embedder,
        model=stats.model,
        duration_ms=round(stats.duration_ms, 2),
        skipped=stats.skipped,
        notes=stats.notes,
        trace_id=current_trace_id(),
    )
