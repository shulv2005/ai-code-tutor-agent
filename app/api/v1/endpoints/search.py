"""混合检索接口：/api/v1/search

对应工作流的「检索层 (RAG)」：把自然语言 Issue 描述定位到具体代码符号。
检索结果自带源码片段与行号，可直接作为 Step 4（测试生成）/ Step 6（修复）的上下文。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from app.api.deps import DbSession, RetrievalServiceDep
from app.core.trace import current_trace_id
from app.models.repository import Repository
from app.schemas.retrieval import SearchHitRead, SearchRequest, SearchResponse
from app.services.retrieval.service import (
    IndexMissingError,
    RetrievalError,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "",
    response_model=SearchResponse,
    summary="混合检索代码",
    description=(
        "BM25 关键词召回 + FAISS 向量召回，用 RRF 融合后返回 Top-K 代码符号。"
        "需先对目标仓库建立索引（POST /api/v1/repositories/{id}/index）。"
    ),
)
async def search_code(
    payload: SearchRequest,
    session: DbSession,
    service: RetrievalServiceDep,
) -> SearchResponse:
    repository = await session.get(Repository, payload.repository_id)
    if repository is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="仓库不存在")

    try:
        result = await service.search(
            session,
            repository,
            payload.query,
            top_k=payload.top_k,
            language=payload.language,
            include_tests=payload.include_tests,
        )
    except IndexMissingError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except RetrievalError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return SearchResponse(
        query=result.query,
        repository_id=result.repository_id,
        total=len(result.hits),
        hits=[SearchHitRead.model_validate(hit, from_attributes=True) for hit in result.hits],
        timings_ms=result.timings_ms,
        stale=result.stale,
        notes=result.notes,
        trace_id=current_trace_id(),
    )
