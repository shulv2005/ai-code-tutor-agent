"""检索编排：建索引（BM25 + FAISS）与混合检索。

整体链路：
    CodeSymbol + 源码切片
        -> CodeChunk
        -> [BM25 关键词索引]  ┐
        -> [FAISS 向量索引]   ┴-> RRF 融合 -> Top-K 命中（含源码片段）

索引按仓库独立存放（data/index/repo_<id>/），并用 head_commit 做陈旧检测。
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.trace import trace_span
from app.models.code import CodeFile, CodeSymbol
from app.models.repository import Repository
from app.services.retrieval.chunker import build_chunks, is_test_chunk, slice_symbol
from app.services.retrieval.dto import IndexBuildStats, SearchHit, SearchResult
from app.services.retrieval.embedder import get_shared_embedder
from app.services.retrieval.hybrid import RankedList, reciprocal_rank_fusion, weighted_score_fusion
from app.services.retrieval.keyword_index import BM25Index
from app.services.retrieval.vector_index import VectorIndex, read_meta, write_meta

logger = logging.getLogger(__name__)

VECTOR_FILE = "vectors.faiss"
BM25_FILE = "bm25.json"
META_FILE = "meta.json"


class RetrievalError(RuntimeError):
    """检索层基类异常。"""


class IndexMissingError(RetrievalError):
    """仓库尚未建立检索索引。"""


class IndexEmptyError(RetrievalError):
    """仓库没有任何可索引的符号。"""


@dataclass(slots=True)
class _IndexBundle:
    """内存中的一套索引。"""

    vectors: VectorIndex
    bm25: BM25Index
    meta: dict
    # 测试符号集合：检索时默认降权，避免测试用例淹没实现代码
    test_symbol_ids: frozenset[int] = frozenset()


class RetrievalService:
    """混合检索服务。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._retrieval = settings.retrieval
        # 索引常驻内存，避免每次检索都从磁盘反序列化
        self._cache: dict[int, _IndexBundle] = {}
        self._cache_lock = threading.Lock()

    # -- 路径 -------------------------------------------------------------
    def index_dir(self, repository_id: int) -> Path:
        """某个仓库的索引目录。"""
        return self._retrieval.index_path / f"repo_{repository_id}"

    @property
    def embedder(self):  # noqa: ANN201 - 返回协议类型，避免循环导入
        """共享的嵌入后端。"""
        return get_shared_embedder(self._retrieval)

    # -- 建索引 -----------------------------------------------------------
    async def build_index(
        self,
        session: AsyncSession,
        repository: Repository,
        *,
        force: bool = False,
    ) -> IndexBuildStats:
        """为仓库构建 BM25 + 向量索引。

        Args:
            force: 为 True 时即使索引已是最新也重建。
        """
        started = time.perf_counter()
        embedder = self.embedder

        with trace_span(
            "retrieval.build_index",
            kind="rag",
            payload={"repository_id": repository.id, "force": force},
            metadata={"embedder": embedder.name, "model": embedder.model_id},
        ) as span:
            meta_path = self.index_dir(repository.id) / META_FILE
            existing_meta = read_meta(meta_path)
            if (
                not force
                and existing_meta is not None
                and self._meta_is_current(existing_meta, repository, embedder)
            ):
                span.set_metadata(reused=True)
                return IndexBuildStats(
                    repository_id=repository.id,
                    chunk_count=int(existing_meta.get("chunk_count", 0)),
                    vector_count=int(existing_meta.get("chunk_count", 0)),
                    dimension=int(existing_meta.get("dimension", embedder.dimension)),
                    embedder=str(existing_meta.get("embedder", embedder.name)),
                    model=str(existing_meta.get("model", embedder.model_id)),
                    duration_ms=(time.perf_counter() - started) * 1000,
                    skipped=True,
                    notes=["索引已是最新（head_commit 未变化）"],
                )

            pairs = await self._load_symbols(session, repository.id)
            if not pairs:
                raise IndexEmptyError(
                    "该仓库没有可索引的代码符号，请先完成仓库解析（POST /repositories）"
                )

            chunks = await asyncio.to_thread(
                build_chunks,
                pairs,
                Path(repository.local_path),
                max_chunk_chars=self._retrieval.max_chunk_chars,
            )
            if not chunks:
                raise IndexEmptyError("分块结果为空，无法建立索引")

            # 1) BM25：分词 + 建倒排
            bm25 = BM25Index()
            await asyncio.to_thread(
                bm25.build,
                [chunk.symbol_id for chunk in chunks],
                [chunk.to_document(max_chars=self._retrieval.max_chunk_chars) for chunk in chunks],
            )

            # 2) 向量：批量嵌入（CPU 密集，放线程池）
            documents = [
                chunk.to_document(max_chars=self._retrieval.max_chunk_chars) for chunk in chunks
            ]
            vectors = await asyncio.to_thread(embedder.embed_documents, documents)

            vector_index = VectorIndex(vectors.shape[1], metric="ip")
            vector_index.add(vectors, [chunk.symbol_id for chunk in chunks])

            # 3) 落盘
            directory = self.index_dir(repository.id)
            await asyncio.to_thread(self._persist, directory, vector_index, bm25)

            built_at = datetime.now(UTC).isoformat()
            stats = IndexBuildStats(
                repository_id=repository.id,
                chunk_count=len(chunks),
                vector_count=len(vector_index),
                dimension=vector_index.dimension,
                embedder=embedder.name,
                model=embedder.model_id,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
            meta = stats.to_meta(
                head_commit=repository.head_commit,
                built_at=built_at,
                symbol_ids=[chunk.symbol_id for chunk in chunks],
            )
            # 记录测试符号，供检索时降权
            test_ids = [chunk.symbol_id for chunk in chunks if is_test_chunk(chunk)]
            meta["test_symbol_ids"] = test_ids
            await asyncio.to_thread(write_meta, directory / META_FILE, meta)

            with self._cache_lock:
                self._cache[repository.id] = _IndexBundle(
                    vectors=vector_index,
                    bm25=bm25,
                    meta=meta,
                    test_symbol_ids=frozenset(test_ids),
                )

            span.set_output(
                {"chunks": stats.chunk_count, "dim": stats.dimension, "embedder": stats.embedder}
            )
            span.set_metadata(duration_ms=round(stats.duration_ms, 1))
            return stats

    async def _load_symbols(
        self, session: AsyncSession, repository_id: int
    ) -> list[tuple[CodeSymbol, CodeFile]]:
        """取出仓库全部符号及其所属文件。"""
        stmt = (
            select(CodeSymbol, CodeFile)
            .join(CodeFile, CodeSymbol.file_id == CodeFile.id)
            .where(CodeSymbol.repository_id == repository_id)
            .order_by(CodeSymbol.id)
        )
        rows = (await session.execute(stmt)).all()
        return [(symbol, code_file) for symbol, code_file in rows]

    @staticmethod
    def _persist(directory: Path, vectors: VectorIndex, bm25: BM25Index) -> None:
        """把索引写入磁盘。"""
        directory.mkdir(parents=True, exist_ok=True)
        vectors.save(directory / VECTOR_FILE)
        bm25.save(directory / BM25_FILE)

    def _meta_is_current(self, meta: dict, repository: Repository, embedder) -> bool:
        """判断已有索引是否仍然有效。"""
        return (
            meta.get("embedder") == embedder.name
            and meta.get("model") == embedder.model_id
            and meta.get("dimension") == embedder.dimension
            and meta.get("head_commit") == repository.head_commit
        )

    # -- 加载 -------------------------------------------------------------
    async def _load_bundle(self, repository: Repository) -> _IndexBundle:
        """从内存缓存或磁盘加载索引。"""
        with self._cache_lock:
            cached = self._cache.get(repository.id)
        if cached is not None:
            return cached

        directory = self.index_dir(repository.id)
        meta = read_meta(directory / META_FILE)
        if meta is None:
            raise IndexMissingError(
                f"仓库 {repository.id} 尚未建立检索索引，"
                f"请先调用 POST /api/v1/repositories/{repository.id}/index"
            )

        dimension = int(meta.get("dimension", self._retrieval.embedding_dim))
        vectors = VectorIndex.load(directory / VECTOR_FILE, dimension, metric="ip")
        bm25 = BM25Index.load(directory / BM25_FILE)
        if vectors is None or bm25 is None or len(vectors) == 0:
            raise IndexMissingError(
                f"仓库 {repository.id} 的索引文件缺失或损坏，请重建索引"
            )

        bundle = _IndexBundle(
            vectors=vectors,
            bm25=bm25,
            meta=meta,
            test_symbol_ids=frozenset(int(item) for item in meta.get("test_symbol_ids", [])),
        )
        with self._cache_lock:
            self._cache[repository.id] = bundle
        return bundle

    def invalidate(self, repository_id: int) -> None:
        """使内存缓存失效（索引重建或仓库删除后调用）。"""
        with self._cache_lock:
            self._cache.pop(repository_id, None)

    # -- 检索 -------------------------------------------------------------
    async def search(
        self,
        session: AsyncSession,
        repository: Repository,
        query: str,
        *,
        top_k: int | None = None,
        language: str | None = None,
        include_tests: bool | None = None,
    ) -> SearchResult:
        """混合检索。

        Args:
            top_k: 返回条数，缺省用配置值。
            language: 按语言过滤（在融合后过滤，避免破坏召回）。
            include_tests: 是否包含测试代码。缺省按配置（默认降权测试符号）。
        """
        query = (query or "").strip()
        if not query:
            raise RetrievalError("查询内容不能为空")

        timings: dict[str, float] = {}
        limit = top_k or self._retrieval.top_k
        # 每路多召回一些再做融合，提升最终精度（单路 top_k 融合后可能全被裁掉）
        candidate_k = max(limit * self._retrieval.candidate_multiplier, limit)

        with trace_span(
            "retrieval.search",
            kind="rag",
            payload={"repository_id": repository.id, "query": query[:200], "top_k": limit},
        ) as span:
            bundle = await self._load_bundle(repository)

            # 1) BM25 关键词召回
            started = time.perf_counter()
            bm25_hits = await asyncio.to_thread(bundle.bm25.search, query, candidate_k)
            timings["bm25_ms"] = (time.perf_counter() - started) * 1000

            # 2) 向量语义召回
            started = time.perf_counter()
            embedder = self.embedder
            query_vector = await asyncio.to_thread(embedder.embed_query, query)
            vector_hits = await asyncio.to_thread(bundle.vectors.search, query_vector, candidate_k)
            timings["vector_ms"] = (time.perf_counter() - started) * 1000

            ranked_lists = [
                RankedList(
                    source="bm25",
                    ids=[symbol_id for symbol_id, _ in bm25_hits],
                    scores={symbol_id: score for symbol_id, score in bm25_hits},
                    weight=self._retrieval.bm25_weight,
                ),
                RankedList(
                    source="vector",
                    ids=[hit.symbol_id for hit in vector_hits],
                    scores={hit.symbol_id: hit.score for hit in vector_hits},
                    weight=self._retrieval.vector_weight,
                ),
            ]

            # 3) 融合
            # 测试符号降权必须在**融合之前**按候选列表切分，而不是融合后重排：
            # 真实仓库里测试用例常有大量近乎重复的文本（test_x_case_1..N），
            # 会把实现函数整个挤出候选窗口，此时任何"后置重排"都无从下手。
            # 做法是先用非测试候选融合，结果不足再用测试候选兜底补齐。
            started = time.perf_counter()
            prefer_non_test = (
                self._retrieval.prefer_non_test if include_tests is None else not include_tests
            )
            test_ids = bundle.test_symbol_ids if prefer_non_test else frozenset()

            def _split(lists: list[RankedList], *, tests: bool) -> list[RankedList]:
                """按是否测试符号拆分候选列表。"""
                return [
                    RankedList(
                        source=item.source,
                        ids=[i for i in item.ids if (i in test_ids) is tests],
                        scores={
                            i: s for i, s in item.scores.items() if (i in test_ids) is tests
                        },
                        weight=item.weight,
                    )
                    for item in lists
                ]

            fused = reciprocal_rank_fusion(
                _split(ranked_lists, tests=False), k=self._retrieval.rrf_k
            )
            if test_ids and len(fused) < limit:
                # 非测试候选不足，用测试候选补齐（降权而非硬过滤）
                already = {item.symbol_id for item in fused}
                fused.extend(
                    item
                    for item in reciprocal_rank_fusion(
                        _split(ranked_lists, tests=True), k=self._retrieval.rrf_k
                    )
                    if item.symbol_id not in already
                )
            timings["fuse_ms"] = (time.perf_counter() - started) * 1000

            # 4) 取详情
            started = time.perf_counter()
            hits = await self._hydrate(session, repository, fused, limit, language)
            timings["hydrate_ms"] = (time.perf_counter() - started) * 1000

            stale = bundle.meta.get("head_commit") != repository.head_commit
            notes: list[str] = []
            if stale:
                notes.append("索引基于旧版本代码（head_commit 已变化），建议重建索引")

            span.set_metadata(
                bm25_candidates=len(bm25_hits),
                vector_candidates=len(vector_hits),
                returned=len(hits),
                stale=stale,
            )
            span.set_output(
                {
                    "hits": [
                        {
                            "id": hit.symbol_id,
                            "name": hit.qualified_name,
                            "score": round(hit.score, 6),
                        }
                        for hit in hits[:5]
                    ]
                }
            )

            return SearchResult(
                query=query,
                repository_id=repository.id,
                hits=hits,
                timings_ms={key: round(value, 3) for key, value in timings.items()},
                stale=stale,
                notes=notes,
            )

    async def _hydrate(
        self,
        session: AsyncSession,
        repository: Repository,
        fused: list,
        limit: int,
        language: str | None,
    ) -> list[SearchHit]:
        """把融合结果补全为带源码片段的 SearchHit。"""
        if not fused:
            return []

        ordered_ids = [item.symbol_id for item in fused]
        stmt = (
            select(CodeSymbol, CodeFile)
            .join(CodeFile, CodeSymbol.file_id == CodeFile.id)
            .where(CodeSymbol.id.in_(ordered_ids))
        )
        rows = (await session.execute(stmt)).all()
        lookup = {symbol.id: (symbol, code_file) for symbol, code_file in rows}

        # 按文件分组读源码，避免同一文件重复读
        repo_root = Path(repository.local_path)
        line_cache: dict[str, list[str]] = {}

        hits: list[SearchHit] = []
        for item in fused:
            if len(hits) >= limit:
                break
            entry = lookup.get(item.symbol_id)
            if entry is None:
                # 符号已被重新索引删除，跳过
                continue
            symbol, code_file = entry
            if language and code_file.language != language:
                continue

            if code_file.path not in line_cache:
                try:
                    text = (repo_root / code_file.path).read_text(
                        encoding="utf-8", errors="replace"
                    )
                    line_cache[code_file.path] = text.splitlines()
                except OSError:
                    line_cache[code_file.path] = []

            hits.append(
                SearchHit(
                    symbol_id=symbol.id,
                    score=item.score,
                    path=code_file.path,
                    language=code_file.language,
                    qualified_name=symbol.qualified_name,
                    kind=symbol.kind,
                    signature=symbol.signature,
                    docstring=symbol.docstring,
                    start_line=symbol.start_line,
                    end_line=symbol.end_line,
                    code=slice_symbol(
                        line_cache[code_file.path], symbol.start_line, symbol.end_line
                    ),
                    bm25_score=item.raw_scores.get("bm25"),
                    vector_score=item.raw_scores.get("vector"),
                    bm25_rank=item.ranks.get("bm25"),
                    vector_rank=item.ranks.get("vector"),
                )
            )
        return hits


__all__ = [
    "BM25_FILE",
    "META_FILE",
    "VECTOR_FILE",
    "IndexEmptyError",
    "IndexMissingError",
    "RetrievalError",
    "RetrievalService",
    "weighted_score_fusion",
]
