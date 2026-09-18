"""混合检索服务包：BM25 关键词召回 + FAISS 向量召回 + RRF 融合。

分层：
- `tokenizer`     代码感知分词（驼峰/下划线拆解 + 中文 jieba）
- `chunker`       CodeSymbol -> CodeChunk（含源码切片）
- `embedder`      EmbeddingProvider 协议 + FastEmbed / Hashing 双实现
- `keyword_index` BM25 索引 + JSON 持久化
- `vector_index`  FAISS IndexIDMap2 索引 + 序列化
- `hybrid`        RRF 融合
- `service`       编排：建索引与检索
"""

from app.services.retrieval.chunker import (
    build_chunks,
    is_test_chunk,
    is_test_path,
    is_test_symbol,
    slice_symbol,
)
from app.services.retrieval.dto import (
    CodeChunk,
    IndexBuildStats,
    SearchHit,
    SearchResult,
)
from app.services.retrieval.embedder import (
    EmbeddingProvider,
    FastEmbedProvider,
    HashingEmbedder,
    build_embedder,
    get_shared_embedder,
    reset_shared_embedders,
)
from app.services.retrieval.hybrid import (
    FusedHit,
    RankedList,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)
from app.services.retrieval.keyword_index import BM25Index
from app.services.retrieval.service import (
    IndexEmptyError,
    IndexMissingError,
    RetrievalError,
    RetrievalService,
)
from app.services.retrieval.tokenizer import tokenize_code, tokenize_query
from app.services.retrieval.vector_index import VectorHit, VectorIndex

__all__ = [
    "BM25Index",
    "CodeChunk",
    "EmbeddingProvider",
    "FastEmbedProvider",
    "FusedHit",
    "HashingEmbedder",
    "IndexBuildStats",
    "IndexEmptyError",
    "IndexMissingError",
    "RankedList",
    "RetrievalError",
    "RetrievalService",
    "SearchHit",
    "SearchResult",
    "VectorHit",
    "VectorIndex",
    "build_chunks",
    "build_embedder",
    "get_shared_embedder",
    "is_test_chunk",
    "is_test_path",
    "is_test_symbol",
    "reciprocal_rank_fusion",
    "reset_shared_embedders",
    "slice_symbol",
    "tokenize_code",
    "tokenize_query",
    "weighted_score_fusion",
]
