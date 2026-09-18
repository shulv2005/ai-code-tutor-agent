"""检索基础组件测试：BM25、FAISS、RRF 融合、嵌入后端。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.services.retrieval.embedder import HashingEmbedder
from app.services.retrieval.hybrid import (
    RankedList,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)
from app.services.retrieval.keyword_index import BM25Index
from app.services.retrieval.vector_index import VectorIndex, read_meta, write_meta

DOCS = [
    (101, "add(a: int, b: int) -> int 两个数相加并返回结果"),
    (102, "fetch(url: str, retries: int = 3) -> str 异步拉取远端数据并在失败时重试"),
    (103, "class UserRepository 负责用户表的增删改查"),
    (104, "parse_config(path) 读取 YAML 配置文件并校验必填字段"),
]


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------
def test_bm25_ranks_keyword_match_first() -> None:
    index = BM25Index()
    index.build([doc_id for doc_id, _ in DOCS], [text for _, text in DOCS])

    hits = index.search("UserRepository 用户表", 3)
    assert hits
    assert hits[0][0] == 103
    assert hits[0][1] > 0


def test_bm25_handles_snake_and_camel_query() -> None:
    """查询写法与代码写法不同也要能命中。"""
    index = BM25Index()
    index.build([doc_id for doc_id, _ in DOCS], [text for _, text in DOCS])

    assert index.search("parseConfig", 1)[0][0] == 104
    assert index.search("parse_config", 1)[0][0] == 104


def test_bm25_returns_empty_when_no_match() -> None:
    index = BM25Index()
    index.build([doc_id for doc_id, _ in DOCS], [text for _, text in DOCS])
    # 选词必须在语料中完全不出现（'yaml' 出现在 DOCS[3] 里，不能用作反例）
    assert index.search("kubernetes deployment helm", 5) == []


def test_bm25_empty_query_and_empty_corpus() -> None:
    index = BM25Index()
    index.build([101], ["add(a, b) 相加"])
    assert index.search("", 5) == []
    assert index.search("   ", 5) == []

    empty = BM25Index()
    empty.build([], [])
    assert empty.search("anything", 5) == []
    assert len(empty) == 0


def test_bm25_length_mismatch_is_rejected() -> None:
    index = BM25Index()
    with pytest.raises(ValueError):
        index.build([1, 2], ["only one"])


def test_bm25_roundtrip_through_disk(tmp_path: Path) -> None:
    index = BM25Index()
    index.build([doc_id for doc_id, _ in DOCS], [text for _, text in DOCS])
    before = index.search("配置文件", 3)

    path = tmp_path / "bm25.json"
    index.save(path)
    restored = BM25Index.load(path)

    assert restored is not None
    assert len(restored) == len(DOCS)
    assert restored.search("配置文件", 3) == before


def test_bm25_load_handles_missing_and_corrupt(tmp_path: Path) -> None:
    assert BM25Index.load(tmp_path / "nope.json") is None

    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert BM25Index.load(bad) is None


# ---------------------------------------------------------------------------
# FAISS
# ---------------------------------------------------------------------------
def test_vector_index_search_returns_symbol_ids() -> None:
    """IndexIDMap2 必须直接返回业务主键（symbol_id），而不是位置下标。"""
    vectors = np.eye(4, dtype="float32")
    index = VectorIndex(4)
    index.add(vectors, [11, 22, 33, 44])

    assert len(index) == 4
    hits = index.search(vectors[2], 2)
    assert hits[0].symbol_id == 33
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_vector_index_roundtrip_through_disk(tmp_path: Path) -> None:
    vectors = np.eye(4, dtype="float32")
    index = VectorIndex(4)
    index.add(vectors, [11, 22, 33, 44])

    path = tmp_path / "vectors.faiss"
    index.save(path)
    restored = VectorIndex.load(path, 4)

    assert restored is not None
    assert len(restored) == 4
    assert restored.search(vectors[1], 1)[0].symbol_id == 22


def test_vector_index_load_missing_and_corrupt(tmp_path: Path) -> None:
    assert VectorIndex.load(tmp_path / "nope.faiss", 4) is None

    bad = tmp_path / "bad.faiss"
    bad.write_bytes(b"not a faiss index")
    assert VectorIndex.load(bad, 4) is None


def test_vector_index_validates_dimension() -> None:
    index = VectorIndex(4)
    with pytest.raises(ValueError, match="维度不匹配"):
        index.add(np.zeros((2, 5), dtype="float32"), [1, 2])
    with pytest.raises(ValueError, match="数量不一致"):
        index.add(np.zeros((2, 4), dtype="float32"), [1])


def test_vector_index_empty_search() -> None:
    assert VectorIndex(4).search(np.zeros(4, dtype="float32"), 5) == []


def test_meta_roundtrip_and_corrupt(tmp_path: Path) -> None:
    path = tmp_path / "meta.json"
    write_meta(path, {"dimension": 512, "head_commit": "abc"})
    assert read_meta(path) == {"dimension": 512, "head_commit": "abc"}

    assert read_meta(tmp_path / "missing.json") is None
    broken = tmp_path / "broken.json"
    broken.write_text("nope", encoding="utf-8")
    assert read_meta(broken) is None


# ---------------------------------------------------------------------------
# RRF 融合
# ---------------------------------------------------------------------------
def test_rrf_rewards_items_found_by_both_sources() -> None:
    """两路都召回的条目应排在只被单路召回的条目之前。"""
    bm25 = RankedList("bm25", ids=[1, 2, 3], scores={1: 5.0, 2: 4.0, 3: 3.0}, weight=1.0)
    vector = RankedList("vector", ids=[3, 4, 1], scores={3: 0.9, 4: 0.8, 1: 0.7}, weight=1.0)

    fused = reciprocal_rank_fusion([bm25, vector])
    ids = [item.symbol_id for item in fused]

    # 1 和 3 都被两路召回，应占据前两位
    assert set(ids[:2]) == {1, 3}
    # 4 只被向量召回排在末尾
    assert ids[-1] == 4


def test_rrf_records_ranks_and_raw_scores() -> None:
    bm25 = RankedList("bm25", ids=[7], scores={7: 2.5}, weight=1.0)
    vector = RankedList("vector", ids=[7], scores={7: 0.8}, weight=1.0)

    top = reciprocal_rank_fusion([bm25, vector])[0]
    assert top.ranks == {"bm25": 1, "vector": 1}
    assert top.raw_scores == {"bm25": 2.5, "vector": 0.8}
    # 1/(60+1) * 2
    assert top.score == pytest.approx(2 / 61)


def test_rrf_respects_weights() -> None:
    bm25 = RankedList("bm25", ids=[1], weight=1.0)
    vector = RankedList("vector", ids=[2], weight=1.0)
    both_equal = reciprocal_rank_fusion([bm25, vector])
    assert both_equal[0].score == pytest.approx(both_equal[1].score)
    # 分数相同时按 symbol_id 升序，保证可复现
    assert [item.symbol_id for item in both_equal] == [1, 2]

    heavier = reciprocal_rank_fusion(
        [RankedList("bm25", ids=[1], weight=3.0), RankedList("vector", ids=[2], weight=1.0)]
    )
    assert heavier[0].symbol_id == 1


def test_rrf_limit_and_empty_input() -> None:
    lists = [RankedList("bm25", ids=[1, 2, 3])]
    assert len(reciprocal_rank_fusion(lists, limit=2)) == 2
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([RankedList("bm25", ids=[])]) == []


def test_weighted_fusion_normalizes_scores() -> None:
    """BM25 与余弦量纲不同，加权融合必须先归一化。"""
    bm25 = RankedList("bm25", ids=[1, 2], scores={1: 100.0, 2: 0.0}, weight=0.5)
    vector = RankedList("vector", ids=[2, 1], scores={2: 1.0, 1: 0.0}, weight=0.5)

    fused = weighted_score_fusion([bm25, vector])
    # 归一化后各拿一个满分，应打成平手
    assert fused[0].score == pytest.approx(fused[1].score)


# ---------------------------------------------------------------------------
# 嵌入后端
# ---------------------------------------------------------------------------
def test_hashing_embedder_is_deterministic() -> None:
    embedder = HashingEmbedder(dimension=128)
    first = embedder.embed_documents(["def add(a, b): return a + b"])
    second = embedder.embed_documents(["def add(a, b): return a + b"])
    np.testing.assert_array_equal(first, second)


def test_hashing_embedder_outputs_normalized_vectors() -> None:
    embedder = HashingEmbedder(dimension=64)
    vectors = embedder.embed_documents(["alpha beta gamma", "delta epsilon"])
    assert vectors.shape == (2, 64)
    norms = np.linalg.norm(vectors, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_hashing_embedder_similarity_tracks_overlap() -> None:
    """词重叠越多，余弦相似度应越高。"""
    embedder = HashingEmbedder(dimension=512)
    base = "def fetch_user_data url retries timeout"
    similar = "def fetch_user_data url retries"
    different = "class Widget render constructor styles"

    vectors = embedder.embed_documents([base, similar, different])
    sim_similar = float(vectors[0] @ vectors[1])
    sim_different = float(vectors[0] @ vectors[2])
    assert sim_similar > sim_different


def test_hashing_embedder_query_shape() -> None:
    embedder = HashingEmbedder(dimension=32)
    vector = embedder.embed_query("hello world")
    assert vector.shape == (32,)
    assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-5)


def test_hashing_embedder_handles_empty_input() -> None:
    embedder = HashingEmbedder(dimension=16)
    assert embedder.embed_documents([]).shape == (0, 16)
    # 空串产生零向量，且不能出现 NaN（归一化除零保护）
    vector = embedder.embed_query("")
    assert not np.isnan(vector).any()
