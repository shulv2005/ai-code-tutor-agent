"""FAISS 向量索引：内积检索 + 磁盘持久化。

用 `IndexIDMap2(IndexFlatIP(dim))` 而不是裸 `IndexFlatIP`：
裸索引只能返回位置下标，还得额外维护「下标 -> symbol_id」映射表，
而 IDMap2 直接把 CodeSymbol.id 存进去，检索结果自带业务主键。

向量已做 L2 归一化，因此内积 == 余弦相似度（见 embedder._l2_normalize）。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class VectorHit:
    """一条向量检索命中。"""

    symbol_id: int
    score: float


class VectorIndex:
    """FAISS 向量索引封装。"""

    def __init__(self, dimension: int, *, metric: str = "ip") -> None:
        self._dimension = int(dimension)
        self._index = self._create(self._dimension, metric)
        self._metric = metric

    @staticmethod
    def _create(dimension: int, metric: str) -> faiss.Index:
        if metric == "l2":
            return faiss.IndexIDMap2(faiss.IndexFlatL2(dimension))
        return faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def metric(self) -> str:
        return self._metric

    def __len__(self) -> int:
        # 注意：IndexIDMap2 不支持 len()（faiss 1.15 实测抛 TypeError），必须读 ntotal
        return int(self._index.ntotal)

    def add(self, vectors: np.ndarray, symbol_ids: Sequence[int]) -> None:
        """批量写入向量。"""
        matrix = np.asarray(vectors, dtype="float32")
        if matrix.ndim != 2 or matrix.shape[1] != self._dimension:
            raise ValueError(
                f"向量维度不匹配：期望 (n, {self._dimension})，收到 {matrix.shape}"
            )
        if len(symbol_ids) != matrix.shape[0]:
            raise ValueError("symbol_ids 与向量数量不一致")
        ids = np.asarray(symbol_ids, dtype="int64")
        self._index.add_with_ids(matrix, ids)

    def search(self, vector: np.ndarray, top_k: int) -> list[VectorHit]:
        """返回 [(symbol_id, score)]，按分数降序。"""
        total = int(self._index.ntotal)
        if total == 0:
            return []
        query = np.asarray(vector, dtype="float32").reshape(1, -1)
        if query.shape[1] != self._dimension:
            raise ValueError(
                f"查询向量维度不匹配：期望 {self._dimension}，收到 {query.shape[1]}"
            )
        k = min(max(top_k, 1), total)
        scores, ids = self._index.search(query, k)

        hits: list[VectorHit] = []
        for score, symbol_id in zip(scores[0], ids[0], strict=True):
            # FAISS 用 -1 填充不足 k 的位置
            if symbol_id < 0:
                continue
            hits.append(VectorHit(symbol_id=int(symbol_id), score=float(score)))
        return hits

    # -- 持久化 -----------------------------------------------------------
    def save(self, path: Path) -> None:
        """序列化索引到磁盘。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(faiss.serialize_index(self._index))

    @classmethod
    def load(cls, path: Path, dimension: int, *, metric: str = "ip") -> VectorIndex | None:
        """从磁盘加载；文件缺失或损坏时返回 None。"""
        if not path.exists():
            return None
        try:
            raw = np.frombuffer(path.read_bytes(), dtype="uint8")
            index = cls(dimension, metric=metric)
            index._index = faiss.deserialize_index(raw)
            return index
        except Exception:  # noqa: BLE001 - faiss 反序列化异常类型不稳定
            logger.warning("FAISS 索引损坏，将重建: %s", path, exc_info=True)
            return None

    def to_dict(self) -> dict[str, Any]:
        """调试用的索引摘要。"""
        return {
            "dimension": self._dimension,
            "metric": self._metric,
            "count": len(self),
        }


def write_meta(path: Path, meta: dict[str, Any]) -> None:
    """写入索引元信息。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def read_meta(path: Path) -> dict[str, Any] | None:
    """读取索引元信息；缺失或损坏返回 None。"""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except (json.JSONDecodeError, OSError):
        logger.warning("索引元信息损坏: %s", path, exc_info=True)
        return None


__all__ = ["VectorHit", "VectorIndex", "read_meta", "write_meta"]
