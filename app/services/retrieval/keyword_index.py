"""BM25 关键词索引：内存索引 + JSON 持久化。

设计取舍：
- rank_bm25 的 BM25Okapi 需要把整个分词语料放进内存，这对「单仓库」粒度
  （数千到数万个符号）完全可接受，比引入 Elasticsearch 轻得多。
- 持久化的是**分词后的语料**而不是原文：重建时无需回查数据库、无需重新分词，
  加载速度最快。代价是磁盘占用略大，可接受。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.services.retrieval.tokenizer import tokenize_code

logger = logging.getLogger(__name__)


class BM25Index:
    """围绕 rank_bm25 的薄封装：维护 token 语料与 symbol_id 的顺序对应关系。"""

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        # 第 i 个语料对应第 i 个 symbol_id（顺序必须严格对齐）
        self._symbol_ids: list[int] = []
        self._corpus_tokens: list[list[str]] = []

    def __len__(self) -> int:
        return len(self._symbol_ids)

    @property
    def symbol_ids(self) -> list[int]:
        return list(self._symbol_ids)

    def build(self, symbol_ids: Sequence[int], documents: Sequence[str]) -> None:
        """用文档文本构建索引。"""
        if len(symbol_ids) != len(documents):
            raise ValueError("symbol_ids 与 documents 长度不一致")
        self._symbol_ids = list(symbol_ids)
        self._corpus_tokens = [tokenize_code(text) for text in documents]
        # 全空语料会让 BM25Okapi 除零，这里显式兜底
        if not self._corpus_tokens or all(not tokens for tokens in self._corpus_tokens):
            self._bm25 = None
            logger.debug("BM25 语料为空，索引置空")
            return
        self._bm25 = BM25Okapi(self._corpus_tokens)

    def build_from_tokens(self, symbol_ids: Sequence[int], tokens: Sequence[Sequence[str]]) -> None:
        """用已分词的语料直接构建（加载持久化索引时走这条路径）。"""
        if len(symbol_ids) != len(tokens):
            raise ValueError("symbol_ids 与 tokens 长度不一致")
        self._symbol_ids = list(symbol_ids)
        self._corpus_tokens = [list(item) for item in tokens]
        self._bm25 = (
            None
            if not self._corpus_tokens or all(not item for item in self._corpus_tokens)
            else BM25Okapi(self._corpus_tokens)
        )

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """返回 [(symbol_id, score)]，按分数降序，仅保留正分。"""
        if self._bm25 is None or not self._symbol_ids:
            return []
        tokens = tokenize_code(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        ranked = [
            (self._symbol_ids[index], float(score))
            for index, score in enumerate(scores)
            # 过滤 0 分：BM25 无匹配时返回 0，带进融合只会稀释精度
            if score > 0.0
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:top_k]

    # -- 持久化 -----------------------------------------------------------
    def save(self, path: Path) -> None:
        """把分词语料与 id 顺序写入 JSON。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"symbol_ids": self._symbol_ids, "tokens": self._corpus_tokens}
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> BM25Index | None:
        """从 JSON 加载；文件缺失或损坏时返回 None（调用方重建索引）。"""
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            index = cls()
            index.build_from_tokens(payload["symbol_ids"], payload["tokens"])
            return index
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            logger.warning("BM25 索引损坏，将重建: %s", path, exc_info=True)
            return None


__all__ = ["BM25Index"]
