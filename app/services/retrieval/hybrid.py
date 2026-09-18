"""混合检索融合：Reciprocal Rank Fusion (RRF)。

为什么默认用 RRF 而不是「加权分数相加」：
BM25 分数是无上界的词频统计量，余弦相似度落在 [-1, 1]，两者量纲不可比。
直接加权求和必须做分数归一化，而归一化方式（min-max / z-score）会引入
额外超参且对异常分数敏感。RRF 只用**排名**，天然免疫量纲问题，
是混合检索里更稳健的默认选择。

    score(d) = Σ_r  weight_r / (k + rank_r(d))

k 默认 60（原论文取值），作用是拉平各排名的差距，避免单一来源霸榜。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

FusionMethod = Literal["rrf", "weighted"]


@dataclass(slots=True)
class RankedList:
    """某一路召回的排名结果。"""

    source: str
    # 按相关性降序排列的 symbol_id
    ids: list[int]
    # 原始分数明细，用于结果解释（不参与 RRF 计算）
    scores: dict[int, float] = field(default_factory=dict)
    weight: float = 1.0


@dataclass(slots=True)
class FusedHit:
    """融合后的一条结果。"""

    symbol_id: int
    score: float
    # 各来源的排名与分数：1-based 排名，None 表示该路未召回
    ranks: dict[str, int] = field(default_factory=dict)
    raw_scores: dict[str, float] = field(default_factory=dict)


def reciprocal_rank_fusion(
    ranked_lists: list[RankedList],
    *,
    k: int = 60,
    limit: int | None = None,
) -> list[FusedHit]:
    """RRF 融合多路召回结果。

    Args:
        ranked_lists: 各路召回结果。
        k: RRF 平滑常数。
        limit: 截断返回条数；None 表示返回全部。
    """
    scores: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}
    raw_scores: dict[int, dict[str, float]] = {}

    for ranked in ranked_lists:
        for position, symbol_id in enumerate(ranked.ids, start=1):
            contribution = ranked.weight / (k + position)
            scores[symbol_id] = scores.get(symbol_id, 0.0) + contribution
            ranks.setdefault(symbol_id, {})[ranked.source] = position
            if symbol_id in ranked.scores:
                raw_scores.setdefault(symbol_id, {})[ranked.source] = ranked.scores[symbol_id]

    fused = [
        FusedHit(
            symbol_id=symbol_id,
            score=score,
            ranks=ranks.get(symbol_id, {}),
            raw_scores=raw_scores.get(symbol_id, {}),
        )
        for symbol_id, score in scores.items()
    ]
    # 分数相同时按 symbol_id 稳定排序，保证结果可复现
    fused.sort(key=lambda item: (-item.score, item.symbol_id))
    return fused[:limit] if limit is not None else fused


def weighted_score_fusion(
    ranked_lists: list[RankedList],
    *,
    limit: int | None = None,
) -> list[FusedHit]:
    """加权分数融合（备选方案）：各路先 min-max 归一化再按权重求和。

    保留它是为了在「分数可比」的场景（例如两路都是余弦相似度）下获得更好效果。
    默认不使用，因为 BM25 与余弦相似度量纲不可比。
    """
    normalized: list[tuple[RankedList, dict[int, float]]] = []
    for ranked in ranked_lists:
        if not ranked.scores:
            continue
        values = list(ranked.scores.values())
        low, high = min(values), max(values)
        span = high - low
        scaled = {
            symbol_id: (1.0 if span <= 0 else (score - low) / span)
            for symbol_id, score in ranked.scores.items()
        }
        normalized.append((ranked, scaled))

    scores: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}
    raw_scores: dict[int, dict[str, float]] = {}

    for ranked, scaled in normalized:
        for position, symbol_id in enumerate(ranked.ids, start=1):
            scores[symbol_id] = scores.get(symbol_id, 0.0) + ranked.weight * scaled.get(
                symbol_id, 0.0
            )
            ranks.setdefault(symbol_id, {})[ranked.source] = position
            raw_scores.setdefault(symbol_id, {})[ranked.source] = ranked.scores[symbol_id]

    fused = [
        FusedHit(
            symbol_id=symbol_id,
            score=score,
            ranks=ranks.get(symbol_id, {}),
            raw_scores=raw_scores.get(symbol_id, {}),
        )
        for symbol_id, score in scores.items()
    ]
    fused.sort(key=lambda item: (-item.score, item.symbol_id))
    return fused[:limit] if limit is not None else fused


__all__ = [
    "FusedHit",
    "FusionMethod",
    "RankedList",
    "reciprocal_rank_fusion",
    "weighted_score_fusion",
]
