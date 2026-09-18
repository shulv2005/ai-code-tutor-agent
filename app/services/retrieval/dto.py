"""检索层 DTO：索引文档、检索命中与检索结果的稳定契约。

与 Step 2 的解析 DTO 一样，这里刻意不依赖 SQLAlchemy 与 numpy，
方便上层（Agent、测试）直接消费。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CodeChunk:
    """送入索引的最小检索单元：一个代码符号 + 其源码文本。"""

    # 直接复用 CodeSymbol.id 作为索引主键，省掉一层 id 映射
    symbol_id: int
    repository_id: int
    path: str
    language: str
    qualified_name: str
    kind: str
    signature: str
    docstring: str | None
    start_line: int
    end_line: int
    code: str

    def to_document(self, *, max_chars: int) -> str:
        """拼装用于嵌入的文档文本。

        顺序有意为之：标识符与签名在前、代码正文在后，并在超长时截断正文，
        保证关键检索信号不会因为截断而丢失。
        """
        header = f"{self.qualified_name}\n{self.signature}"
        if self.docstring:
            header = f"{header}\n{self.docstring}"
        body = self.code
        budget = max(max_chars - len(header), 0)
        if len(body) > budget:
            body = body[:budget]
        return f"{header}\n{body}"


@dataclass(slots=True)
class SearchHit:
    """一条检索命中。"""

    symbol_id: int
    score: float
    path: str
    language: str
    qualified_name: str
    kind: str
    signature: str
    docstring: str | None
    start_line: int
    end_line: int
    code: str
    # 分路明细：便于排查"为什么这条排前面"，也是 Agent 解释性的依据
    bm25_score: float | None = None
    vector_score: float | None = None
    bm25_rank: int | None = None
    vector_rank: int | None = None


@dataclass(slots=True)
class SearchResult:
    """一次检索的完整结果。"""

    query: str
    repository_id: int
    hits: list[SearchHit] = field(default_factory=list)
    # 各阶段耗时，供性能观察
    timings_ms: dict[str, float] = field(default_factory=dict)
    # 索引是否因代码更新而陈旧（陈旧时结果基于旧索引）
    stale: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class IndexBuildStats:
    """一次索引构建的统计。"""

    repository_id: int
    chunk_count: int
    vector_count: int
    dimension: int
    embedder: str
    model: str
    duration_ms: float = 0.0
    skipped: bool = False
    notes: list[str] = field(default_factory=list)

    def to_meta(
        self, *, head_commit: str | None, built_at: str, symbol_ids: list[int]
    ) -> dict[str, Any]:
        """生成落盘的元信息（用于陈旧检测与加载校验）。"""
        return {
            "repository_id": self.repository_id,
            "chunk_count": self.chunk_count,
            "dimension": self.dimension,
            "embedder": self.embedder,
            "model": self.model,
            "head_commit": head_commit,
            "built_at": built_at,
            "symbol_ids": symbol_ids,
        }
