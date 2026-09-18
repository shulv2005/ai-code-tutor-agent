"""检索模块的 API 契约。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class IndexBuildResponse(BaseModel):
    """索引构建结果。"""

    repository_id: int
    chunk_count: int
    vector_count: int
    dimension: int
    embedder: str = Field(description="实际使用的嵌入后端：fastembed（语义）或 hashing（兜底）")
    model: str
    duration_ms: float
    skipped: bool = Field(default=False, description="为 true 表示索引已最新，未重建")
    notes: list[str] = Field(default_factory=list)
    trace_id: str | None = None


class SearchRequest(BaseModel):
    """混合检索请求。"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "repository_id": 1,
                "query": "解析配置文件并校验必填字段的函数",
                "top_k": 10,
            }
        }
    )

    repository_id: int = Field(..., ge=1, description="目标仓库 ID")
    query: str = Field(..., min_length=1, max_length=2000, description="自然语言或代码查询")
    top_k: int | None = Field(default=None, ge=1, le=100, description="返回条数，缺省用配置值")
    language: str | None = Field(default=None, max_length=64, description="按语言过滤，如 python")
    include_tests: bool | None = Field(
        default=None,
        description=(
            "是否包含测试代码。缺省按配置（默认把测试符号排到实现代码之后）。"
            "Step 4 生成测试用例时可传 true 以检索现有测试作为参考。"
        ),
    )


class SearchHitRead(BaseModel):
    """单条检索命中。"""

    symbol_id: int
    score: float = Field(description="RRF 融合分（量纲无关，仅用于排序）")
    path: str
    language: str
    qualified_name: str
    kind: str
    signature: str
    docstring: str | None = None
    start_line: int
    end_line: int
    code: str = Field(description="源码片段，可直接作为 Step 4/6 Agent 的上下文")
    bm25_score: float | None = None
    vector_score: float | None = None
    bm25_rank: int | None = None
    vector_rank: int | None = None


class SearchResponse(BaseModel):
    """检索响应。"""

    query: str
    repository_id: int
    total: int
    hits: list[SearchHitRead] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    stale: bool = Field(default=False, description="索引是否落后于仓库最新提交")
    notes: list[str] = Field(default_factory=list)
    trace_id: str | None = None
