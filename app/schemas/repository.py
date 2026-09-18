"""仓库解析模块的 API 契约（请求/响应模型）。"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RepoStatusLiteral = Literal["pending", "cloning", "parsing", "ready", "failed"]
SymbolKindLiteral = Literal["function", "method", "class", "struct", "interface"]


class RepositoryCreateRequest(BaseModel):
    """注册并索引一个远程仓库。"""

    model_config = ConfigDict(
        json_schema_extra={"example": {"url": "https://github.com/psf/requests.git"}}
    )

    url: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="仓库地址，支持 https://host/owner/repo(.git) 或 git@host:owner/repo.git",
    )
    branch: str | None = Field(
        default=None, max_length=255, description="指定分支；缺省用仓库默认分支"
    )
    force: bool = Field(default=False, description="为 true 时忽略已有索引，强制重新克隆并索引")


class RepositoryRead(BaseModel):
    """仓库详情。"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    url: str
    host: str
    owner: str
    name: str
    local_path: str
    default_branch: str | None = None
    head_commit: str | None = None
    status: RepoStatusLiteral
    error_message: str | None = None
    file_count: int
    symbol_count: int
    size_bytes: int
    cloned_at: datetime | None = None
    indexed_at: datetime | None = None
    created_at: datetime


class IndexStatsRead(BaseModel):
    """索引统计。"""

    file_count: int
    symbol_count: int
    failed_files: int
    duration_ms: float
    skipped: bool = False
    notes: list[str] = Field(default_factory=list)


class RepositoryIndexResponse(BaseModel):
    """注册/索引接口的响应。"""

    repository: RepositoryRead
    stats: IndexStatsRead
    trace_id: str | None = None


class RepositoryListResponse(BaseModel):
    """仓库分页列表。"""

    total: int
    items: list[RepositoryRead]


class CodeSymbolRead(BaseModel):
    """代码符号。"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    qualified_name: str
    kind: SymbolKindLiteral
    start_line: int
    end_line: int
    signature: str
    docstring: str | None = None
    is_async: bool
    complexity: int


class CodeFileRead(BaseModel):
    """文件及其符号。"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    path: str
    language: str
    size_bytes: int
    total_lines: int
    symbol_count: int
    parse_error: str | None = None
    symbols: list[CodeSymbolRead] = Field(default_factory=list)


class RepositoryStructureResponse(BaseModel):
    """仓库代码结构（按文件聚合）。"""

    repository_id: int
    total_files: int
    total_symbols: int
    languages: dict[str, int]
    files: list[CodeFileRead]
