"""仓库解析服务包：Git 克隆、多语言代码解析、索引落库。

分层：
- `git_service`  GitPython 封装（克隆/更新 + URL 安全校验）
- `language`     语言识别与 tree-sitter 语法 profile
- `python_parser` / `tree_sitter_parser`  两种解析引擎
- `parser`       统一解析门面 + 文件遍历
- `service`      编排：克隆 -> 解析 -> 落库
"""

from app.services.repo.dto import ParsedFile, ParsedImport, ParsedSymbol, SymbolKind
from app.services.repo.git_service import (
    CloneResult,
    GitService,
    InvalidRepoUrlError,
    RepoCloneError,
    RepoRef,
    RepositoryError,
    RepoTooLargeError,
    parse_repo_url,
    remove_tree,
)
from app.services.repo.language import detect_language, supported_languages
from app.services.repo.parser import (
    iter_source_files,
    parse_file,
    parse_source,
    to_posix_relative,
)
from app.services.repo.service import IndexStats, RepositoryService

__all__ = [
    "CloneResult",
    "GitService",
    "IndexStats",
    "InvalidRepoUrlError",
    "ParsedFile",
    "ParsedImport",
    "ParsedSymbol",
    "RepoCloneError",
    "RepoRef",
    "RepoTooLargeError",
    "RepositoryError",
    "RepositoryService",
    "SymbolKind",
    "detect_language",
    "iter_source_files",
    "parse_file",
    "parse_repo_url",
    "parse_source",
    "remove_tree",
    "supported_languages",
    "to_posix_relative",
]
