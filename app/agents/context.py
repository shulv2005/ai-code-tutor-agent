"""代码上下文构建：把不同来源（直接代码 / 符号 ID / 自然语言检索）归一成 CodeContext。

这是 Step 3（检索）与 Step 4（测试生成）之间的适配层，
让 Agent 不必关心上下文是"用户贴的代码"还是"检索出来的符号"。
"""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.dto import CodeContext
from app.models.code import CodeFile, CodeSymbol
from app.models.repository import Repository
from app.services.retrieval.chunker import slice_symbol
from app.services.retrieval.service import RetrievalService

logger = logging.getLogger(__name__)

# 相关符号与现有测试的附加上限，避免提示词过长推高成本
MAX_RELATED_SYMBOLS = 8
MAX_EXISTING_TESTS = 3
MAX_TEST_SNIPPET_CHARS = 1200


class ContextError(RuntimeError):
    """无法构建代码上下文。"""


def compute_import_hint(path: str, qualified_name: str) -> str | None:
    """由文件路径与符号名推导导入语句。

    规则：
    - `pkg/core.py` + `add`          -> `from pkg.core import add`
    - `src/requests/api.py` + `get`  -> `from requests.api import get`（剥掉 src 布局前缀）
    - `pkg/__init__.py` + `helper`   -> `from pkg import helper`
    - `Service.run`（方法）           -> `from pkg.core import Service`

    这只是给模型的提示而非硬约束：真实仓库的导入路径可能受包安装方式影响，
    LLM 可以自行调整。但没有它，模型会倾向于内联一份实现副本，
    导致后续补丁打不到测试实际调用的代码上。
    """
    normalized = path.replace("\\", "/").strip("/")
    if not normalized.endswith(".py"):
        return None

    module_path = normalized[: -len(".py")]
    parts = [part for part in module_path.split("/") if part and part != "."]
    # 常见 src 布局：src/<pkg>/... -> <pkg>...
    if parts and parts[0] == "src":
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts:
        return None

    module = ".".join(parts)
    # 方法取所属类名（Service.run -> Service）
    name = qualified_name.split(".")[0] if qualified_name else ""
    if not name or not name.isidentifier():
        return f"from {module} import *  # 请按实际符号名调整"
    return f"from {module} import {name}"


def _read_repo_file(repo_root: Path, relative_path: str) -> list[str]:
    """读取仓库内文件的所有行；失败返回空列表。"""
    try:
        text = (repo_root / relative_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        logger.warning("读取源码失败: %s", relative_path, exc_info=True)
        return []
    return text.splitlines()


async def context_from_symbol(
    session: AsyncSession,
    repository: Repository,
    symbol_id: int,
) -> CodeContext:
    """按符号 ID 从索引构建上下文。"""
    stmt = (
        select(CodeSymbol, CodeFile)
        .join(CodeFile, CodeSymbol.file_id == CodeFile.id)
        .where(CodeSymbol.id == symbol_id)
    )
    row = (await session.execute(stmt)).first()
    if row is None:
        raise ContextError(f"符号不存在: {symbol_id}")
    symbol, code_file = row

    lines = _read_repo_file(Path(repository.local_path), code_file.path)
    return CodeContext(
        path=code_file.path,
        code=slice_symbol(lines, symbol.start_line, symbol.end_line),
        language=code_file.language,
        qualified_name=symbol.qualified_name,
        kind=symbol.kind,
        signature=symbol.signature,
        docstring=symbol.docstring,
        start_line=symbol.start_line,
        end_line=symbol.end_line,
        repository=f"{repository.owner}/{repository.name}",
        symbol_id=symbol.id,
        import_hint=compute_import_hint(code_file.path, symbol.qualified_name),
    )


async def context_from_query(
    session: AsyncSession,
    repository: Repository,
    retrieval: RetrievalService,
    query: str,
) -> CodeContext:
    """用混合检索定位待测代码，取排名第一的命中构建上下文。"""
    result = await retrieval.search(session, repository, query, top_k=1)
    if not result.hits:
        raise ContextError(f"检索未命中任何代码符号，无法生成测试：{query}")

    hit = result.hits[0]
    return CodeContext(
        path=hit.path,
        code=hit.code,
        language=hit.language,
        qualified_name=hit.qualified_name,
        kind=hit.kind,
        signature=hit.signature,
        docstring=hit.docstring,
        start_line=hit.start_line,
        end_line=hit.end_line,
        repository=f"{repository.owner}/{repository.name}",
        symbol_id=hit.symbol_id,
        import_hint=compute_import_hint(hit.path, hit.qualified_name),
    )


async def enrich_context(
    session: AsyncSession,
    repository: Repository | None,
    retrieval: RetrievalService | None,
    context: CodeContext,
    *,
    include_related: bool = True,
    include_existing_tests: bool = True,
) -> CodeContext:
    """补充相关符号与现有测试，提升生成质量。"""
    if repository is None:
        return context

    # 1) 同文件其它符号：帮助模型理解模块内的调用约定与命名风格
    if include_related and context.symbol_id is not None:
        stmt = (
            select(CodeSymbol)
            .join(CodeFile, CodeSymbol.file_id == CodeFile.id)
            .where(
                CodeFile.repository_id == repository.id,
                CodeFile.path == context.path,
                CodeSymbol.id != context.symbol_id,
            )
            .order_by(CodeSymbol.start_line)
            .limit(MAX_RELATED_SYMBOLS)
        )
        siblings = (await session.execute(stmt)).scalars().all()
        context.related = [
            f"{item.kind} {item.signature or item.qualified_name}"
            + (f"  # {item.docstring.splitlines()[0]}" if item.docstring else "")
            for item in siblings
        ]

    # 2) 现有测试：作为风格参考（Step 3 检索的 include_tests=True）
    if include_existing_tests and retrieval is not None:
        query = context.qualified_name or Path(context.path).stem
        try:
            result = await retrieval.search(
                session,
                repository,
                query,
                top_k=MAX_EXISTING_TESTS,
                include_tests=True,
            )
        except Exception:  # noqa: BLE001 - 索引缺失等不应阻断生成
            logger.debug("检索现有测试失败，跳过风格参考", exc_info=True)
        else:
            from app.services.retrieval.chunker import is_test_chunk
            from app.services.retrieval.dto import CodeChunk

            snippets: list[str] = []
            for hit in result.hits:
                probe = CodeChunk(
                    symbol_id=hit.symbol_id,
                    repository_id=repository.id,
                    path=hit.path,
                    language=hit.language,
                    qualified_name=hit.qualified_name,
                    kind=hit.kind,
                    signature=hit.signature,
                    docstring=hit.docstring,
                    start_line=hit.start_line,
                    end_line=hit.end_line,
                    code=hit.code,
                )
                if not is_test_chunk(probe):
                    continue
                snippet = hit.code[:MAX_TEST_SNIPPET_CHARS]
                snippets.append(f"# {hit.path}::{hit.qualified_name}\n{snippet}")
            context.existing_tests = snippets[:MAX_EXISTING_TESTS]

    return context


def context_from_snippet(
    code: str,
    *,
    file_path: str = "snippet.py",
    symbol_name: str | None = None,
    language: str = "python",
) -> CodeContext:
    """用用户直接给出的代码片段构建上下文（不依赖仓库与索引）。"""
    if not code.strip():
        raise ContextError("代码片段为空")

    return CodeContext(
        path=file_path,
        code=code,
        language=language,
        qualified_name=symbol_name or Path(file_path).stem,
        kind="snippet",
        signature="",
        docstring=None,
        start_line=1,
        end_line=code.count("\n") + 1,
    )


__all__ = [
    "ContextError",
    "compute_import_hint",
    "context_from_query",
    "context_from_snippet",
    "context_from_symbol",
    "enrich_context",
]
