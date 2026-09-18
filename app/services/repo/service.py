"""仓库索引编排：克隆 -> 遍历 -> 解析 -> 落库。

并发模型：文件遍历与解析是 CPU/IO 密集的同步操作，整体丢进 `asyncio.to_thread`
执行，只在最后落库阶段回到事件循环，避免拖住 FastAPI。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import delete, insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.trace import trace_span
from app.models.code import CodeFile, CodeSymbol
from app.models.repository import Repository
from app.services.repo.dto import ParsedFile
from app.services.repo.git_service import GitService, RepoRef, parse_repo_url
from app.services.repo.parser import iter_source_files, parse_file

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IndexStats:
    """一次索引的统计结果。"""

    file_count: int = 0
    symbol_count: int = 0
    failed_files: int = 0
    duration_ms: float = 0.0
    skipped: bool = False
    notes: list[str] = field(default_factory=list)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _parse_repository_sync(root: Path, settings: Settings) -> tuple[list[ParsedFile], list[str]]:
    """同步遍历并解析整个仓库（在工作线程中执行）。"""
    repo_settings = settings.repository
    excluded = set(repo_settings.excluded_dirs)
    results: list[ParsedFile] = []
    notes: list[str] = []

    for path in iter_source_files(root, excluded_dirs=excluded, max_files=repo_settings.max_files):
        results.append(parse_file(path, root, max_file_bytes=repo_settings.max_file_bytes))

    if len(results) >= repo_settings.max_files:
        notes.append(f"文件数达到上限 {repo_settings.max_files}，索引被截断")

    return results, notes


class RepositoryService:
    """仓库生命周期管理：注册、克隆、索引。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._git = GitService(settings.repository)

    @property
    def git(self) -> GitService:
        return self._git

    async def get_by_url(self, session: AsyncSession, sanitized_url: str) -> Repository | None:
        """按规范化地址查询仓库。"""
        stmt = select(Repository).where(Repository.url == sanitized_url)
        return (await session.execute(stmt)).scalar_one_or_none()

    async def get_by_id(self, session: AsyncSession, repository_id: int) -> Repository | None:
        """按主键查询仓库。"""
        return await session.get(Repository, repository_id)

    async def list_repositories(
        self, session: AsyncSession, *, limit: int = 50, offset: int = 0
    ) -> list[Repository]:
        """分页列出仓库（按创建时间倒序）。"""
        stmt = (
            select(Repository)
            .order_by(Repository.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list((await session.execute(stmt)).scalars().all())

    async def register_and_index(
        self,
        session: AsyncSession,
        url: str,
        *,
        branch: str | None = None,
        force: bool = False,
    ) -> tuple[Repository, IndexStats]:
        """注册仓库并完成克隆 + 索引。

        Args:
            force: True 时即使已索引过也重新克隆并索引。
        """
        ref: RepoRef = parse_repo_url(url, self._settings.repository.allowed_hosts)

        with trace_span(
            "repository.register_and_index",
            kind="agent",
            payload={"url": ref.sanitized_url, "branch": branch, "force": force},
        ) as root_span:
            repository = await self.get_by_url(session, ref.sanitized_url)

            if repository is None:
                repository = Repository(
                    url=ref.sanitized_url,
                    host=ref.host,
                    owner=ref.owner,
                    name=ref.name,
                    local_path=str(self._git.target_path(ref)),
                    status="pending",
                )
                session.add(repository)
                await session.flush()
            elif repository.status == "ready" and not force:
                # 已索引过且不强制刷新，直接复用，省掉一次克隆
                root_span.set_metadata(cached=True)
                return repository, IndexStats(
                    file_count=repository.file_count,
                    symbol_count=repository.symbol_count,
                    skipped=True,
                    notes=["仓库已索引，如需刷新请传 force=true"],
                )

            repository.status = "cloning"
            repository.error_message = None
            await session.flush()

            try:
                clone = await self._git.clone_or_update(ref, branch)
            except Exception as exc:
                repository.status = "failed"
                repository.error_message = str(exc)
                await session.flush()
                root_span.set_metadata(failed_stage="clone")
                raise

            repository.local_path = str(clone.local_path)
            repository.head_commit = clone.head_commit
            repository.default_branch = clone.default_branch
            repository.size_bytes = clone.size_bytes
            repository.cloned_at = _utc_now()
            repository.status = "parsing"
            await session.flush()

            stats = await self._index_files(session, repository, clone.local_path)

            repository.status = "ready"
            repository.file_count = stats.file_count
            repository.symbol_count = stats.symbol_count
            repository.indexed_at = _utc_now()
            await session.commit()
            await session.refresh(repository)

            root_span.set_output(
                {
                    "repository_id": repository.id,
                    "files": stats.file_count,
                    "symbols": stats.symbol_count,
                    "failed_files": stats.failed_files,
                }
            )
            return repository, stats

    async def reindex(self, session: AsyncSession, repository: Repository) -> IndexStats:
        """对已有仓库重新解析（不重新克隆）。"""
        root = Path(repository.local_path)
        if not root.exists():
            raise FileNotFoundError(f"本地克隆不存在，需要重新克隆: {root}")

        with trace_span("repository.reindex", kind="agent", payload={"id": repository.id}):
            repository.status = "parsing"
            await session.flush()
            stats = await self._index_files(session, repository, root)
            repository.status = "ready"
            repository.file_count = stats.file_count
            repository.symbol_count = stats.symbol_count
            repository.indexed_at = _utc_now()
            await session.commit()
            await session.refresh(repository)
            return stats

    async def _index_files(
        self, session: AsyncSession, repository: Repository, root: Path
    ) -> IndexStats:
        """遍历解析并落库，同时维护统计信息。"""
        started = time.perf_counter()

        with trace_span(
            "repository.parse_files", kind="tool", payload={"root": str(root)}
        ) as span:
            parsed, notes = await asyncio.to_thread(
                _parse_repository_sync, root, self._settings
            )
            span.set_metadata(parsed_files=len(parsed))

        # 重新索引时先清空旧数据（cascade 会一并删除 symbols）
        await session.execute(delete(CodeFile).where(CodeFile.repository_id == repository.id))

        file_rows = [
            CodeFile(
                repository_id=repository.id,
                path=item.path,
                language=item.language,
                size_bytes=item.size_bytes,
                total_lines=item.total_lines,
                symbol_count=item.symbol_count,
                parse_error=item.parse_error,
            )
            for item in parsed
        ]
        session.add_all(file_rows)
        await session.flush()  # 拿到自增主键

        symbol_rows: list[dict[str, object]] = []
        for file_row, item in zip(file_rows, parsed, strict=True):
            for symbol in item.symbols:
                symbol_rows.append(
                    {
                        "file_id": file_row.id,
                        "repository_id": repository.id,
                        "name": symbol.name[:255],
                        "qualified_name": symbol.qualified_name[:512],
                        "kind": symbol.kind,
                        "start_line": symbol.start_line,
                        "end_line": symbol.end_line,
                        "signature": symbol.signature,
                        "docstring": symbol.docstring,
                        "is_async": symbol.is_async,
                        "complexity": symbol.complexity,
                    }
                )

        # 符号量可达数万条，走批量 insert 而不是 ORM 逐条 add
        chunk_size = 1000
        for index in range(0, len(symbol_rows), chunk_size):
            await session.execute(
                insert(CodeSymbol), symbol_rows[index : index + chunk_size]
            )
        await session.flush()

        failed = sum(1 for item in parsed if item.parse_error)
        return IndexStats(
            file_count=len(parsed),
            symbol_count=len(symbol_rows),
            failed_files=failed,
            duration_ms=(time.perf_counter() - started) * 1000,
            notes=notes,
        )


__all__ = ["IndexStats", "RepositoryService"]
