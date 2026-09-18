"""Git 操作服务：基于 GitPython 的仓库克隆与更新。

异步策略：GitPython 是同步阻塞库（内部调用 git 子进程），因此这里统一用
`asyncio.to_thread` 把它挪出事件循环，避免阻塞 FastAPI 的请求处理。

安全边界（面向"用户提交任意仓库地址"的场景，必须做）：
1. 只允许 http(s) 协议与白名单主机 —— 挡掉 `file://` 本地文件读取与 SSRF。
2. owner / name 用严格字符集校验 —— 因为这两个值会参与拼接本地目录名，
   不校验的话 `https://github.com/../../x/y` 可造成目录穿越。
3. 关闭 git 交互式凭据提示 —— 否则私有仓库会让请求一直挂住直到超时。
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import git
from git.exc import GitCommandError, InvalidGitRepositoryError, NoSuchPathError

from app.core.config import RepositorySettings
from app.core.trace import trace_span

logger = logging.getLogger(__name__)

# 关闭交互式凭据提示：私有仓库/失效凭据时应快速失败，而不是挂起 HTTP 请求
os.environ.setdefault("GIT_TERMINAL_PROMPT", "0")

# owner / repo 名允许的字符集（GitHub/Gitee 均在此范围内）
_SLUG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
# git@host:owner/repo.git 这种 scp 风格地址
_SCP_LIKE_PATTERN = re.compile(r"^(?P<user>[^@/]+)@(?P<host>[^:/]+):(?P<path>.+)$")


class RepositoryError(RuntimeError):
    """仓库操作基类异常。"""


class InvalidRepoUrlError(RepositoryError):
    """仓库地址不合法或不被允许。"""


class RepoCloneError(RepositoryError):
    """克隆/更新失败。"""


class RepoTooLargeError(RepositoryError):
    """仓库体积超过配置上限。"""


@dataclass(frozen=True, slots=True)
class RepoRef:
    """校验并归一化后的仓库引用。"""

    host: str
    owner: str
    name: str
    clone_url: str
    sanitized_url: str

    @property
    def slug(self) -> str:
        """本地目录名（已由字符集校验保证安全）。"""
        return f"{self.host}__{self.owner}__{self.name}"

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"


@dataclass(slots=True)
class CloneResult:
    """一次克隆/更新的结果。"""

    local_path: Path
    head_commit: str
    default_branch: str
    size_bytes: int
    # True 表示复用了已有克隆并做了增量更新
    reused: bool = False
    duration_ms: float = 0.0


def _canonical_url(
    host: str, owner: str, name: str, userinfo: str = "", scheme: str = "https"
) -> str:
    """构造规范化的克隆地址。"""
    credentials = f"{userinfo}@" if userinfo else ""
    return f"{scheme}://{credentials}{host}/{owner}/{name}.git"


def parse_repo_url(url: str, allowed_hosts: list[str] | tuple[str, ...] | set[str]) -> RepoRef:
    """校验并解析仓库地址。

    Raises:
        InvalidRepoUrlError: 协议不支持、主机不在白名单、或路径格式非法。
    """
    raw = (url or "").strip()
    if not raw:
        raise InvalidRepoUrlError("仓库地址不能为空")

    userinfo = ""
    scheme = "https"

    scp_match = _SCP_LIKE_PATTERN.match(raw)
    if scp_match and "://" not in raw:
        # git@github.com:owner/repo.git -> 统一转成 https 克隆
        host = scp_match.group("host").lower()
        path = scp_match.group("path")
    else:
        parsed = urlparse(raw)
        if parsed.scheme not in ("http", "https"):
            raise InvalidRepoUrlError(
                f"仅支持 http/https 协议，收到: {parsed.scheme or '(空)'}"
            )
        if not parsed.hostname:
            raise InvalidRepoUrlError(f"无法解析主机名: {raw}")
        scheme = parsed.scheme
        host = parsed.hostname.lower()
        userinfo = f"{parsed.username}:{parsed.password}" if parsed.username else ""
        path = parsed.path

    normalized_hosts = {item.strip().lower() for item in allowed_hosts if item.strip()}
    if host not in normalized_hosts:
        raise InvalidRepoUrlError(
            f"主机 {host} 不在白名单内，当前允许: {sorted(normalized_hosts)}"
        )

    parts = [segment for segment in path.strip("/").split("/") if segment]
    # 必须严格是 owner/repo 两段：多余路径段（网页链接 /tree/main）或
    # "." ".." 段一律拒绝，避免"静默丢弃越界段"这种含糊行为
    if len(parts) != 2 or any(segment in (".", "..") for segment in parts):
        raise InvalidRepoUrlError(
            f"仓库路径应为 owner/repo 形式，收到: {path}"
        )
    owner, name = parts[0], parts[1]
    if name.endswith(".git"):
        name = name[:-4]

    # 关键：owner/name 会拼进本地目录名，必须严格校验防目录穿越
    for label, value in (("owner", owner), ("repo", name)):
        if not value or not _SLUG_PATTERN.match(value):
            raise InvalidRepoUrlError(f"{label} 含非法字符: {value!r}")

    sanitized = _canonical_url(host, owner, name, scheme=scheme)
    return RepoRef(
        host=host,
        owner=owner,
        name=name,
        clone_url=_canonical_url(host, owner, name, userinfo=userinfo, scheme=scheme),
        sanitized_url=sanitized,
    )


def remove_tree(path: Path) -> None:
    """彻底删除目录树。

    Windows 上 git 会把 .git/objects 下的对象文件置为只读，
    `shutil.rmtree(path, ignore_errors=True)` 会因此**静默失败**并留下垃圾目录
    （曾导致超限仓库清理不掉、重新克隆时把残缺目录当成有效副本）。
    这里显式清掉只读位后逐项删除，失败也不抛异常，只告警。

    注意：调用前必须确保 GitPython 的 Repo 已 close()。gitdb 会对 pack 文件做
    mmap，句柄未释放时 Windows 会拒绝删除，表现为「目录已空但 rmdir 失败」。
    """
    if not path.exists():
        return

    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            target = Path(root) / name
            try:
                os.chmod(target, stat.S_IWRITE)
                target.unlink()
            except OSError:
                logger.warning("删除文件失败，已跳过: %s", target)
        for name in dirs:
            target = Path(root) / name
            try:
                os.chmod(target, stat.S_IWRITE)
                target.rmdir()
            except OSError:
                logger.warning("删除目录失败，已跳过: %s", target)

    try:
        os.chmod(path, stat.S_IWRITE)
        path.rmdir()
    except OSError:
        logger.warning("删除仓库目录失败: %s", path)


def _directory_size(root: Path, *, limit_bytes: int) -> int:
    """统计目录体积，超过 limit_bytes 提前返回，避免在巨型仓库上白跑。"""
    total = 0
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except (OSError, PermissionError):
            continue
        for entry in entries:
            try:
                if entry.is_symlink():
                    continue
                if entry.is_dir():
                    stack.append(entry)
                else:
                    total += entry.stat().st_size
            except OSError:
                continue
            if total > limit_bytes:
                return total
    return total


class GitService:
    """仓库克隆与更新（同步逻辑 + 异步包装）。"""

    def __init__(self, settings: RepositorySettings) -> None:
        self._settings = settings

    @property
    def workspace(self) -> Path:
        return self._settings.workspace_path

    def target_path(self, ref: RepoRef) -> Path:
        """仓库在本地的克隆路径。"""
        return self.workspace / ref.slug

    def _existing_repo(self, target: Path) -> git.Repo | None:
        """复用已有的有效克隆；目录损坏或非 git 仓库时返回 None。"""
        if not (target / ".git").exists():
            return None
        try:
            return git.Repo(target)
        except (InvalidGitRepositoryError, NoSuchPathError):
            logger.warning("已有目录不是有效 git 仓库，将重新克隆: %s", target)
            return None

    def _clone_or_update_sync(self, ref: RepoRef, branch: str | None) -> CloneResult:
        """同步执行克隆或增量更新（在工作线程中运行）。"""
        started = time.perf_counter()
        settings = self._settings
        target = self.target_path(ref)
        target.parent.mkdir(parents=True, exist_ok=True)

        repo = self._existing_repo(target)
        reused = repo is not None

        if reused:
            assert repo is not None
            origin = repo.remotes.origin
            origin.fetch(prune=True, depth=settings.clone_depth or None)
            branch_name = branch or repo.active_branch.name
            repo.git.checkout(branch_name)
            repo.git.reset("--hard", f"origin/{branch_name}")
        else:
            if target.exists():
                # 上次失败留下的残留目录，清掉再克隆
                remove_tree(target)
            repo = git.Repo.clone_from(
                ref.clone_url,
                target,
                depth=settings.clone_depth or None,
                single_branch=bool(settings.clone_depth),
                branch=branch,
            )

        head_commit = repo.head.commit.hexsha
        default_branch = repo.active_branch.name

        # 必须显式关闭：读取 commit 会让 gitdb 对 .git/objects 下的 pack 文件做
        # mmap，句柄不释放时 Windows 既删不掉目录也覆盖不了（实测确认）。
        repo.close()

        limit_bytes = settings.max_repo_size_mb * 1024 * 1024
        size_bytes = _directory_size(target, limit_bytes=limit_bytes)
        if size_bytes > limit_bytes:
            # 超限则清理，避免占满磁盘
            remove_tree(target)
            raise RepoTooLargeError(
                f"仓库体积 {size_bytes / 1024 / 1024:.1f}MB 超过上限 "
                f"{settings.max_repo_size_mb}MB"
            )

        return CloneResult(
            local_path=target,
            head_commit=head_commit,
            default_branch=default_branch,
            size_bytes=size_bytes,
            reused=reused,
            duration_ms=(time.perf_counter() - started) * 1000,
        )

    async def clone_or_update(self, ref: RepoRef, branch: str | None = None) -> CloneResult:
        """异步克隆或更新仓库。

        注意：超时用 asyncio.wait_for 实现，但线程无法被强制中断，超时后
        git 子进程仍会在后台跑完。生产环境建议后续换成独立 worker + 进程级 kill。
        """
        timeout = self._settings.clone_timeout_seconds
        with trace_span(
            "git.clone_or_update",
            kind="tool",
            payload={"repo": ref.full_name, "branch": branch},
            metadata={"host": ref.host, "depth": self._settings.clone_depth},
        ) as span:
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(self._clone_or_update_sync, ref, branch),
                    timeout=timeout,
                )
            except TimeoutError as exc:
                raise RepoCloneError(f"克隆超时（{timeout} 秒）: {ref.full_name}") from exc
            except GitCommandError as exc:
                raise RepoCloneError(f"git 命令失败: {exc.stderr or exc}") from exc
            except RepoTooLargeError:
                raise
            except RepositoryError:
                raise
            except Exception as exc:  # noqa: BLE001 - 统一收敛为领域异常
                raise RepoCloneError(f"克隆失败: {type(exc).__name__}: {exc}") from exc

            span.set_output(
                {
                    "head_commit": result.head_commit[:8],
                    "branch": result.default_branch,
                    "size_mb": round(result.size_bytes / 1024 / 1024, 2),
                    "reused": result.reused,
                }
            )
            span.set_metadata(reused=result.reused)
            return result


__all__ = [
    "CloneResult",
    "GitService",
    "InvalidRepoUrlError",
    "RepoCloneError",
    "RepoRef",
    "RepoTooLargeError",
    "RepositoryError",
    "parse_repo_url",
    "remove_tree",
]
