"""GitService 测试：真实执行 GitPython 克隆/更新（数据源为本地样例仓库）。

用本地仓库而非网络仓库，既保证测试离线可重复，又走完整真实代码路径
（GitPython 同样支持从本地路径克隆）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import get_settings
from app.services.repo.git_service import (
    GitService,
    RepoRef,
    RepoTooLargeError,
)


@pytest.fixture()
def git_service(repo_workspace: Path) -> GitService:
    return GitService(get_settings().repository)


def _ref(source: Path, name: str = "sample") -> RepoRef:
    return RepoRef(
        host="github.com",
        owner="local",
        name=name,
        clone_url=str(source),
        sanitized_url=f"https://github.com/local/{name}.git",
    )


async def test_clone_creates_working_copy(git_service: GitService, sample_repo: Path) -> None:
    result = await git_service.clone_or_update(_ref(sample_repo))

    assert result.local_path.exists()
    assert (result.local_path / "pkg" / "core.py").exists()
    assert (result.local_path / "web.js").exists()
    assert result.default_branch == "main"
    assert len(result.head_commit) == 40
    assert result.reused is False
    assert result.size_bytes > 0


async def test_second_call_reuses_clone(git_service: GitService, sample_repo: Path) -> None:
    """第二次调用应复用已有克隆做增量更新，而不是重新克隆。"""
    first = await git_service.clone_or_update(_ref(sample_repo))
    second = await git_service.clone_or_update(_ref(sample_repo))

    assert second.reused is True
    assert second.head_commit == first.head_commit
    assert second.local_path == first.local_path


async def test_picks_up_new_commits(git_service: GitService, sample_repo: Path) -> None:
    """上游有新提交时，更新应拉到最新 commit。"""
    import git

    first = await git_service.clone_or_update(_ref(sample_repo))

    source = git.Repo(sample_repo)
    (sample_repo / "pkg" / "extra.py").write_text("x = 1\n", encoding="utf-8")
    source.index.add(["pkg/extra.py"])
    actor = git.Actor("Tester", "tester@example.com")
    new_commit = source.index.commit("add extra", author=actor, committer=actor)

    second = await git_service.clone_or_update(_ref(sample_repo))
    assert second.head_commit == new_commit.hexsha
    assert second.head_commit != first.head_commit
    assert (second.local_path / "pkg" / "extra.py").exists()


async def test_repo_size_limit_is_enforced(
    git_service: GitService, sample_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """超过体积上限应拒绝并把残留目录清掉，避免占满磁盘。"""
    monkeypatch.setattr(git_service._settings, "max_repo_size_mb", 0)

    with pytest.raises(RepoTooLargeError):
        await git_service.clone_or_update(_ref(sample_repo))

    assert not git_service.target_path(_ref(sample_repo)).exists()


async def test_corrupt_existing_dir_is_recloned(
    git_service: GitService, sample_repo: Path
) -> None:
    """上次失败留下的残缺目录应被识别并重新克隆。"""
    ref = _ref(sample_repo)
    target = git_service.target_path(ref)
    target.mkdir(parents=True)
    (target / "junk.txt").write_text("leftover", encoding="utf-8")

    result = await git_service.clone_or_update(ref)
    assert result.reused is False
    assert (result.local_path / "pkg" / "core.py").exists()
    assert not (result.local_path / "junk.txt").exists()


async def test_clone_failure_is_wrapped(git_service: GitService, tmp_path: Path) -> None:
    """上游不可达时应抛领域异常，而不是泄漏 GitPython 的原始异常。"""
    missing = tmp_path / "does_not_exist"
    with pytest.raises(Exception) as excinfo:
        await git_service.clone_or_update(_ref(missing))
    assert "克隆失败" in str(excinfo.value) or "clone" in str(excinfo.value).lower()


def test_target_path_is_under_workspace(git_service: GitService, sample_repo: Path) -> None:
    ref = _ref(sample_repo)
    assert git_service.target_path(ref).parent == git_service.workspace
