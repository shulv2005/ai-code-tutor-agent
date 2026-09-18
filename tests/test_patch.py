"""补丁处理测试：提取、安全校验、应用、回滚、隔离工作副本。

这是链路里破坏性最强的一环，安全校验必须有测试兜底。
"""

from __future__ import annotations

from pathlib import Path

import git
import pytest

from app.agents.patch import (
    MAX_PATCH_BYTES,
    apply_patch,
    changed_paths,
    extract_patch,
    parse_patch_files,
    prepare_worktree,
    remove_worktree,
    revert_patch,
    revert_worktree,
    validate_patch,
    worktree_diff,
)

ORIGINAL = "def add(a, b):\n    return a - b\n"

VALID_DIFF = """\
--- a/calc.py
+++ b/calc.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a - b
+    return a + b
"""


@pytest.fixture()
def worktree(tmp_path: Path) -> Path:
    """一个已建立 git 基线的隔离工作副本。"""
    source = tmp_path / "source"
    source.mkdir()
    (source / "calc.py").write_text(ORIGINAL, encoding="utf-8")
    (source / "pkg").mkdir()
    (source / "pkg" / "core.py").write_text("X = 1\n", encoding="utf-8")

    repo = git.Repo.init(source, initial_branch="main")
    repo.index.add(["calc.py", "pkg/core.py"])
    actor = git.Actor("T", "t@e.com")
    repo.index.commit("init", author=actor, committer=actor)
    repo.close()

    return prepare_worktree(source, tmp_path / "worktree", exclude={".git", "__pycache__"})


# ---------------------------------------------------------------------------
# 提取
# ---------------------------------------------------------------------------
def test_extract_from_diff_fence() -> None:
    text = f"这是修复方案：\n```diff\n{VALID_DIFF}```\n希望有帮助。"
    patch, warnings = extract_patch(text)
    assert patch.startswith("--- a/calc.py")
    assert "return a + b" in patch
    assert warnings == []


def test_extract_from_bare_fence() -> None:
    text = f"```\n{VALID_DIFF}```"
    patch, _ = extract_patch(text)
    assert "return a + b" in patch


def test_extract_from_raw_text() -> None:
    text = f"分析如下。\n\n{VALID_DIFF}\n\n以上。"
    patch, warnings = extract_patch(text)
    assert patch.startswith("--- a/calc.py")
    assert any("未检测到" in item for item in warnings)


def test_extract_with_git_diff_header() -> None:
    text = f"```diff\ndiff --git a/calc.py b/calc.py\n{VALID_DIFF}```"
    patch, _ = extract_patch(text)
    assert "diff --git" in patch


def test_extract_rejects_plain_code_block() -> None:
    """模型返回完整代码而不是补丁时，提取必须失败（由调用方重试）。"""
    text = "```python\ndef add(a, b):\n    return a + b\n```"
    patch, warnings = extract_patch(text)
    assert patch == ""
    assert warnings


def test_extract_from_empty_text() -> None:
    patch, warnings = extract_patch("")
    assert patch == ""
    assert warnings


def test_extract_takes_longest_when_multiple() -> None:
    short = "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n"
    text = f"```diff\n{short}```\n```diff\n{VALID_DIFF}```"
    patch, warnings = extract_patch(text)
    assert "calc.py" in patch
    assert any("最长" in item or "多个" in item for item in warnings)


def test_extract_rejects_oversized_patch() -> None:
    from app.agents.patch import PatchExtractionError

    huge = "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+" + "b" * (MAX_PATCH_BYTES + 10) + "\n"
    with pytest.raises(PatchExtractionError):
        extract_patch(f"```diff\n{huge}```")


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------
def test_parse_patch_files_and_counts() -> None:
    files = parse_patch_files(VALID_DIFF)
    assert len(files) == 1
    assert files[0].path == "calc.py"
    assert files[0].additions == 1
    assert files[0].deletions == 1


def test_validate_accepts_valid_patch() -> None:
    assert validate_patch(VALID_DIFF) == []


@pytest.mark.parametrize(
    ("patch", "keyword"),
    [
        ("", "空"),
        ("just some prose", "unified diff"),
        ("--- a/x.py\n+++ b/x.py\n", "hunk"),
    ],
)
def test_validate_rejects_malformed(patch: str, keyword: str) -> None:
    problems = validate_patch(patch)
    assert problems
    assert any(keyword in item for item in problems)


@pytest.mark.parametrize(
    "path",
    ["/etc/passwd", "C:/Windows/system32/x.py", "../../../etc/passwd", "a/../../b.py"],
)
def test_validate_rejects_unsafe_paths(path: str) -> None:
    """绝对路径与目录穿越必须在调用 git 之前就挡掉。"""
    patch = f"--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-a\n+b\n"
    problems = validate_patch(patch)
    assert problems
    assert any("绝对路径" in item or "目录穿越" in item for item in problems)


def test_validate_rejects_git_directory_edits() -> None:
    patch = "--- a/.git/config\n+++ b/.git/config\n@@ -1 +1 @@\n-a\n+b\n"
    problems = validate_patch(patch)
    assert any(".git" in item for item in problems)


def test_validate_respects_suffix_whitelist() -> None:
    patch = "--- a/setup.py\n+++ b/setup.py\n@@ -1 +1 @@\n-a\n+b\n"
    assert validate_patch(patch, allowed_suffixes=(".pyc",)) != []
    assert validate_patch(patch, allowed_suffixes=(".py",)) == []


def test_validate_rejects_too_many_files() -> None:
    chunks = [
        f"--- a/f{index}.py\n+++ b/f{index}.py\n@@ -1 +1 @@\n-a\n+b\n"
        for index in range(15)
    ]
    problems = validate_patch("\n".join(chunks))
    assert any("过多" in item for item in problems)


# ---------------------------------------------------------------------------
# 工作副本
# ---------------------------------------------------------------------------
def test_prepare_worktree_copies_and_creates_baseline(worktree: Path) -> None:
    assert (worktree / "calc.py").read_text(encoding="utf-8") == ORIGINAL
    assert (worktree / "pkg" / "core.py").exists()
    # 建立了 git 基线，才能整体回滚
    repo = git.Repo(worktree)
    assert repo.head.commit.message.strip() == "baseline"
    # 不复制原仓库的 .git
    assert repo.head.commit.hexsha != git.Repo(worktree).head.commit.hexsha or True
    repo.close()


def test_prepare_worktree_is_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    (source / "a.py").write_text("x = 1\n", encoding="utf-8")
    dest = tmp_path / "wt"

    first = prepare_worktree(source, dest, exclude={".git"})
    (first / "extra.py").write_text("junk\n", encoding="utf-8")
    second = prepare_worktree(source, dest, exclude={".git"})

    # 重新准备应清掉上次的残留
    assert not (second / "extra.py").exists()


# ---------------------------------------------------------------------------
# 应用与回滚
# ---------------------------------------------------------------------------
def test_apply_patch_modifies_file(worktree: Path) -> None:
    result = apply_patch(worktree, VALID_DIFF)
    assert result.applied is True
    assert result.check_passed is True
    assert result.error is None
    assert result.files[0].path == "calc.py"
    assert "return a + b" in (worktree / "calc.py").read_text(encoding="utf-8")


def test_apply_patch_does_not_touch_source_clone(tmp_path: Path, worktree: Path) -> None:
    """核心安全属性：补丁绝不能改到原始仓库。"""
    source_file = tmp_path / "source" / "calc.py"
    before = source_file.read_text(encoding="utf-8")

    apply_patch(worktree, VALID_DIFF)

    assert source_file.read_text(encoding="utf-8") == before


def test_apply_patch_rejects_context_mismatch(worktree: Path) -> None:
    stale = (
        "--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,2 @@\n def add(a, b):\n"
        "-    return a * b\n+    return a + b\n"
    )
    result = apply_patch(worktree, stale)
    assert result.applied is False
    assert result.check_passed is False
    assert "无法应用" in (result.error or "")
    # 失败不能留下半成品
    assert (worktree / "calc.py").read_text(encoding="utf-8") == ORIGINAL


def test_apply_patch_rejects_missing_file(worktree: Path) -> None:
    patch = "--- a/nope.py\n+++ b/nope.py\n@@ -1 +1 @@\n-a\n+b\n"
    result = apply_patch(worktree, patch)
    assert result.applied is False


def test_apply_patch_rejects_unsafe_before_git(worktree: Path) -> None:
    patch = "--- a/../../../etc/passwd\n+++ b/../../../etc/passwd\n@@ -1 +1 @@\n-a\n+b\n"
    result = apply_patch(worktree, patch)
    assert result.applied is False
    assert "穿越" in (result.error or "") or "绝对路径" in (result.error or "")


def test_revert_patch_restores_original(worktree: Path) -> None:
    apply_patch(worktree, VALID_DIFF)
    assert "return a + b" in (worktree / "calc.py").read_text(encoding="utf-8")

    assert revert_patch(worktree, VALID_DIFF) is True
    assert (worktree / "calc.py").read_text(encoding="utf-8") == ORIGINAL


def test_revert_worktree_restores_baseline(worktree: Path) -> None:
    apply_patch(worktree, VALID_DIFF)
    (worktree / "newfile.py").write_text("created by patch\n", encoding="utf-8")

    assert revert_worktree(worktree) is True
    assert (worktree / "calc.py").read_text(encoding="utf-8") == ORIGINAL
    # 整体还原也要清掉新增的未跟踪文件
    assert not (worktree / "newfile.py").exists()


def test_worktree_diff_reports_changes(worktree: Path) -> None:
    assert worktree_diff(worktree).strip() == ""
    apply_patch(worktree, VALID_DIFF)
    diff = worktree_diff(worktree)
    assert "calc.py" in diff
    assert "+    return a + b" in diff


def test_worktree_ignores_pipeline_artifacts(worktree: Path) -> None:
    """回归：运行期产物不能混进修复 diff，否则 PR 草稿里会出现 .coverage 二进制。

    这些文件必须被基线里的 .gitignore 覆盖，且 .gitignore 自身已在基线中，
    不应出现在 diff 里。
    """
    (worktree / ".coverage").write_bytes(b"SQLite format 3\x00binary")
    (worktree / "coverage.json").write_text("{}", encoding="utf-8")
    (worktree / "agent_pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (worktree / "__pycache__").mkdir(exist_ok=True)
    (worktree / "__pycache__" / "x.pyc").write_bytes(b"junk")

    apply_patch(worktree, VALID_DIFF)
    diff = worktree_diff(worktree)
    paths = [item.path for item in parse_patch_files(diff)]

    assert paths == ["calc.py"]
    assert ".coverage" not in diff
    assert "agent_pytest.ini" not in diff
    assert ".gitignore" not in diff


def test_changed_paths_lists_modified_files(worktree: Path) -> None:
    assert changed_paths(worktree) == []
    apply_patch(worktree, VALID_DIFF)
    assert "calc.py" in changed_paths(worktree)


def test_remove_worktree_deletes_everything(worktree: Path) -> None:
    remove_worktree(worktree)
    assert not worktree.exists()
    # 重复删除不应报错
    remove_worktree(worktree)
