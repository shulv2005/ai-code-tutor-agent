"""补丁处理：从 LLM 输出提取 diff、校验、应用到隔离工作副本、回滚。

这是整个 Agent 链路里**破坏性最强**的一环：把模型生成的补丁打到代码上。
因此安全设计是重点：

1. **绝不改动用户的克隆仓库**。所有补丁只应用到 `sandbox_repo` 下的
   隔离工作副本（`prepare_worktree` 复制 + `git init` 建立基线提交），
   原始克隆始终只读。
2. **先 `--check` 再 apply**。`git apply --check` 是干跑校验，不碰文件；
   只有校验通过才真正落地，避免"打了一半"的中间状态。
3. **显式路径校验**。即使 git 自身会拒绝越界路径，也要在调用前主动挡掉
   绝对路径、`..`、以及指向 `.git` 的改动（防御性深度）。
4. **可回滚**。`revert_patch` 用 `git apply -R` 精确撤销单个补丁，
   `revert_worktree` 用 `git checkout -- .` 整体还原到基线。
5. **不使用 `--3way`**：实测浅克隆缺少必要 blob，3-way 合并会失败。

实测结论（git 2.52）：
- `git apply` 在普通目录与 git 仓库中均可工作；
- hunk 行数写错会被判 `corrupt patch`，这是 LLM 生成 diff 的常见错误；
- 上下文不匹配报 `patch does not apply`，目标文件缺失报 `No such file or directory`。
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# 补丁总长度上限：正常修复补丁不会这么大，超限基本是模型跑飞
MAX_PATCH_BYTES = 200_000
# 单个补丁最多改动的文件数
MAX_PATCH_FILES = 10

_DIFF_HEADER = re.compile(r"^diff --git a/(?P<a>\S+) b/(?P<b>\S+)\s*$")
_OLD_HEADER = re.compile(r"^---\s+(?P<path>\S+)\s*$")
_NEW_HEADER = re.compile(r"^\+\+\+\s+(?P<path>\S+)\s*$")
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@", re.MULTILINE)
# --- a/x 与 +++ b/x 相邻出现，即一份不含 diff --git 行的纯 unified diff
_UNIFIED_HEADER = re.compile(r"^---\s+\S+\r?\n\+\+\+\s+\S+\s*$", re.MULTILINE)
_FENCE = re.compile(r"```(?P<lang>[A-Za-z0-9_+-]*)[ \t]*\r?\n(?P<body>.*?)```", re.DOTALL)
_DIFF_LANGS = frozenset({"diff", "patch", "git", ""})
_DEV_NULL = "/dev/null"


def _clean_patch_path(raw: str) -> str:
    """去掉 diff 路径前缀（a/ 或 b/），其余原样保留。

    只在确实存在前缀时剥离：`a/calc.py` -> `calc.py`；
    而 `/etc/passwd` 保持原样，才能在后继校验中被识别为绝对路径。
    """
    path = raw.strip()
    for prefix in ("a/", "b/"):
        if path.startswith(prefix):
            return path[len(prefix) :]
    return path


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------
class PatchError(RuntimeError):
    """补丁处理基类异常。"""


class PatchExtractionError(PatchError):
    """无法从模型输出中提取出可用补丁。"""


class PatchValidationError(PatchError):
    """补丁内容不安全或不合法。"""


class PatchApplyError(PatchError):
    """补丁应用失败（校验不通过或执行报错）。"""


# ---------------------------------------------------------------------------
# DTO
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class PatchFileChange:
    """补丁涉及的一个文件。"""

    path: str
    additions: int = 0
    deletions: int = 0


@dataclass(slots=True)
class PatchApplyResult:
    """补丁应用结果。"""

    applied: bool
    files: list[PatchFileChange] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    # 干跑校验是否通过（applied 为 False 时用于区分"校验失败"与"未执行"）
    check_passed: bool = False


# ---------------------------------------------------------------------------
# 提取
# ---------------------------------------------------------------------------
def extract_patch(text: str) -> tuple[str, list[str]]:
    """从模型输出中提取 unified diff。

    Returns:
        (patch_text, warnings)。提取不到时 patch_text 为空串。

    模型输出常见形态：
    1. 直接给 diff（可能被 ``` 或 ```diff 包裹）
    2. 先解释再给 diff
    3. 给出"修复后的完整代码"而非 diff（本函数无法处理，由调用方重试）
    """
    warnings: list[str] = []
    if not text or not text.strip():
        return "", ["模型输出为空"]

    candidates: list[str] = []

    # 优先取标注为 diff/patch 的代码块
    for match in _FENCE.finditer(text):
        lang = match.group("lang").lower()
        body = match.group("body")
        if _looks_like_diff(body):
            candidates.append(body)
            if lang not in _DIFF_LANGS:
                warnings.append(f"代码块标注为 {lang!r} 但内容是 diff，已按补丁处理")

    # 没有围栏时，尝试从原文里切出 diff 段落
    if not candidates and _looks_like_diff(text):
        candidates.append(_strip_to_diff(text))
        warnings.append("未检测到 Markdown 代码块，已从原文中提取 diff")

    if not candidates:
        return "", ["未找到 unified diff（模型可能返回了完整代码而不是补丁）"]

    patch = max(candidates, key=len).strip()
    if len(candidates) > 1:
        warnings.append(f"检测到 {len(candidates)} 个 diff 片段，已取最长的一个")

    if not patch.endswith("\n"):
        patch += "\n"
    if len(patch.encode("utf-8")) > MAX_PATCH_BYTES:
        raise PatchExtractionError(
            f"补丁过大（>{MAX_PATCH_BYTES} 字节），已拒绝处理"
        )
    return patch, warnings


def _looks_like_diff(text: str) -> bool:
    """粗略判断文本是否是 unified diff。"""
    if "diff --git" in text:
        return True
    return bool(_UNIFIED_HEADER.search(text) and _HUNK.search(text))


def _strip_to_diff(text: str) -> str:
    """去掉 diff 之前的解释文字。"""
    match = _DIFF_HEADER.search(text)
    if match:
        return text[match.start() :]
    match = _UNIFIED_HEADER.search(text)
    if match:
        return text[match.start() :]
    return text


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------
def parse_patch_files(patch: str) -> list[PatchFileChange]:
    """解析补丁涉及的文件与增删行数。

    同时支持两种常见形态：
    1. `diff --git a/x b/x`（git 风格）
    2. 只有 `--- a/x` / `+++ b/x` 的纯 unified diff
       —— LLM 极常输出这种，而它**没有** diff --git 行，
       只认第一种会导致所有补丁被误判为"未包含任何文件改动"。
    """
    files: list[PatchFileChange] = []
    current: PatchFileChange | None = None
    pending_old: str | None = None
    # 刚由 diff --git 记过文件，紧随其后的 ---/+++ 属于同一文件，不应重复计数
    skip_headers = False

    for line in patch.splitlines():
        git_match = _DIFF_HEADER.match(line)
        if git_match:
            current = PatchFileChange(path=git_match.group("b"))
            files.append(current)
            pending_old = None
            skip_headers = True
            continue

        old_match = _OLD_HEADER.match(line)
        if old_match:
            pending_old = _clean_patch_path(old_match.group("path"))
            continue

        new_match = _NEW_HEADER.match(line)
        if new_match:
            if skip_headers:
                skip_headers = False
                pending_old = None
                continue
            new_path = _clean_patch_path(new_match.group("path"))
            # 新增文件时 +++ 侧是 /dev/null，此时用 --- 侧路径
            path = new_path if new_path and new_path != _DEV_NULL else pending_old
            if path:
                current = PatchFileChange(path=path)
                files.append(current)
            pending_old = None
            continue

        if line.startswith("@@"):
            skip_headers = False
            continue
        # "\ No newline at end of file" 之类的元信息行
        if line.startswith("\\"):
            continue
        if current is None:
            continue
        if line.startswith("+"):
            current.additions += 1
        elif line.startswith("-"):
            current.deletions += 1

    return files


def validate_patch(patch: str, *, allowed_suffixes: tuple[str, ...] = ()) -> list[str]:
    """校验补丁的安全性与合法性，返回问题列表（空列表表示通过）。

    即使 git 自身会拒绝越界路径，也要主动挡一道：多层防御，
    且能给出比 "No such file or directory" 更有用的报错。
    """
    problems: list[str] = []
    if not patch.strip():
        return ["补丁为空"]

    if not _looks_like_diff(patch):
        return ["内容不是合法的 unified diff（缺少 ---/+++ 或 @@ hunk 结构）"]

    if len(patch.encode("utf-8")) > MAX_PATCH_BYTES:
        problems.append(f"补丁过大（>{MAX_PATCH_BYTES} 字节）")

    files = parse_patch_files(patch)
    if not files:
        problems.append("补丁未包含任何文件改动")
    if len(files) > MAX_PATCH_FILES:
        problems.append(f"补丁改动文件过多（{len(files)} > {MAX_PATCH_FILES}）")

    for item in files:
        path = item.path.strip()
        if not path:
            problems.append("补丁中存在空文件路径")
            continue
        # 绝对路径与目录穿越
        if path.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", path):
            problems.append(f"补丁包含绝对路径：{path}")
        parts = Path(path.replace("\\", "/")).parts
        if ".." in parts:
            problems.append(f"补丁包含目录穿越：{path}")
        # 不允许改动 .git（否则可篡改仓库元数据）
        if ".git" in parts:
            problems.append(f"补丁试图改动 .git 目录：{path}")
        # 可选的扩展名白名单：只允许改源码，不允许改配置/CI 脚本
        if allowed_suffixes and not path.endswith(allowed_suffixes):
            problems.append(f"补丁试图改动不允许的文件类型：{path}")

    if not _HUNK.search(patch):
        problems.append("补丁缺少 hunk（@@ ... @@）")

    return problems


# ---------------------------------------------------------------------------
# 工作副本
# ---------------------------------------------------------------------------
def _run_git(worktree: Path, args: list[str], *, stdin: str | None = None) -> tuple[int, str, str]:
    """在指定目录执行 git 命令。"""
    completed = subprocess.run(
        ["git", *args],
        cwd=str(worktree),
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    return completed.returncode, completed.stdout or "", completed.stderr or ""


# 沙箱运行产物与流水线脚手架：不应计入"修复 diff"。
# 否则 Step 8 生成的 PR 草稿里会混进 .coverage 这样的二进制文件。
WORKTREE_GITIGNORE = """\
# 由 AutoFix Planner 注入：沙箱运行产物与流水线脚手架不计入修复 diff
.coverage
.coverage.*
coverage.json
coverage.xml
htmlcov/
.pytest_cache/
__pycache__/
*.pyc
*.pyo
agent_pytest.ini
"""


def prepare_worktree(
    source: Path,
    destination: Path,
    *,
    exclude: set[str] | frozenset[str],
    max_files: int = 20_000,
) -> Path:
    """把仓库复制成隔离工作副本，并建立 git 基线提交。

    为什么不用原克隆仓库：
    - 补丁可能把工作区改脏，影响后续任务与并发调用；
    - 需要一个"干净的基线"来做整体回滚（`git checkout -- .`）。

    复制时排除 `.git`（后面会重新 init）与依赖目录，
    避免把 node_modules / .venv 这种巨型目录搬过来。
    """
    if destination.exists():
        remove_worktree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    def ignore(directory: str, names: list[str]) -> set[str]:
        skipped = {name for name in names if name in exclude}
        # 始终排除原仓库的 git 元数据：我们会建立自己的基线
        skipped.add(".git")
        return skipped

    copied = 0
    for entry in source.iterdir():
        if entry.name in exclude or entry.name == ".git":
            continue
        target = destination / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target, ignore=ignore, symlinks=False, dirs_exist_ok=True)
        else:
            shutil.copy2(entry, target)
        copied += 1
        if copied > max_files:
            logger.warning("工作副本复制条目数超过上限 %d，已停止", max_files)
            break

    # 在建立基线**之前**写好 .gitignore：
    # 这样它本身进入基线提交，不会出现在最终 diff 里，
    # 而运行期产物（.coverage / coverage.json / 脚手架）会被自动排除。
    gitignore = destination / ".gitignore"
    existing = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
    gitignore.write_text(
        f"{existing.rstrip()}\n\n{WORKTREE_GITIGNORE}" if existing.strip() else WORKTREE_GITIGNORE,
        encoding="utf-8",
    )

    # 建立基线：这样才有无损回滚能力
    _run_git(destination, ["init", "-q", "-b", "main"])
    _run_git(destination, ["config", "user.email", "agent@localhost"])
    _run_git(destination, ["config", "user.name", "AutoFix Agent"])
    _run_git(destination, ["add", "-A"])
    code, _, err = _run_git(destination, ["commit", "-q", "-m", "baseline"])
    if code != 0:
        logger.warning("建立基线提交失败（可能无文件可提交）: %s", err.strip())

    return destination


def remove_worktree(path: Path) -> None:
    """删除工作副本（Windows 上只读文件需要先改权限）。"""
    import os
    import stat

    if not path.exists():
        return
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            target = Path(root) / name
            try:
                os.chmod(target, stat.S_IWRITE)
                target.unlink()
            except OSError:
                logger.debug("删除文件失败: %s", target)
        for name in dirs:
            target = Path(root) / name
            try:
                os.chmod(target, stat.S_IWRITE)
                target.rmdir()
            except OSError:
                logger.debug("删除目录失败: %s", target)
    try:
        os.chmod(path, stat.S_IWRITE)
        path.rmdir()
    except OSError:
        logger.warning("删除工作副本失败: %s", path)


def worktree_diff(worktree: Path) -> str:
    """返回工作副本相对基线的完整 diff（用于最终报告）。"""
    _run_git(worktree, ["add", "-A", "-N"])  # 让新增文件也进入 diff
    code, out, err = _run_git(worktree, ["diff", "--no-color"])
    if code != 0:
        logger.warning("生成 diff 失败: %s", err.strip())
        return ""
    return out


def changed_paths(worktree: Path) -> list[str]:
    """列出相对基线有改动的文件。"""
    code, out, _ = _run_git(worktree, ["status", "--porcelain"])
    if code != 0:
        return []
    paths: list[str] = []
    for line in out.splitlines():
        if len(line) > 3:
            paths.append(line[3:].strip().strip('"'))
    return paths


# ---------------------------------------------------------------------------
# 应用与回滚
# ---------------------------------------------------------------------------
def apply_patch(
    worktree: Path, patch: str, *, allowed_suffixes: tuple[str, ...] = ()
) -> PatchApplyResult:
    """把补丁应用到工作副本。

    流程：校验 -> `--check` 干跑 -> 实际应用。
    任何一步失败都不会让工作副本处于"打了一半"的状态。
    """
    problems = validate_patch(patch, allowed_suffixes=allowed_suffixes)
    if problems:
        return PatchApplyResult(
            applied=False, error="；".join(problems), stderr="；".join(problems)
        )

    files = parse_patch_files(patch)

    # 干跑校验：不修改任何文件
    code, out, err = _run_git(
        worktree, ["apply", "--check", "--whitespace=nowarn", "-"], stdin=patch
    )
    if code != 0:
        return PatchApplyResult(
            applied=False,
            files=files,
            stdout=out,
            stderr=err,
            error=f"补丁无法应用（校验失败）：{(err or out).strip()[:400]}",
            check_passed=False,
        )

    code, out, err = _run_git(worktree, ["apply", "--whitespace=nowarn", "-"], stdin=patch)
    if code != 0:
        return PatchApplyResult(
            applied=False,
            files=files,
            stdout=out,
            stderr=err,
            error=f"补丁应用失败：{(err or out).strip()[:400]}",
            check_passed=True,
        )

    return PatchApplyResult(
        applied=True, files=files, stdout=out, stderr=err, check_passed=True
    )


def revert_patch(worktree: Path, patch: str) -> bool:
    """撤销单个已应用的补丁（`git apply -R`）。"""
    code, _, err = _run_git(worktree, ["apply", "-R", "--whitespace=nowarn", "-"], stdin=patch)
    if code != 0:
        logger.warning("撤销补丁失败: %s", err.strip())
        return False
    return True


def revert_worktree(worktree: Path) -> bool:
    """把工作副本整体还原到基线提交。"""
    code, _, err = _run_git(worktree, ["checkout", "--", "."])
    if code != 0:
        logger.warning("还原工作副本失败: %s", err.strip())
        return False
    # 清掉补丁新增的未跟踪文件
    _run_git(worktree, ["clean", "-fdq", "--", "."])
    return True


__all__ = [
    "MAX_PATCH_BYTES",
    "MAX_PATCH_FILES",
    "PatchApplyError",
    "PatchApplyResult",
    "PatchError",
    "PatchExtractionError",
    "PatchFileChange",
    "PatchValidationError",
    "apply_patch",
    "changed_paths",
    "extract_patch",
    "parse_patch_files",
    "prepare_worktree",
    "remove_worktree",
    "revert_patch",
    "revert_worktree",
    "validate_patch",
    "worktree_diff",
]
