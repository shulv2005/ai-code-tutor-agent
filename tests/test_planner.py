"""Planner 反馈循环测试：真实跑通「生成测试 → 执行 → 修复 → 重跑」。

用真实 git 仓库、真实 pytest 执行、真实补丁应用，只把 LLM 换成可编排的假客户端。
这样验证的是循环本身，而不是模型质量。
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import git
import pytest

from app.agents.dto import CodeContext
from app.agents.fix_agent import FixAgent
from app.agents.planner import (
    AGENT_CONFTEST,
    AGENT_TEST_SUBDIR,
    PYTEST_CONFIG_NAME,
    AutoFixPlanner,
    failure_signature,
)
from app.agents.test_agent import TestAgent
from app.core.config import get_settings
from app.services.sandbox import LocalSubprocessSandboxRunner, SandboxResult
from tests.conftest import FakeLLMClient

BUGGY = "def add(a, b):\n    return a - b\n"
FIXED = "def add(a, b):\n    return a + b\n"

# 修复循环里测试必须导入真实模块（而不是内联替身），否则打补丁无效
GENERATED_TEST = '''```python
from calc import add


def test_add_sums_two_numbers():
    assert add(1, 2) == 3
```
'''

FIX_PATCH = """\
## 分析
实现用的是减号，测试期望求和。这是源码 bug。

## 归类
source_bug

## 补丁
```diff
--- a/calc.py
+++ b/calc.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a - b
+    return a + b
```
"""

# 有效补丁但改错了地方：失败仍在，用于测 max_attempts
WRONG_FIX_PATCH = """\
## 分析
改成乘法试试。

## 归类
source_bug

## 补丁
```diff
--- a/calc.py
+++ b/calc.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a - b
+    return a * b
```
"""

# 只改注释：失败输出完全不变，用于测"无进展检测"
NOOP_PATCH = """\
## 分析
加个注释。

## 归类
source_bug

## 补丁
```diff
--- a/calc.py
+++ b/calc.py
@@ -1,2 +1,3 @@
+# helper
 def add(a, b):
     return a - b
```
"""

STALE_PATCH = """\
## 分析
上下文对不上。

## 归类
source_bug

## 补丁
```diff
--- a/calc.py
+++ b/calc.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a * 999
+    return a + b
```
"""

ENV_REPLY = """\
## 分析
容器缺少依赖。

## 归类
environment

## 补丁
无需补丁
"""


@pytest.fixture()
def planner_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """把沙箱工作区指向临时目录并强制 local 后端。"""
    workspace = tmp_path / "workspaces"
    monkeypatch.setenv("DOCKER__WORKSPACE_DIR", str(workspace))
    monkeypatch.setenv("DOCKER__BACKEND", "local")
    monkeypatch.setenv("DOCKER__INSTALL_DEPENDENCIES", "false")
    get_settings.cache_clear()
    workspace.mkdir(parents=True, exist_ok=True)
    yield workspace
    get_settings.cache_clear()


@pytest.fixture()
def buggy_clone(tmp_path: Path) -> Path:
    """一个含真实 bug 的本地克隆仓库。"""
    root = tmp_path / "clone"
    root.mkdir()
    (root / "calc.py").write_text(BUGGY, encoding="utf-8")
    repo = git.Repo.init(root, initial_branch="main")
    repo.index.add(["calc.py"])
    actor = git.Actor("T", "t@e.com")
    repo.index.commit("init", author=actor, committer=actor)
    repo.close()
    return root


def _context() -> CodeContext:
    return CodeContext(
        path="calc.py",
        code=BUGGY,
        qualified_name="add",
        kind="function",
        signature="def add(a, b)",
        language="python",
    )


def _planner(responses: list[str]) -> tuple[AutoFixPlanner, FakeLLMClient]:
    fake = FakeLLMClient(responses)
    settings = get_settings()
    planner = AutoFixPlanner(
        settings,
        test_agent=TestAgent(settings, fake),
        fix_agent=FixAgent(settings, fake),
        sandbox=LocalSubprocessSandboxRunner(settings.docker),
    )
    return planner, fake


# ---------------------------------------------------------------------------
# 主路径：一轮修复后测试通过
# ---------------------------------------------------------------------------
async def test_loop_converges_after_one_fix(
    planner_env: Path, buggy_clone: Path
) -> None:
    """完整闭环：生成测试 → 执行失败 → 生成补丁 → 应用 → 重跑通过。"""
    planner, fake = _planner([GENERATED_TEST, FIX_PATCH])

    result = await planner.run(_context(), buggy_clone, max_attempts=3, run_id="loop1")

    assert result.status == "passed", result.message
    assert result.success is True
    assert len(result.iterations) == 2

    # 第 1 轮：测试失败，随后生成了补丁（记录的是这一整轮的动作）
    first = result.iterations[0]
    assert first.test_result is not None
    assert first.test_result.failed == 1
    assert first.patch_applied is True
    assert first.proposal is not None
    assert first.proposal.category == "source_bug"

    # 第 2 轮：补丁已应用并测试通过
    second = result.iterations[1]
    assert second.test_result is not None
    assert second.test_result.succeeded is True
    assert second.test_result.passed >= 1

    # 最终 diff 应体现修复
    assert "calc.py" in result.changed_files
    assert "+    return a + b" in result.final_diff
    assert fake.call_count == 2  # 1 次生成测试 + 1 次生成补丁


async def test_source_clone_is_never_modified(
    planner_env: Path, buggy_clone: Path
) -> None:
    """核心安全属性：修复只作用于工作副本，原始克隆保持不变。"""
    planner, _ = _planner([GENERATED_TEST, FIX_PATCH])

    await planner.run(_context(), buggy_clone, max_attempts=3, run_id="loop2")

    assert (buggy_clone / "calc.py").read_text(encoding="utf-8") == BUGGY


async def test_worktree_contains_patch_and_generated_test(
    planner_env: Path, buggy_clone: Path
) -> None:
    planner, _ = _planner([GENERATED_TEST, FIX_PATCH])
    result = await planner.run(_context(), buggy_clone, max_attempts=3, run_id="loop3")

    assert result.worktree is not None
    worktree = Path(result.worktree)
    assert (worktree / "calc.py").read_text(encoding="utf-8") == FIXED
    assert (worktree / AGENT_TEST_SUBDIR / "test_generated.py").exists()
    # 隔离的 pytest 配置也被写入，避免仓库自带 addopts 干扰
    assert (worktree / "agent_pytest.ini").exists()


async def test_already_passing_test_needs_no_fix(
    planner_env: Path, buggy_clone: Path
) -> None:
    """测试直接通过时不应调用 Fix Agent。"""
    (buggy_clone / "calc.py").write_text(FIXED, encoding="utf-8")
    planner, fake = _planner([GENERATED_TEST])

    result = await planner.run(_context(), buggy_clone, max_attempts=3, run_id="loop4")

    assert result.status == "passed"
    assert len(result.iterations) == 1
    assert fake.call_count == 1  # 只调用了测试生成


# ---------------------------------------------------------------------------
# 终止条件
# ---------------------------------------------------------------------------
async def test_stops_at_max_attempts(planner_env: Path, buggy_clone: Path) -> None:
    """补丁改了失败特征但没修好，应跑满轮次后停止。"""
    planner, fake = _planner([GENERATED_TEST, WRONG_FIX_PATCH, WRONG_FIX_PATCH])
    result = await planner.run(_context(), buggy_clone, max_attempts=2, run_id="loop5")

    assert result.status == "max_attempts"
    assert result.success is False
    assert len(result.iterations) == 2
    assert "最大修复轮次" in result.message


async def test_detects_no_progress_and_stops_early(
    planner_env: Path, buggy_clone: Path
) -> None:
    """补丁只改注释、失败输出完全一致时，应提前终止而不是烧完剩余轮次。"""
    planner, _ = _planner([GENERATED_TEST, NOOP_PATCH, NOOP_PATCH])
    result = await planner.run(_context(), buggy_clone, max_attempts=3, run_id="loop6")

    assert result.status == "no_progress"
    assert len(result.iterations) == 2  # 第 2 轮就识别出无进展
    assert "无进展" in result.message


async def test_environment_failure_is_not_patched(
    planner_env: Path, buggy_clone: Path
) -> None:
    """归类为环境问题时按设计不出补丁，如实报告而不是硬凑。"""
    planner, fake = _planner([GENERATED_TEST, ENV_REPLY])
    result = await planner.run(_context(), buggy_clone, max_attempts=3, run_id="loop7")

    assert result.status == "not_patchable"
    assert result.success is False
    assert "environment" in result.message
    assert fake.call_count == 2


async def test_consecutive_patch_rejections_terminate(
    planner_env: Path, buggy_clone: Path
) -> None:
    """补丁连续无法应用时终止，且工作副本保持干净。"""
    planner, _ = _planner([GENERATED_TEST, STALE_PATCH, STALE_PATCH])
    result = await planner.run(_context(), buggy_clone, max_attempts=3, run_id="loop8")

    assert result.status == "patch_rejected"
    assert all(not item.patch_applied for item in result.iterations)
    # 补丁没应用成功，工作副本源码应保持原样
    assert result.worktree is not None
    assert (Path(result.worktree) / "calc.py").read_text(encoding="utf-8") == BUGGY


async def test_no_patch_from_model(planner_env: Path, buggy_clone: Path) -> None:
    planner, _ = _planner([GENERATED_TEST, "## 分析\n不清楚\n\n## 归类\nunclear\n"])
    result = await planner.run(_context(), buggy_clone, max_attempts=3, run_id="loop9")

    assert result.status == "no_patch"
    assert result.success is False


# ---------------------------------------------------------------------------
# 工作副本生命周期
# ---------------------------------------------------------------------------
async def test_worktree_can_be_discarded(planner_env: Path, buggy_clone: Path) -> None:
    planner, _ = _planner([GENERATED_TEST, FIX_PATCH])
    result = await planner.run(
        _context(), buggy_clone, max_attempts=2, run_id="loop10", keep_worktree=False
    )

    assert result.worktree is None
    assert not planner.worktree_path("loop10").exists()


async def test_worktree_is_rebuilt_between_runs(
    planner_env: Path, buggy_clone: Path
) -> None:
    """同一 run_id 重跑应重建干净副本，不能带着上一轮的改动。"""
    planner, _ = _planner([GENERATED_TEST, FIX_PATCH])
    await planner.run(_context(), buggy_clone, max_attempts=2, run_id="same")
    assert (planner.worktree_path("same") / "calc.py").read_text(encoding="utf-8") == FIXED

    planner2, _ = _planner([GENERATED_TEST])
    result = await planner2.run(
        _context(), buggy_clone, max_attempts=2, run_id="same"
    )
    # 重建后回到未修复状态（测试会失败）
    assert (planner2.worktree_path("same") / "calc.py").read_text(encoding="utf-8") == BUGGY
    assert result.iterations[0].test_result is not None


# ---------------------------------------------------------------------------
# 失败特征
# ---------------------------------------------------------------------------
def test_failure_signature_is_stable_and_discriminating() -> None:
    def make(exit_code: int, failed: int, stderr: str) -> SandboxResult:
        return SandboxResult(
            exit_code=exit_code, stdout="out", stderr=stderr,
            duration_ms=1.0, backend="local", failed=failed,
        )

    a = make(1, 1, "assert -1 == 3")
    b = make(1, 1, "assert -1 == 3")
    c = make(1, 1, "assert 2 == 3")
    d = make(0, 0, "assert -1 == 3")

    assert failure_signature(a) == failure_signature(b)
    assert failure_signature(a) != failure_signature(c)
    assert failure_signature(a) != failure_signature(d)


def test_failure_signature_ignores_timing_and_paths() -> None:
    """回归：耗时与绝对路径是易变内容，不能影响指纹。

    曾因 pytest 汇总行的 `in 0.04s` 每次不同，导致"无进展检测"偶发失效。
    """
    fast = SandboxResult(
        exit_code=1,
        stdout="F  [100%]\nFAILED agent_tests/test_x.py::test_a - assert -1 == 3\n1 failed in 0.03s\n",
        stderr="",
        duration_ms=30.0,
        backend="local",
        failed=1,
    )
    slow = SandboxResult(
        exit_code=1,
        stdout="F  [100%]\nFAILED agent_tests/test_x.py::test_a - assert -1 == 3\n1 failed in 1.87s\n",
        stderr="",
        duration_ms=1870.0,
        backend="local",
        failed=1,
    )
    assert failure_signature(fast) == failure_signature(slow)


# ---------------------------------------------------------------------------
# 生成测试目录的收集过滤（回归：真实仓库上跑出来的一个致命坑）
# ---------------------------------------------------------------------------
def test_prepare_writes_collection_filter(planner_env: Path, buggy_clone: Path) -> None:
    """prepare() 必须在 agent_tests/ 放好收集过滤器。"""
    planner, _ = _planner(["x"])
    worktree = planner.prepare(buggy_clone, "filter1")

    conftest = worktree / AGENT_TEST_SUBDIR / "conftest.py"
    assert conftest.is_file()
    assert "pytest_collection_modifyitems" in conftest.read_text(encoding="utf-8")
    # 隔离用的 pytest 配置也要在
    assert (worktree / PYTEST_CONFIG_NAME).is_file()


def test_collection_filter_drops_imported_tests(tmp_path: Path) -> None:
    """过滤器要精确剔除"import 进来的"测试，且不误伤本目录自己的测试。

    这是真实踩到的坑：模型常写 `from tests.test_requests import TestRequests`，
    pytest 会把 import 进来的测试类再收集一遍，凭空多出上百个收集错误，
    让修复循环永远不收敛（实测 79 passed / 180 errors）。
    这里用真实的 pytest 子进程验证修复效果。
    """
    import subprocess
    import sys

    # 伪造一个"仓库自带测试模块"
    (tmp_path / "repo_tests.py").write_text(
        "class TestFromRepo:\n"
        "    def test_should_not_run_here(self):\n"
        "        raise AssertionError('import 进来的测试不该被收集')\n",
        encoding="utf-8",
    )
    # 再伪造一个"生成文件"：import 了仓库的测试类，自己也定义了一个
    (tmp_path / "test_generated.py").write_text(
        "from repo_tests import TestFromRepo\n"
        "\n"
        "import pytest\n"
        "\n"
        "\n"
        "def test_own_case():\n"
        "    assert True\n"
        "\n"
        "\n"
        "def test_uses_imported_class():\n"
        "    with pytest.raises(AssertionError):\n"
        "        TestFromRepo().test_should_not_run_here()\n",
        encoding="utf-8",
    )
    (tmp_path / "conftest.py").write_text(AGENT_CONFTEST, encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "pytest", ".", "-q", "-p", "no:cacheprovider"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = result.stdout + result.stderr
    # 自己的两个用例都跑，import 进来的那个被剔除
    assert "2 passed" in output, output
    assert "TestFromRepo::test_should_not_run_here" not in output, output


def test_collection_filter_keeps_local_test_class(tmp_path: Path) -> None:
    """本文件自己定义的测试类（__module__ 一致）不能被误删。"""
    import subprocess
    import sys

    (tmp_path / "test_generated.py").write_text(
        "class TestOwnClass:\n"
        "    def test_inside(self):\n"
        "        assert True\n",
        encoding="utf-8",
    )
    (tmp_path / "conftest.py").write_text(AGENT_CONFTEST, encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "pytest", ".", "-q", "-p", "no:cacheprovider"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert "1 passed" in result.stdout + result.stderr
