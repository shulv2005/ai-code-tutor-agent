"""Planner：串联「生成测试 → 沙箱执行 → 生成补丁 → 应用 → 重跑」的反馈闭环。

工作流位置：Step 6/7，把 Step 3/4/5 串成可自动收敛的修复循环。

    隔离工作副本（克隆副本 + git 基线）
        ├─ Test Agent 生成测试
        ├─ Sandbox 执行
        │     ├─ 通过 → 结束
        │     └─ 失败 → Fix Agent 生成补丁 → 校验并应用 → 回到执行
        └─ 最多 max_attempts 轮，或检测到"无进展"提前停止

三条安全/效率护栏：
1. **隔离工作副本**：所有补丁只打到 `sandbox_repo` 的副本上，用户的克隆仓库
   全程只读。副本自带 git 基线，可随时整体回滚。
2. **无进展检测**：若补丁应用后失败特征与上一轮完全一致，说明这条路走不通，
   立即停止而不是把剩余轮次白白烧掉。
3. **不可修补类别直接终止**：Fix Agent 判定为 environment / already_fixed 时
   不出补丁，Planner 相应地结束并如实报告，而不是硬凑一个补丁。
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from app.agents.context import ContextError
from app.agents.dto import CodeContext
from app.agents.fix_agent import (
    PATCHABLE_CATEGORIES,
    FailureContext,
    FixAgent,
    FixProposal,
)
from app.agents.patch import (
    apply_patch,
    changed_paths,
    prepare_worktree,
    remove_worktree,
    worktree_diff,
)
from app.agents.test_agent import TestAgent, TestGenerationFailed
from app.core.config import Settings
from app.core.trace import current_trace_id, trace_span
from app.services.sandbox import SandboxResult, SandboxRunner, SandboxRunRequest

logger = logging.getLogger(__name__)

# 生成测试放在独立目录：不能与仓库自带 tests/ 混在一起，
# 否则仓库原有测试会被一并收集执行，失败原因与本次修复无关。
AGENT_TEST_SUBDIR = "agent_tests"
# 传给 pytest 的隔离配置文件名
PYTEST_CONFIG_NAME = "agent_pytest.ini"

# 循环终止原因
LoopStatus = Literal[
    "passed",            # 测试通过
    "max_attempts",      # 达到最大轮次仍未通过
    "no_patch",          # 模型给不出可用补丁
    "not_patchable",     # 归类为环境问题/已修复，按设计不出补丁
    "patch_rejected",    # 补丁连续无法应用
    "no_progress",       # 补丁应用后失败特征完全一致
    "error",             # 执行过程中的异常
]

PYTEST_CONFIG = """\
[pytest]
# 由 AutoFix Planner 生成：隔离仓库自带的 pytest 配置，
# 避免其 addopts 依赖沙箱内未安装的插件导致 pytest 直接报错退出。
#
# 注意：这里**不要**再写 -q —— 命令本身已经带了 -q，叠加会变成 -qq，
# 而 -qq 会抑制 "1 failed" 汇总行，导致失败计数解析不到、反馈闭环失灵。
addopts = -p no:cacheprovider
"""

# 放进 agent_tests/ 的 conftest.py：把"import 进来的"仓库测试类排除出收集范围。
#
# 为什么必须有它（在真实仓库上跑出来才发现的坑，别删）：
# 模型很常见的做法是 `from tests.test_requests import TestRequests` 之后调用
# 仓库已有的用例方法。pytest 会把**import 进来的测试类**当成当前模块的测试类
# 再收集一遍，于是凭空多出上百个收集/初始化错误（仓库的测试依赖 httpbin 之类
# 并没有装进沙箱）。后果是致命的：
#   1. SandboxResult.succeeded 永远为 False —— 哪怕真正的 bug 已经修好，
#      修复循环也永远不会收敛（status 卡在 max_attempts）；
#   2. 失败特征里全是无关错误，Fix Agent 拿不到有效信号，只能"给不出补丁"。
# 实测（requests 仓库）：一个 5 个用例的生成文件跑出 `79 passed, 180 errors`，
# 那 180 个错误全部来自被重复收集的仓库自带测试类。
#
# 为什么用 conftest 而不是给生成文件追加 `__test__ = False`：
# 后者是修改 import 进来的类对象本身，会连带把仓库自己的 `tests/test_requests.py`
# 里的同一个类也一起屏蔽掉（当整仓测试被收集时），属于误伤。
# conftest 只影响本次会话的收集结果，判据也精确：
# 当某个用例所属类的 `__module__` 与它所在模块不一致时，说明它是 import 来的。
AGENT_CONFTEST = '''\
"""由 AutoFix Planner 生成：让 agent_tests 只跑"本目录自己写的"测试。

判据：用例所属类/函数的 __module__ 与它被收集到的模块不一致 -> 它是 import 来的，
不是本目录定义的测试，收集阶段直接剔除。仓库自带的测试目录不受本文件影响。
"""

from __future__ import annotations


def pytest_collection_modifyitems(config, items):  # noqa: ANN001, ARG001
    """剔除"从其它模块 import 进来"的测试用例。"""
    kept = []
    for item in items:
        owner = getattr(item, "cls", None) or getattr(item, "obj", None)
        owner_module = getattr(owner, "__module__", None)
        module = getattr(getattr(item, "module", None), "__name__", None)
        # 拿不到模块信息时一律保留：宁可多跑，不可误删真正的测试
        if owner_module is None or module is None or owner_module == module:
            kept.append(item)
    items[:] = kept
'''


@dataclass(slots=True)
class LoopIteration:
    """一轮循环的记录。"""

    index: int
    test_result: SandboxResult | None = None
    proposal: FixProposal | None = None
    patch_applied: bool = False
    patch_error: str | None = None
    note: str = ""

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "index": self.index,
            "patch_applied": self.patch_applied,
            "note": self.note,
        }
        if self.test_result is not None:
            payload["tests"] = {
                "exit_code": self.test_result.exit_code,
                "passed": self.test_result.passed,
                "failed": self.test_result.failed,
                "errors": self.test_result.errors,
                "duration_ms": round(self.test_result.duration_ms, 1),
            }
        if self.proposal is not None:
            payload["category"] = self.proposal.category
            payload["analysis"] = self.proposal.analysis[:500]
            payload["patch_files"] = [item.path for item in _patch_files(self.proposal.patch)]
        if self.patch_error:
            payload["patch_error"] = self.patch_error
        return payload


def _patch_files(patch: str) -> list:
    from app.agents.patch import parse_patch_files

    return parse_patch_files(patch) if patch else []


@dataclass(slots=True)
class AutoFixResult:
    """自动修复的完整结果。"""

    run_id: str
    status: LoopStatus
    success: bool
    iterations: list[LoopIteration] = field(default_factory=list)
    target: CodeContext | None = None
    generated_test: str = ""
    final_test: SandboxResult | None = None
    # 累计应用成功的补丁
    applied_patch: str = ""
    # 工作副本相对基线的最终 diff（Step 8 生成 PR 用）
    final_diff: str = ""
    changed_files: list[str] = field(default_factory=list)
    worktree: str | None = None
    message: str = ""
    timings_ms: dict[str, float] = field(default_factory=dict)
    trace_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "success": self.success,
            "attempts": len(self.iterations),
            "changed_files": self.changed_files,
            "message": self.message,
            "iterations": [item.to_dict() for item in self.iterations],
        }


# 签名必须剔掉的易变内容：否则同一失败在不同次运行里会得到不同指纹，
# "无进展检测"就永远不生效（曾因 pytest 的 `in 0.04s` 计时导致偶发失效）。
_VOLATILE_PATTERNS = (
    re.compile(r"\bin\s+\d+(?:\.\d+)?s\b"),          # pytest 耗时
    re.compile(r"0x[0-9a-fA-F]+"),                    # 对象地址
    re.compile(r"\d+(?:\.\d+)?\s*(?:ms|us|µs)\b"),   # 毫秒/微秒
    re.compile(r"[A-Za-z]:\\[^\s\"']+"),              # Windows 绝对路径
    re.compile(r"/(?:tmp|var|home)/[^\s\"']+"),       # POSIX 临时路径
)


def _normalize_for_signature(text: str) -> str:
    """去掉输出中的易变成分，使其只反映"失败本身"。"""
    normalized = text
    for pattern in _VOLATILE_PATTERNS:
        normalized = pattern.sub("<var>", normalized)
    return " ".join(normalized.split())


def failure_signature(result: SandboxResult) -> str:
    """为一次失败生成特征指纹，用于检测"补丁没带来任何变化"。

    只取"失败/错误的用例数 + 错误信息"，并剔除耗时、路径等易变内容，
    保证同一失败在不同运行中得到相同指纹。
    """
    stderr_tail = _normalize_for_signature((result.stderr or "")[-1000:])
    stdout_tail = _normalize_for_signature((result.stdout or "")[-1000:])
    raw = f"{result.exit_code}|{result.failed}|{result.errors}|{stderr_tail}|{stdout_tail}"
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:16]


class AutoFixPlanner:
    """自动修复编排器。"""

    def __init__(
        self,
        settings: Settings,
        *,
        test_agent: TestAgent,
        fix_agent: FixAgent,
        sandbox: SandboxRunner,
    ) -> None:
        self._settings = settings
        self._test_agent = test_agent
        self._fix_agent = fix_agent
        self._sandbox = sandbox

    # -- 工作副本 ---------------------------------------------------------
    def worktree_path(self, run_id: str) -> Path:
        """隔离工作副本路径（与 Test Agent 的 sandbox_root 一致）。"""
        return self._test_agent.sandbox_root(run_id)

    def prepare(
        self, clone_path: Path, run_id: str
    ) -> Path:
        """从克隆仓库建立隔离工作副本。

        刻意复制副本而不是直接改克隆：
        - 补丁可能把工作区改脏，影响后续任务与并发调用；
        - 副本自带 git 基线，可整体回滚（`git checkout -- .`）。
        """
        worktree = self.worktree_path(run_id)
        prepare_worktree(
            clone_path,
            worktree,
            exclude=set(self._settings.repository.excluded_dirs),
        )
        # 写入隔离的 pytest 配置
        (worktree / PYTEST_CONFIG_NAME).write_text(PYTEST_CONFIG, encoding="utf-8")
        # 在 agent 自己的测试目录里放一个收集过滤器（见 AGENT_CONFTEST 的说明）
        agent_tests_dir = worktree / AGENT_TEST_SUBDIR
        agent_tests_dir.mkdir(parents=True, exist_ok=True)
        (agent_tests_dir / "conftest.py").write_text(AGENT_CONFTEST, encoding="utf-8")
        return worktree

    def cleanup(self, run_id: str) -> None:
        """删除工作副本。"""
        remove_worktree(self.worktree_path(run_id))

    # -- 主循环 -----------------------------------------------------------
    async def run(
        self,
        context: CodeContext,
        clone_path: Path,
        *,
        max_attempts: int = 3,
        run_id: str | None = None,
        keep_worktree: bool = True,
    ) -> AutoFixResult:
        """执行完整的自动修复循环。

        Args:
            context: 待修复的代码上下文（来自 Step 3 检索或直接给出）。
            clone_path: 仓库的本地克隆路径（只读使用）。
            max_attempts: 最大修复轮次。
            keep_worktree: 结束后是否保留工作副本（保留便于查看 diff 与产物）。
        """
        run_id = run_id or uuid.uuid4().hex[:16]
        started = time.perf_counter()
        timings: dict[str, float] = {}
        iterations: list[LoopIteration] = []

        with trace_span(
            "planner.auto_fix",
            kind="agent",
            payload={
                "run_id": run_id,
                "target": context.display_name,
                "path": context.path,
                "max_attempts": max_attempts,
            },
        ) as span:
            # 1) 建立隔离工作副本
            prep_started = time.perf_counter()
            worktree = await _to_thread(self.prepare, clone_path, run_id)
            timings["prepare_ms"] = (time.perf_counter() - prep_started) * 1000

            # 2) 生成测试（此时仓库可用，必须导入真实模块而非内联替身）
            context.module_available = True
            gen_started = time.perf_counter()
            try:
                generated = await self._test_agent.generate(
                    context,
                    max_attempts=2,
                    run_id=run_id,
                    test_subdir=AGENT_TEST_SUBDIR,
                )
            except TestGenerationFailed as exc:
                return self._finish(
                    span, run_id, "error", False, iterations, context,
                    message=f"测试生成失败：{exc}", worktree=worktree,
                    started=started, timings=timings,
                )
            timings["generate_test_ms"] = (time.perf_counter() - gen_started) * 1000

            test_code = generated.test.code
            applied_patches: list[str] = []
            history: list[str] = []
            previous_signature: str | None = None
            # 只有"上一轮确实成功打了补丁"时，失败特征不变才叫无进展。
            # 补丁被拒绝时什么都没改，此时特征必然相同，不能误判为无进展
            # （否则会掩盖 patch_rejected 这个真实原因）。
            previous_patch_applied = False
            consecutive_rejects = 0
            final_test: SandboxResult | None = None
            status: LoopStatus = "max_attempts"
            message = ""

            # 3) 反馈循环
            for attempt in range(1, max_attempts + 1):
                iteration = LoopIteration(index=attempt)

                run_started = time.perf_counter()
                result = await self._sandbox.run(
                    SandboxRunRequest(
                        workspace=worktree,
                        test_target=AGENT_TEST_SUBDIR,
                        coverage_enabled=True,
                        extra_pytest_args=["-c", PYTEST_CONFIG_NAME],
                    )
                )
                timings[f"run_tests_{attempt}_ms"] = (
                    time.perf_counter() - run_started
                ) * 1000
                iteration.test_result = result
                final_test = result

                if result.succeeded:
                    iteration.note = "测试通过"
                    iterations.append(iteration)
                    status, message = "passed", "测试通过，无需修复"
                    break

                iterations.append(iteration)

                # 无进展检测：只有上一轮真的改了代码、失败特征却完全没变，
                # 才说明这条路走不通，提前终止而不是把剩余轮次白白烧掉。
                signature = failure_signature(result)
                if (
                    previous_patch_applied
                    and previous_signature is not None
                    and signature == previous_signature
                ):
                    status = "no_progress"
                    message = (
                        "补丁已应用但失败特征与上一轮完全一致，判定为无进展，提前终止"
                    )
                    iteration.note = message
                    break
                previous_signature = signature

                if attempt >= max_attempts:
                    status = "max_attempts"
                    message = f"已达最大修复轮次（{max_attempts}），测试仍未通过"
                    iteration.note = message
                    break

                # 4) 生成修复补丁
                proposal = await self._fix_agent.propose_fix(
                    FailureContext(
                        target=context,
                        test_code=test_code,
                        exit_code=result.exit_code,
                        stdout=result.stdout,
                        stderr=result.stderr,
                        passed=result.passed,
                        failed=result.failed,
                        errors=result.errors,
                        coverage_percent=(
                            result.coverage.percent_covered if result.coverage else None
                        ),
                        history=list(history),
                        attempt=attempt,
                    )
                )
                iteration.proposal = proposal

                if proposal.category not in PATCHABLE_CATEGORIES:
                    status = "not_patchable"
                    message = (
                        f"Fix Agent 归类为 {proposal.category}，按设计不出补丁。"
                        f"分析：{proposal.analysis[:200]}"
                    )
                    iteration.note = message
                    break

                if not proposal.has_patch:
                    status = "no_patch"
                    message = (
                        "模型未能给出可用的 unified diff："
                        + "；".join(proposal.warnings[:3])
                    )
                    iteration.note = message
                    break

                # 5) 应用补丁（先 --check 干跑，失败不留半成品）
                apply_result = apply_patch(worktree, proposal.patch)
                iteration.patch_applied = apply_result.applied
                iteration.patch_error = apply_result.error

                if not apply_result.applied:
                    consecutive_rejects += 1
                    previous_patch_applied = False
                    history.append(
                        f"第 {attempt} 轮补丁被拒绝：{apply_result.error}"
                    )
                    iteration.note = f"补丁未应用：{apply_result.error}"
                    if consecutive_rejects >= 2:
                        status = "patch_rejected"
                        message = f"补丁连续 {consecutive_rejects} 次无法应用，终止"
                        break
                    continue

                consecutive_rejects = 0
                previous_patch_applied = True
                applied_patches.append(proposal.patch)
                changed = changed_paths(worktree)
                history.append(
                    f"第 {attempt} 轮已应用补丁（{proposal.category}）："
                    f"改动 {', '.join(changed[:5]) or '未知'}"
                )
                iteration.note = f"已应用补丁，改动 {len(changed)} 个文件"
            else:  # pragma: no cover - 循环必然通过 break 退出
                status = "max_attempts"

            # 6) 汇总
            final_diff = ""
            files: list[str] = []
            if worktree.exists():
                final_diff = await _to_thread(worktree_diff, worktree)
                files = await _to_thread(changed_paths, worktree)

            if not keep_worktree:
                await _to_thread(self.cleanup, run_id)

            timings["total_ms"] = (time.perf_counter() - started) * 1000
            return self._finish(
                span,
                run_id,
                status,
                status == "passed",
                iterations,
                context,
                message=message,
                worktree=worktree if keep_worktree else None,
                started=started,
                timings=timings,
                test_code=test_code,
                final_test=final_test,
                applied_patch="\n".join(applied_patches),
                final_diff=final_diff,
                changed_files=files,
            )

    def _finish(
        self,
        span: object,
        run_id: str,
        status: LoopStatus,
        success: bool,
        iterations: list[LoopIteration],
        context: CodeContext,
        *,
        message: str,
        worktree: Path | None,
        started: float,
        timings: dict[str, float],
        test_code: str = "",
        final_test: SandboxResult | None = None,
        applied_patch: str = "",
        final_diff: str = "",
        changed_files: list[str] | None = None,
    ) -> AutoFixResult:
        """统一收尾并写 Trace。"""
        if "total_ms" not in timings:
            timings["total_ms"] = (time.perf_counter() - started) * 1000

        result = AutoFixResult(
            run_id=run_id,
            status=status,
            success=success,
            iterations=iterations,
            target=context,
            generated_test=test_code,
            final_test=final_test,
            applied_patch=applied_patch,
            final_diff=final_diff,
            changed_files=changed_files or [],
            worktree=str(worktree) if worktree else None,
            message=message,
            timings_ms={key: round(value, 1) for key, value in timings.items()},
            trace_id=current_trace_id(),
        )

        try:
            span.set_metadata(status=status, attempts=len(iterations), success=success)  # type: ignore[attr-defined]
            span.set_output(result.to_dict())  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - Trace 写入失败不应影响主流程
            logger.debug("写入 Trace 失败", exc_info=True)
        return result


async def _to_thread(func, *args):  # noqa: ANN001, ANN202
    """把同步阻塞操作挪出事件循环。"""
    import asyncio

    return await asyncio.to_thread(func, *args)


__all__ = [
    "AGENT_TEST_SUBDIR",
    "PYTEST_CONFIG",
    "PYTEST_CONFIG_NAME",
    "AutoFixPlanner",
    "AutoFixResult",
    "ContextError",
    "LoopIteration",
    "LoopStatus",
    "failure_signature",
]
