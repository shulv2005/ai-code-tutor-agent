"""测试生成 Agent：读取代码上下文 → 调用 LLM → 提取 pytest 测试 → 落盘沙箱。

在整条工作流中的位置：
    Step 3 检索结果（CodeContext）
        -> 本 Agent 构造 Prompt 调 LLM
        -> 提取并校验纯 Python 代码
        -> 写入沙箱目录，交给 Step 5 执行

设计要点：
1. **生成-校验-重试闭环**：提取出的代码先做静态校验（能否解析、是否有 test_ 函数、
   pytest 是否导入），不合格则带着具体问题重试，最多 max_attempts 次。
   这把"模型偶尔输出残缺代码"从链路故障降级为一次重试。
2. **与 LLM 解耦**：只依赖 `LLMClient` 协议，测试可注入假客户端，
   无需真实模型即可验证整条链路。
3. **Trace 贯穿**：整个生成过程是一个 span，每次 LLM 调用是子 span，
   记录模型、token、重试次数与耗时。
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path

from app.agents.dto import CodeContext, GeneratedTest, TestGenerationResult
from app.agents.extraction import check_generated_test, extract_python_code, python_test_functions
from app.agents.prompts import PROMPT_VERSION, build_test_generation_messages
from app.core.config import Settings
from app.core.llm_client import LLMClient, LLMError
from app.core.trace import current_trace_id, trace_span

logger = logging.getLogger(__name__)

# 沙箱内的固定落盘位置（Step 5 直接从这里跑 pytest）
SANDBOX_DIR_NAME = "sandbox_repo"
GENERATED_TEST_FILENAME = "test_generated.py"
TEST_SUBDIR = "tests"


class TestAgentError(RuntimeError):
    """测试生成失败。"""

    # 告诉 pytest 不要把这个类当作测试类收集（类名以 Test 开头会被误判）
    __test__ = False


class TestGenerationFailed(TestAgentError):
    """重试耗尽仍无法得到可用的测试代码。"""

    __test__ = False


def _new_run_id() -> str:
    """生成一次生成任务的运行 ID（也是沙箱目录名）。"""
    return uuid.uuid4().hex[:16]


class TestAgent:
    """测试生成 Agent。"""

    # 类名以 Test 开头，显式告知 pytest 这不是测试类
    __test__ = False

    def __init__(self, settings: Settings, llm: LLMClient) -> None:
        self._settings = settings
        self._llm = llm

    @property
    def llm(self) -> LLMClient:
        return self._llm

    # -- 沙箱路径 ---------------------------------------------------------
    def sandbox_root(self, run_id: str) -> Path:
        """本次运行的沙箱仓库根目录。"""
        return self._settings.docker.workspace_path / run_id / SANDBOX_DIR_NAME

    def sandbox_test_path(self, run_id: str, test_subdir: str | None = None) -> Path:
        """生成测试的落盘路径：<sandbox_repo>/<test_subdir>/test_generated.py。

        test_subdir 可覆盖：修复循环会把它设成独立目录（如 agent_tests），
        避免与仓库自带的 tests/ 混在一起而被一并收集执行。
        """
        subdir = test_subdir or self._settings.docker.test_dir
        return self.sandbox_root(run_id) / subdir / GENERATED_TEST_FILENAME

    # -- 主流程 -----------------------------------------------------------
    async def generate(
        self,
        context: CodeContext,
        *,
        max_attempts: int = 2,
        save_to_sandbox: bool = True,
        run_id: str | None = None,
        test_subdir: str | None = None,
    ) -> TestGenerationResult:
        """为给定代码上下文生成 pytest 测试。

        Args:
            max_attempts: 含首次在内的最大尝试次数；校验失败会带反馈重试。
            save_to_sandbox: 是否把结果写入沙箱目录。
            run_id: 指定运行 ID（测试用），缺省随机生成。
            test_subdir: 生成测试所在子目录，缺省用 DOCKER__TEST_DIR。

        Raises:
            TestGenerationFailed: 重试耗尽仍未得到通过校验的代码。
            LLMError: LLM 调用本身失败（配置、网络、鉴权）。
        """
        run_id = run_id or _new_run_id()
        timings: dict[str, float] = {}
        started = time.perf_counter()

        with trace_span(
            "agent.test_agent.generate",
            kind="agent",
            payload={
                "run_id": run_id,
                "target": context.display_name,
                "path": context.path,
                "prompt_version": PROMPT_VERSION,
                "max_attempts": max_attempts,
            },
            metadata={"model": context.language, "symbol_id": context.symbol_id},
        ) as span:
            feedback: str | None = None
            previous_output: str | None = None
            last_code = ""
            last_warnings: list[str] = []
            usage_total = 0
            completion_total = 0
            attempts_used = 0
            model_name = self._llm.model

            for attempt in range(1, max_attempts + 1):
                attempts_used = attempt
                messages = build_test_generation_messages(
                    context, retry_feedback=feedback, previous_attempt=previous_output
                )

                attempt_started = time.perf_counter()
                response = await self._llm.chat(messages)
                timings[f"llm_attempt_{attempt}_ms"] = (
                    time.perf_counter() - attempt_started
                ) * 1000

                model_name = response.model or model_name
                usage_total += response.usage.prompt_tokens
                completion_total += response.usage.completion_tokens
                previous_output = response.content

                code, warnings = extract_python_code(response.content)
                if response.truncated:
                    # 被 max_tokens 截断是"代码不完整"的常见原因，明确告知模型
                    warnings.append(
                        "模型输出被 max_tokens 截断，代码可能不完整"
                        "（请精简测试或调大 LLM__MAX_TOKENS）"
                    )

                problems = check_generated_test(code)
                if problems:
                    feedback = "；".join(problems)
                    logger.info(
                        "第 %d/%d 次生成未通过校验: %s", attempt, max_attempts, feedback
                    )
                    last_code = code
                    last_warnings = warnings + problems
                    if attempt < max_attempts:
                        continue
                    span.set_metadata(failed_after_attempts=attempt)
                    raise TestGenerationFailed(
                        f"重试 {attempt} 次仍未生成可用的测试代码：{feedback}"
                    )

                # 通过校验
                last_code = code
                last_warnings = warnings
                break

            test = GeneratedTest(
                code=last_code,
                model=model_name,
                attempts=attempts_used,
                usage=_usage(usage_total, completion_total),
                warnings=last_warnings,
                test_functions=python_test_functions(last_code),
                latency_ms=(time.perf_counter() - started) * 1000,
            )

            saved_path: Path | None = None
            if save_to_sandbox:
                save_started = time.perf_counter()
                saved_path = await asyncio.to_thread(
                    self._write_sandbox, run_id, last_code, test_subdir
                )
                timings["save_ms"] = (time.perf_counter() - save_started) * 1000

            timings["total_ms"] = (time.perf_counter() - started) * 1000
            span.set_metadata(
                attempts=attempts_used,
                test_functions=len(test.test_functions),
                saved=bool(saved_path),
            )
            span.set_output(
                {
                    "run_id": run_id,
                    "code_chars": len(last_code),
                    "tests": test.test_functions,
                }
            )

            return TestGenerationResult(
                run_id=run_id,
                target=context,
                test=test,
                saved_path=saved_path,
                timings_ms={key: round(value, 3) for key, value in timings.items()},
                trace_id=current_trace_id(),
            )

    # -- 落盘 -------------------------------------------------------------
    def _write_sandbox(self, run_id: str, code: str, test_subdir: str | None = None) -> Path:
        """把生成的测试写入沙箱目录。"""
        target = self.sandbox_test_path(run_id, test_subdir)
        target.parent.mkdir(parents=True, exist_ok=True)

        header = (
            '"""由 AI 测试生成 Agent 自动生成，请勿手工修改。\n\n'
            f"prompt_version: {PROMPT_VERSION}\n"
            '"""\n\n'
        )
        target.write_text(header + code.rstrip() + "\n", encoding="utf-8")

        # 让 tests 目录可被 pytest 正确收集（避免与仓库其它同名模块冲突）
        init_file = target.parent / "__init__.py"
        if not init_file.exists():
            init_file.write_text("", encoding="utf-8")

        logger.info("已写入生成测试: %s", target)
        return target


def _usage(prompt_tokens: int, completion_tokens: int):  # noqa: ANN201
    """组装 token 用量（延迟导入避免循环）。"""
    from app.core.llm_client import LLMUsage

    return LLMUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


__all__ = [
    "GENERATED_TEST_FILENAME",
    "SANDBOX_DIR_NAME",
    "TEST_SUBDIR",
    "TestAgent",
    "TestAgentError",
    "TestGenerationFailed",
    "LLMError",
]
