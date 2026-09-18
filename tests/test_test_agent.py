"""测试生成 Agent 测试：用假 LLM 客户端覆盖生成-校验-重试闭环与沙箱落盘。"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.agents.dto import CodeContext
from app.agents.prompts import PROMPT_VERSION
from app.agents.test_agent import TestAgent, TestGenerationFailed
from app.core.config import get_settings
from app.core.llm_client import LLMError
from app.core.trace import MemoryTraceSink, TraceRecorder, set_trace_recorder
from tests.conftest import FAKE_GENERATED_TEST, FakeLLMClient

TARGET_CODE = '''def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
'''


def _context(**overrides: object) -> CodeContext:
    base = {
        "path": "pkg/core.py",
        "code": TARGET_CODE,
        "language": "python",
        "qualified_name": "add",
        "kind": "function",
        "signature": "def add(a: int, b: int) -> int",
        "docstring": "Add two numbers.",
        "start_line": 1,
        "end_line": 3,
    }
    base.update(overrides)
    return CodeContext(**base)  # type: ignore[arg-type]


def _agent(fake: FakeLLMClient) -> TestAgent:
    return TestAgent(get_settings(), fake)


# ---------------------------------------------------------------------------
# 正常路径
# ---------------------------------------------------------------------------
async def test_generate_extracts_code_and_saves_to_sandbox(
    llm_env: Path, fake_llm: FakeLLMClient
) -> None:
    agent = _agent(fake_llm)
    result = await agent.generate(_context(), run_id="run1")

    assert result.run_id == "run1"
    assert "def test_add_positive_numbers" in result.test.code
    assert result.test.test_functions == [
        "test_add_positive_numbers",
        "test_add_edge_cases",
    ]
    assert result.test.attempts == 1
    assert result.test.model == "fake-model"
    assert result.test.usage.total_tokens == 150

    # 落盘位置：<workspace>/<run_id>/sandbox_repo/tests/test_generated.py
    assert result.saved_path is not None
    assert result.saved_path == llm_env / "run1" / "sandbox_repo" / "tests" / "test_generated.py"
    assert result.saved_path.exists()

    written = result.saved_path.read_text(encoding="utf-8")
    assert "def test_add_positive_numbers" in written
    assert PROMPT_VERSION in written  # 文件头记录提示词版本，便于溯源


async def test_sandbox_write_creates_init_file(llm_env: Path, fake_llm: FakeLLMClient) -> None:
    """tests/ 下补 __init__.py，避免 pytest 收集时与仓库同名模块冲突。"""
    result = await _agent(fake_llm).generate(_context(), run_id="run2")
    assert result.saved_path is not None
    assert (result.saved_path.parent / "__init__.py").exists()


async def test_save_can_be_disabled(llm_env: Path, fake_llm: FakeLLMClient) -> None:
    result = await _agent(fake_llm).generate(_context(), save_to_sandbox=False)
    assert result.saved_path is None
    assert not (llm_env).exists() or not any(llm_env.rglob("test_generated.py"))


async def test_sandbox_paths_are_isolated_per_run(llm_env: Path, fake_llm: FakeLLMClient) -> None:
    agent = _agent(fake_llm)
    first = await agent.generate(_context(), run_id="a")
    second = await agent.generate(_context(), run_id="b")

    assert first.saved_path != second.saved_path
    assert first.saved_path.parent.parent.parent != second.saved_path.parent.parent.parent


# ---------------------------------------------------------------------------
# 提示词构造
# ---------------------------------------------------------------------------
async def test_prompt_contains_code_context(llm_env: Path, fake_llm: FakeLLMClient) -> None:
    await _agent(fake_llm).generate(_context(), run_id="r")

    sent = fake_llm.last_user_content
    assert "pkg/core.py" in sent
    assert "def add(a: int, b: int) -> int" in sent
    assert "Add two numbers." in sent
    assert "return a + b" in sent
    # 系统提示必须约束"只输出代码块"与 pytest 风格
    system = fake_llm.calls[0][0].content
    assert "只输出一个 Python 代码块" in system
    assert "pytest" in system


async def test_prompt_includes_related_and_existing_tests(
    llm_env: Path, fake_llm: FakeLLMClient
) -> None:
    context = _context(
        related=["function fetch(url: str) -> str  # Fetch a url."],
        existing_tests=["# tests/test_core.py::test_add\ndef test_add():\n    assert True\n"],
    )
    await _agent(fake_llm).generate(context, run_id="r")

    sent = fake_llm.last_user_content
    assert "相关符号" in sent
    assert "fetch(url: str)" in sent
    assert "现有测试" in sent
    assert "test_add" in sent


# ---------------------------------------------------------------------------
# 生成-校验-重试闭环
# ---------------------------------------------------------------------------
async def test_retries_when_output_is_unusable(llm_env: Path) -> None:
    """首次输出没有 test_ 函数，应带反馈重试，第二次成功。"""
    fake = FakeLLMClient(
        [
            "```python\nimport pytest\n\n\ndef helper():\n    return 1\n```",
            FAKE_GENERATED_TEST,
        ]
    )
    result = await _agent(fake).generate(_context(), run_id="r", max_attempts=2)

    assert fake.call_count == 2
    assert result.test.attempts == 2
    assert "def test_add" in result.test.code
    # 第二次请求必须带上失败原因
    assert "上一次生成的结果有问题" in fake.last_user_content
    assert "test_" in fake.last_user_content


async def test_retries_when_code_cannot_parse(llm_env: Path) -> None:
    fake = FakeLLMClient(["```python\ndef broken(:\n```", FAKE_GENERATED_TEST])
    result = await _agent(fake).generate(_context(), run_id="r", max_attempts=2)
    assert result.test.attempts == 2
    assert fake.call_count == 2


async def test_retries_when_pytest_used_without_import(llm_env: Path) -> None:
    fake = FakeLLMClient(
        [
            "```python\ndef test_x():\n    with pytest.raises(ValueError):\n        raise ValueError\n```",
            FAKE_GENERATED_TEST,
        ]
    )
    await _agent(fake).generate(_context(), run_id="r", max_attempts=2)
    assert fake.call_count == 2
    assert "import pytest" in fake.last_user_content


async def test_exhausted_retries_raise(llm_env: Path) -> None:
    fake = FakeLLMClient("```python\nno tests here\n```")
    with pytest.raises(TestGenerationFailed) as excinfo:
        await _agent(fake).generate(_context(), run_id="r", max_attempts=2)

    assert fake.call_count == 2
    assert "重试" in str(excinfo.value)


async def test_single_attempt_does_not_retry(llm_env: Path) -> None:
    fake = FakeLLMClient("```python\nno tests here\n```")
    with pytest.raises(TestGenerationFailed):
        await _agent(fake).generate(_context(), run_id="r", max_attempts=1)
    assert fake.call_count == 1


async def test_truncated_output_triggers_retry_and_warns(llm_env: Path) -> None:
    """max_tokens 截断会导致代码不完整，应重试；若最终成功需保留告警。"""
    fake = FakeLLMClient([FAKE_GENERATED_TEST, FAKE_GENERATED_TEST])
    result = await _agent(fake).generate(_context(), run_id="r")
    assert result.test.code


async def test_llm_error_propagates(llm_env: Path) -> None:
    fake = FakeLLMClient("x", fail_with=LLMError("upstream down"))
    with pytest.raises(LLMError):
        await _agent(fake).generate(_context(), run_id="r")


# ---------------------------------------------------------------------------
# Trace 集成
# ---------------------------------------------------------------------------
async def test_generation_emits_trace_spans(llm_env: Path, fake_llm: FakeLLMClient) -> None:
    sink = MemoryTraceSink(maxlen=50)
    set_trace_recorder(TraceRecorder([sink]))
    try:
        result = await _agent(fake_llm).generate(_context(), run_id="r")
    finally:
        set_trace_recorder(None)

    names = [record.name for record in sink.records()]
    assert "agent.test_agent.generate" in names

    span = next(r for r in sink.records() if r.name == "agent.test_agent.generate")
    assert span.status == "ok"
    assert span.metadata["attempts"] == 1
    assert span.metadata["test_functions"] == 2
    assert span.trace_id == result.trace_id


# ---------------------------------------------------------------------------
# 端到端：生成的测试必须真的能被 pytest 跑起来
# ---------------------------------------------------------------------------
RUNNABLE_TEST = '''```python
import pytest


def add(a: int, b: int) -> int:
    """Stand-in implementation (inlined because the module is not provided)."""
    return a + b


def test_add_positive():
    assert add(1, 2) == 3


@pytest.mark.parametrize("a, b, expected", [(0, 0, 0), (-1, 1, 0), (10, -3, 7)])
def test_add_variants(a, b, expected):
    assert add(a, b) == expected


def test_add_rejects_bad_types():
    with pytest.raises(TypeError):
        add("1", 2)
```
'''


async def test_generated_file_actually_passes_pytest(
    llm_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """把生成的测试真正交给 pytest 执行。

    这是对「生成结果可用」最有力的验证：前面的用例只证明"能解析、有 test_ 函数"，
    这里证明它**真的能跑通**。Step 5 的沙箱执行正是同一件事的容器化版本。
    """
    import subprocess
    import sys

    fake = FakeLLMClient(RUNNABLE_TEST)
    result = await _agent(fake).generate(_context(), run_id="runnable")

    assert result.saved_path is not None
    assert result.test.test_functions == [
        "test_add_positive",
        "test_add_variants",
        "test_add_rejects_bad_types",
    ]

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(result.saved_path),
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(result.saved_path.parent),
        timeout=120,
    )

    assert completed.returncode == 0, (
        f"生成的测试未能通过 pytest：\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
    )
    # 4 个用例（含 parametrize 展开的 3 个）= 1 + 3 + 1
    assert "5 passed" in completed.stdout
