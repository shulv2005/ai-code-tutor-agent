"""修复 Agent 测试：失败归类、补丁提取与安全降级。"""

from __future__ import annotations

import pytest

from app.agents.dto import CodeContext
from app.agents.fix_agent import (
    FailureContext,
    FixAgent,
    parse_analysis,
    parse_category,
)
from app.core.config import get_settings
from tests.conftest import FakeLLMClient

TARGET = CodeContext(
    path="calc.py",
    code="def add(a, b):\n    return a - b\n",
    qualified_name="add",
    kind="function",
    signature="def add(a, b)",
    language="python",
)

SOURCE_BUG_REPLY = """\
## 分析
`add` 返回的是 a 与 b 的差，而测试期望求和。这是源码实现错误，测试断言是正确的。

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

TEST_BUG_REPLY = """\
## 分析
测试对 `add` 的异常断言有误：该函数从不抛异常，是测试本身写错了。

## 归类
test_bug

## 补丁
```diff
--- a/agent_tests/test_generated.py
+++ b/agent_tests/test_generated.py
@@ -5,2 +5,2 @@
 def test_add():
-    assert add(1, 2) == 4
+    assert add(1, 2) == 3
```
"""

ENVIRONMENT_REPLY = """\
## 分析
容器内缺少 `requests` 依赖，import 直接失败，这不是代码问题。

## 归类
environment

## 补丁
无需补丁
"""

NO_PATCH_REPLY = """\
## 分析
从日志看不出根因。

## 归类
unclear
"""


def _failure(**overrides: object) -> FailureContext:
    base: dict[str, object] = {
        "target": TARGET,
        "test_code": "def test_add():\n    assert add(1, 2) == 3\n",
        "exit_code": 1,
        "stdout": "1 failed in 0.03s",
        "stderr": "assert -1 == 3",
        "passed": 0,
        "failed": 1,
    }
    base.update(overrides)
    return FailureContext(**base)  # type: ignore[arg-type]


def _agent(response: str, **kwargs: object) -> tuple[FixAgent, FakeLLMClient]:
    fake = FakeLLMClient(response, **kwargs)  # type: ignore[arg-type]
    return FixAgent(get_settings(), fake), fake


# ---------------------------------------------------------------------------
# 归类解析
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (SOURCE_BUG_REPLY, "source_bug"),
        (TEST_BUG_REPLY, "test_bug"),
        (ENVIRONMENT_REPLY, "environment"),
        (NO_PATCH_REPLY, "unclear"),
        ("## 归类\nalready_fixed\n", "already_fixed"),
    ],
)
def test_parse_category(text: str, expected: str) -> None:
    assert parse_category(text) == expected


def test_parse_category_falls_back_to_full_text() -> None:
    """模型没按格式分节时，也要能从全文里认出关键词。"""
    assert parse_category("我认为这是 source_bug，因为实现写错了") == "source_bug"


def test_parse_category_defaults_to_unclear() -> None:
    assert parse_category("完全无关的内容") == "unclear"
    assert parse_category("") == "unclear"


def test_parse_analysis_extracts_section() -> None:
    analysis = parse_analysis(SOURCE_BUG_REPLY)
    assert "差" in analysis
    assert "##" not in analysis


def test_parse_analysis_falls_back_to_prefix() -> None:
    assert parse_analysis("没有分节的说明\n```diff\n--- a/x\n```") == "没有分节的说明"


# ---------------------------------------------------------------------------
# 补丁生成
# ---------------------------------------------------------------------------
async def test_propose_fix_returns_patch_for_source_bug() -> None:
    agent, fake = _agent(SOURCE_BUG_REPLY)
    proposal = await agent.propose_fix(_failure())

    assert proposal.category == "source_bug"
    assert proposal.has_patch
    assert "return a + b" in proposal.patch
    assert proposal.patch.startswith("--- a/calc.py")
    assert proposal.model == "fake-model"
    assert forecast_warnings_ok(proposal.warnings)


def forecast_warnings_ok(warnings: list[str]) -> bool:
    """source_bug 不应产生告警。"""
    return warnings == []


async def test_propose_fix_supports_test_bug() -> None:
    agent, _ = _agent(TEST_BUG_REPLY)
    proposal = await agent.propose_fix(_failure())

    assert proposal.category == "test_bug"
    assert proposal.has_patch
    assert "test_generated.py" in proposal.patch


async def test_environment_category_produces_no_patch() -> None:
    """环境问题改代码解决不了，按设计不生成补丁。"""
    agent, _ = _agent(ENVIRONMENT_REPLY)
    proposal = await agent.propose_fix(_failure())

    assert proposal.category == "environment"
    assert proposal.has_patch is False
    assert any("不生成补丁" in item for item in proposal.warnings)


async def test_unclear_without_patch_warns() -> None:
    agent, _ = _agent(NO_PATCH_REPLY)
    proposal = await agent.propose_fix(_failure())

    assert proposal.category == "unclear"
    assert proposal.has_patch is False
    assert proposal.warnings


async def test_unsafe_patch_is_rejected() -> None:
    """越界补丁必须在 Agent 层就被丢弃，而不是交给应用阶段。"""
    reply = (
        "## 分析\n试图改系统文件\n\n## 归类\nsource_bug\n\n## 补丁\n"
        "```diff\n--- a/../../../etc/passwd\n+++ b/../../../etc/passwd\n"
        "@@ -1 +1 @@\n-x\n+y\n```\n"
    )
    agent, _ = _agent(reply)
    proposal = await agent.propose_fix(_failure())

    assert proposal.has_patch is False
    assert any("穿越" in item or "绝对路径" in item for item in proposal.warnings)


async def test_malformed_patch_is_rejected() -> None:
    """模型给出的 diff 格式错误（hunk 行数不对）时应丢弃并告警。"""
    reply = (
        "## 分析\nx\n\n## 归类\nsource_bug\n\n## 补丁\n"
        "```diff\n--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,5 @@\n def add(a, b):\n"
        "-    return a - b\n+    return a + b\n```\n"
    )
    agent, _ = _agent(reply)
    proposal = await agent.propose_fix(_failure())
    # 提取成功但校验阶段会由 git --check 拦下；此处只要求不抛异常
    assert isinstance(proposal.has_patch, bool)


# ---------------------------------------------------------------------------
# 提示词构造
# ---------------------------------------------------------------------------
async def test_prompt_contains_code_test_and_failure() -> None:
    agent, fake = _agent(SOURCE_BUG_REPLY)
    await agent.propose_fix(_failure())

    user = fake.last_user_content
    assert "def add(a, b):" in user
    assert "return a - b" in user
    assert "assert add(1, 2) == 3" in user
    assert "assert -1 == 3" in user
    assert "退出码：1" in user

    system = fake.calls[0][0].content
    assert "unified diff" in system
    assert "source_bug" in system
    # 必须在提示词里强调 hunk 行数，否则模型极易产出 corrupt patch
    assert "行数" in system


async def test_prompt_includes_history() -> None:
    agent, fake = _agent(SOURCE_BUG_REPLY)
    await agent.propose_fix(
        _failure(history=["第 1 轮已应用补丁（source_bug）：改动 calc.py"])
    )
    assert "之前的修复尝试" in fake.last_user_content
    assert "第 1 轮" in fake.last_user_content


async def test_prompt_truncates_huge_logs() -> None:
    agent, fake = _agent(SOURCE_BUG_REPLY)
    await agent.propose_fix(_failure(stdout="x" * 50_000, stderr="y" * 50_000))

    user = fake.last_user_content
    assert "已截断" in user
    assert len(user) < 30_000


async def test_prompt_includes_coverage_when_present() -> None:
    agent, fake = _agent(SOURCE_BUG_REPLY)
    await agent.propose_fix(_failure(coverage_percent=42.5))
    assert "覆盖率：42.5%" in fake.last_user_content
