"""修复 Agent：分析测试失败原因，生成 unified diff 补丁。

在整条工作流中的位置：
    Step 5 沙箱执行失败（stdout / stderr / exit_code）
        -> 本 Agent 分析根因
        -> 输出 git diff 补丁
        -> Step 7 Planner 应用到隔离工作副本并重跑

关键设计：**先分类，再出补丁**。

失败未必是源码有 bug，也可能是：
- 测试本身写错了（断言不成立、导入路径不对）—— 应该修测试
- 环境/依赖缺失 —— 不该出补丁，应直接上报
- 被测代码行为其实正确、是模型误解了语义 —— 应该修测试

如果不做分类就让模型"生成修复补丁"，它会倾向于去改源码迁就一个错误的测试，
把好代码改坏。因此提示词要求先判定 category，且 `environment` 类直接不出补丁。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Literal

from app.agents.dto import CodeContext
from app.agents.patch import extract_patch, validate_patch
from app.core.config import Settings
from app.core.llm_client import LLMClient, LLMUsage
from app.core.trace import trace_span

logger = logging.getLogger(__name__)

# 失败归类
FailureCategory = Literal["source_bug", "test_bug", "environment", "unclear", "already_fixed"]

# 需要产出补丁的类别；environment/already_fixed 不出补丁
PATCHABLE_CATEGORIES: frozenset[str] = frozenset({"source_bug", "test_bug", "unclear"})

FIX_PROMPT_VERSION = "fix-agent/v1"

SYSTEM_PROMPT = """你是一位资深的 Python 调试与修复专家。

你会收到：一段被测代码、一份针对它生成的 pytest 测试、以及测试执行失败的输出。
你的任务是**先判断失败根因，再决定是否需要修改代码**。

请按以下格式输出（严格遵守）：

## 分析
<用 2-4 句话说明失败的直接原因与根本原因。>

## 归类
<从以下五个中选一个，只写这个英文单词：>
- source_bug：被测源码确实有 bug，测试是对的
- test_bug：测试本身写错了（断言不对、导入路径错、误解了函数语义、遗漏了前置条件）
- environment：环境问题（缺依赖、缺文件、版本不兼容），改代码解决不了
- already_fixed：代码其实已经满足测试意图，无需改动
- unclear：信息不足以判断

## 补丁
<如果需要修改代码，输出一个 unified diff 代码块；
 如果归类为 environment 或 already_fixed，写"无需补丁"。>

输出补丁的严格要求：
1. 必须是标准 unified diff，格式如下（注意 hunk 头里的行数必须**精确**）：
   ```diff
   --- a/path/to/file.py
   +++ b/path/to/file.py
   @@ -10,7 +10,9 @@
    上下文行
   -被删除的行
   +新增的行
   ```
2. 路径必须是**相对仓库根目录**的相对路径，且必须与上方给出的文件路径一致。
3. 每个 hunk 的 `@@ -原起始,原行数 +新起始,新行数 @@` 行数必须与实际内容严格相符，
   否则补丁会因 "corrupt patch" 被拒绝。
4. 上下文行必须逐字符匹配现有代码（含缩进），建议上下各保留 3 行上下文。
5. 只修改必要内容，不要顺手重构、不要改格式、不要动无关文件。
6. 不要修改 .git 目录，不要使用绝对路径，不要使用 `..`。
"""

USER_TEMPLATE = """## 被测代码
- 文件路径（相对仓库根目录）：{path}
- 符号：{qualified_name}（{kind}）
- 语言：{language}

```python
{code}
```

## 生成的测试
```python
{test_code}
```

## 执行结果
- 退出码：{exit_code}
- 用例统计：通过 {passed}，失败 {failed}，错误 {errors}
{coverage_line}

### stdout
```
{stdout}
```

### stderr
```
{stderr}
```
{history}
请分析失败原因，按要求格式输出。
"""


@dataclass(slots=True)
class FailureContext:
    """一次失败的完整上下文。"""

    target: CodeContext
    test_code: str
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    passed: int = 0
    failed: int = 0
    errors: int = 0
    coverage_percent: float | None = None
    # 前几轮的修复历史（补丁 + 结果），避免模型重复给同样的补丁
    history: list[str] = field(default_factory=list)
    attempt: int = 1


@dataclass(slots=True)
class FixProposal:
    """修复建议。"""

    category: FailureCategory
    analysis: str
    patch: str = ""
    warnings: list[str] = field(default_factory=list)
    model: str = ""
    usage: LLMUsage = field(default_factory=LLMUsage)
    latency_ms: float = 0.0

    @property
    def has_patch(self) -> bool:
        return bool(self.patch.strip())


class FixAgentError(RuntimeError):
    """修复 Agent 失败。"""

    __test__ = False


class FixAgent:
    """修复 Agent。"""

    __test__ = False

    def __init__(self, settings: Settings, llm: LLMClient) -> None:
        self._settings = settings
        self._llm = llm

    @property
    def llm(self) -> LLMClient:
        return self._llm

    def build_messages(self, context: FailureContext) -> list[object]:
        """构造对话消息（独立出来便于测试断言提示词内容）。"""
        from app.core.llm_client import LLMMessage

        target = context.target
        coverage_line = (
            f"- 覆盖率：{context.coverage_percent:.1f}%\n"
            if context.coverage_percent is not None
            else ""
        )

        history_block = ""
        if context.history:
            history_block = "\n## 之前的修复尝试（均已失败，请勿重复同样的思路）\n"
            history_block += "\n".join(f"- {item}" for item in context.history[-5:])
            history_block += "\n"

        user = USER_TEMPLATE.format(
            path=target.path,
            qualified_name=target.display_name,
            kind=target.kind,
            language=target.language,
            code=target.code.rstrip(),
            test_code=(context.test_code or "(无)").rstrip(),
            exit_code=context.exit_code,
            passed=context.passed,
            failed=context.failed,
            errors=context.errors,
            coverage_line=coverage_line,
            stdout=_clip(context.stdout, 6000),
            stderr=_clip(context.stderr, 4000),
            history=history_block,
        )
        return [
            LLMMessage(role="system", content=SYSTEM_PROMPT),
            LLMMessage(role="user", content=user),
        ]

    async def propose_fix(self, context: FailureContext) -> FixProposal:
        """分析失败并给出修复建议。

        Raises:
            FixAgentError: 模型既没给出可解析的归类，也没给出补丁。
        """
        started = time.perf_counter()

        with trace_span(
            "agent.fix_agent.propose_fix",
            kind="agent",
            payload={
                "target": context.target.display_name,
                "path": context.target.path,
                "attempt": context.attempt,
                "prompt_version": FIX_PROMPT_VERSION,
            },
            metadata={"failed": context.failed, "errors": context.errors},
        ) as span:
            messages = self.build_messages(context)
            response = await self._llm.chat(messages)  # type: ignore[arg-type]

            category = parse_category(response.content)
            analysis = parse_analysis(response.content)

            warnings: list[str] = []
            patch = ""
            if category in PATCHABLE_CATEGORIES:
                try:
                    patch, warnings = extract_patch(response.content)
                except Exception as exc:  # noqa: BLE001 - 提取失败降级为"无补丁"
                    warnings.append(f"补丁提取失败：{type(exc).__name__}: {exc}")
                    patch = ""

                if patch:
                    problems = validate_patch(patch)
                    if problems:
                        warnings.extend(problems)
                        logger.info("补丁未通过校验: %s", problems)
                        patch = ""
                else:
                    warnings.append("模型未返回可用的 unified diff")
            else:
                warnings.append(f"归类为 {category}，按设计不生成补丁")

            proposal = FixProposal(
                category=category,
                analysis=analysis,
                patch=patch,
                warnings=warnings,
                model=response.model,
                usage=response.usage,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

            span.set_metadata(category=category, has_patch=proposal.has_patch)
            span.set_output(
                {
                    "category": category,
                    "patch_chars": len(patch),
                    "warnings": warnings,
                }
            )
            return proposal


def _clip(text: str, limit: int) -> str:
    """截断日志，保留头尾（报错常在末尾，上下文常在开头）。"""
    if not text:
        return "(空)"
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-(limit // 2) :]
    return f"{head}\n...<已截断 {len(text) - limit} 字符>...\n{tail}"


_CATEGORY_PATTERN = None


def parse_category(text: str) -> FailureCategory:
    """从模型输出中解析失败归类。

    优先读 "## 归类" 段落；找不到时全文本匹配关键词。
    解析不出来时返回 unclear（而不是抛异常）——由调用方决定是否继续。
    """
    import re

    known = ("source_bug", "test_bug", "environment", "already_fixed", "unclear")

    # 1) 定位"归类"段落之后的第一行内容
    section = re.search(
        r"#{1,3}\s*归类\s*\n+(?P<body>.{0,400})", text, re.DOTALL
    )
    if section:
        body = section.group("body")
        for name in known:
            if name in body:
                return name  # type: ignore[return-value]

    # 2) 退化：全文找关键词，按出现顺序取第一个
    lowered = text.lower()
    positions = [(lowered.find(name), name) for name in known if name in lowered]
    if positions:
        positions.sort()
        return positions[0][1]  # type: ignore[return-value]
    return "unclear"


def parse_analysis(text: str) -> str:
    """从模型输出中解析分析段落。"""
    import re

    section = re.search(r"#{1,3}\s*分析\s*\n+(?P<body>.*?)(?=\n#{1,3}\s|\Z)", text, re.DOTALL)
    if section:
        return section.group("body").strip()[:2000]
    # 没有分节时，取补丁之前的所有文字
    before = text.split("```")[0].strip()
    return before[:2000] if before else "(模型未给出分析)"


__all__ = [
    "FIX_PROMPT_VERSION",
    "PATCHABLE_CATEGORIES",
    "FailureCategory",
    "FailureContext",
    "FixAgent",
    "FixAgentError",
    "FixProposal",
    "parse_analysis",
    "parse_category",
]
