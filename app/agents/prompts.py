"""提示词模板：集中管理，便于单独迭代与回归对比。

把提示词从 Agent 逻辑里抽出来，是为了能在不改代码的前提下调整措辞，
并让"提示词版本"成为可追踪的东西（Step 7 记录每轮生成用的是哪版提示词）。
"""

from __future__ import annotations

from app.agents.dto import CodeContext
from app.core.llm_client import LLMMessage

# 提示词版本号：改动措辞时递增，便于回溯生成质量变化
PROMPT_VERSION = "test-agent/v2"

SYSTEM_PROMPT = """你是一位资深的 Python 测试工程师，精通 pytest。

你的任务：为给定的代码编写高质量、**可以直接运行**的 pytest 测试。

严格遵守以下要求：
1. 只输出一个 Python 代码块，不要任何解释、总结或前后缀文字。
2. 完整可运行：包含所有必要的 import 语句。
3. 使用 pytest 原生风格：`assert` 断言、`pytest.raises` 断言异常、
   `pytest.mark.parametrize` 覆盖多组输入。
4. 测试必须**确定性**：禁止依赖网络、当前时间、随机数、文件系统或环境变量。
5. 测试必须**自包含**：若依赖其它未提供的函数/类，
   请在测试文件内用等价的简单替身（stub）替代，并在注释中说明。
6. 不要 mock 被测函数本身；要测它的真实行为。
7. 覆盖三类场景：正常路径、边界条件（空值/极值/长度 0 或 1）、异常路径。
8. 测试函数命名清晰：`test_<被测函数>_<场景>`。
9. 不要写 `if __name__ == "__main__"` 块，不要写 `print`。
"""

# 当沙箱内存在完整仓库时，必须改为导入真实模块。
# 否则测试内联了一份实现副本，后续对源码打补丁不会影响测试结果，
# 反馈闭环会退化成"永远修不好"。
IMPORT_SYSTEM_PROMPT = """你是一位资深的 Python 测试工程师，精通 pytest。

你的任务：为给定的代码编写高质量、**可以直接运行**的 pytest 测试。

测试将在一个包含完整仓库副本的目录中运行，因此**必须导入真实实现**：

严格遵守以下要求：
1. 只输出一个 Python 代码块，不要任何解释、总结或前后缀文字。
2. **必须从真实模块导入被测对象**（导入语句会在下方给出），
   **绝对不要**在测试文件里重新定义/内联一份被测函数的副本。
3. 完整可运行：包含所有必要的 import 语句。
4. 使用 pytest 原生风格：`assert` 断言、`pytest.raises` 断言异常、
   `pytest.mark.parametrize` 覆盖多组输入。
5. 测试必须**确定性**：禁止依赖网络、当前时间、随机数、环境变量。
   若被测代码需要文件，请用 pytest 的 `tmp_path` 夹具。
6. 不要 mock 被测函数本身；要测它的真实行为。
7. 覆盖三类场景：正常路径、边界条件（空值/极值/长度 0 或 1）、异常路径。
8. 测试函数命名清晰：`test_<被测函数>_<场景>`。
9. 不要写 `if __name__ == "__main__"` 块，不要写 `print`。
"""

RETRY_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + "\n10. 上一次生成的结果有问题，本次务必修正，只输出修正后的完整代码块。"
)

IMPORT_RETRY_SYSTEM_PROMPT = (
    IMPORT_SYSTEM_PROMPT
    + "\n10. 上一次生成的结果有问题，本次务必修正，只输出修正后的完整代码块。"
)


def system_prompt_for(context: CodeContext) -> str:
    """按上下文选择系统提示词。"""
    return IMPORT_SYSTEM_PROMPT if context.module_available else SYSTEM_PROMPT


def _format_context(context: CodeContext) -> str:
    """把代码上下文渲染成提示词片段。"""
    lines = [
        "## 待测代码",
        f"- 文件路径：{context.path}",
        f"- 语言：{context.language}",
        f"- 符号：{context.qualified_name or '(未命名)'}（{context.kind}）",
    ]
    if context.signature:
        lines.append(f"- 签名：{context.signature}")
    if context.start_line:
        lines.append(f"- 行号：{context.start_line}-{context.end_line}")
    if context.repository:
        lines.append(f"- 所属仓库：{context.repository}")
    if context.docstring:
        lines.append(f"- 文档说明：{context.docstring.strip()}")

    lines.append("\n### 源码")
    lines.append("```python")
    lines.append(context.code.rstrip())
    lines.append("```")

    if context.related:
        lines.append("\n### 同文件相关符号（仅供理解上下文，不要为它们写测试）")
        for item in context.related:
            lines.append(f"- {item}")

    if context.existing_tests:
        lines.append("\n### 仓库现有测试（请模仿其风格与导入方式）")
        for index, snippet in enumerate(context.existing_tests, start=1):
            lines.append(f"现有测试片段 {index}：")
            lines.append("```python")
            lines.append(snippet.rstrip())
            lines.append("```")

    if context.module_available and context.import_hint:
        lines.append("\n### 必须使用的导入语句")
        lines.append("测试文件中请这样导入被测对象（不要内联实现副本）：")
        lines.append("```python")
        lines.append(context.import_hint)
        lines.append("```")

    lines.append(
        f"\n请为 `{context.display_name}` 编写 pytest 测试，只输出一个 Python 代码块。"
    )
    return "\n".join(lines)


def build_test_generation_messages(
    context: CodeContext,
    *,
    retry_feedback: str | None = None,
    previous_attempt: str | None = None,
) -> list[LLMMessage]:
    """构造测试生成的对话消息。

    Args:
        retry_feedback: 上一次生成的问题描述，触发修正轮。
        previous_attempt: 上一次生成的原始输出，供模型对照修正。
    """
    if context.module_available:
        system = IMPORT_RETRY_SYSTEM_PROMPT if retry_feedback else IMPORT_SYSTEM_PROMPT
    else:
        system = RETRY_SYSTEM_PROMPT if retry_feedback else SYSTEM_PROMPT

    messages = [LLMMessage(role="system", content=system)]

    user_content = _format_context(context)
    if retry_feedback:
        parts = [
            user_content,
            "\n## 上一次生成的结果有问题",
            f"问题：{retry_feedback}",
        ]
        if previous_attempt:
            parts.append("上一次的输出如下（供参考，请修正后重新输出完整代码块）：")
            parts.append("```text")
            # 截断避免上下文爆炸（截断本身也可能是问题原因）
            parts.append(previous_attempt[:4000])
            parts.append("```")
        parts.append("\n请输出修正后的完整 pytest 测试代码块。")
        user_content = "\n".join(parts)

    messages.append(LLMMessage(role="user", content=user_content))
    return messages


__all__ = [
    "IMPORT_RETRY_SYSTEM_PROMPT",
    "IMPORT_SYSTEM_PROMPT",
    "PROMPT_VERSION",
    "RETRY_SYSTEM_PROMPT",
    "SYSTEM_PROMPT",
    "build_test_generation_messages",
    "system_prompt_for",
]
