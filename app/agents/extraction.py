"""LLM 输出解析：从自由文本里提取可运行的 Python 代码，以及抠出 JSON 对象。

为什么单独成模块：模型输出格式极其不稳定（带解释文字、带 Markdown 围栏、
被 max_tokens 截断、整段缩进），而"提取失败"是 Agent 链路上最常见的故障点，
必须独立可测。Step 6 的修复 Agent 解析 patch 时也会复用这里的思路。

`extract_json_object` 原先写在 `tutor_agent.py` 里，后来被「AI 自动检测」等
多个模块复用，故上移到本模块统一维护（`tutor_agent` 仍然原样再导出，
不影响既有调用方）。
"""

from __future__ import annotations

import ast
import json
import logging
import re
import textwrap
from typing import Any

logger = logging.getLogger(__name__)

# ```python / ```py / ``` 三种围栏写法
_FENCED_BLOCK = re.compile(
    r"```[ \t]*(?P<lang>[A-Za-z0-9_+#-]*)[ \t]*\r?\n(?P<body>.*?)```",
    re.DOTALL,
)
# 顶层语句起始行：用于在没有围栏时裁掉前后解释文字
_TOP_LEVEL_START = re.compile(r"^[ \t]*(import\s|from\s|def\s|class\s|@|#)")
_PYTHON_LANGS = frozenset({"python", "py", "python3", ""})


def syntax_error(code: str) -> str | None:
    """返回语法错误描述；代码可解析时返回 None。"""
    if not code.strip():
        return "代码为空"
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return f"SyntaxError: {exc.msg} (line {exc.lineno})"
    except ValueError as exc:  # NUL 字节等
        return f"ValueError: {exc}"
    return None


def _dedent(code: str) -> str:
    """去掉整体缩进。

    模型在列表/解释里给出的代码常被统一缩进 4 空格，
    直接 ast.parse 会因 "unexpected indent" 失败。
    """
    return textwrap.dedent(code.replace("\t", "    "))


def _top_level_boundaries(lines: list[str], start: int) -> list[int]:
    """返回可作为截断点的行下标（下一个顶层语句的起始行）。

    只在顶层语句边界试截断，避免 O(n²) 规模的逐行解析尝试。
    """
    boundaries = []
    for index in range(start + 1, len(lines) + 1):
        if index == len(lines) or (lines[index] and not lines[index][0].isspace()):
            boundaries.append(index)
    return boundaries


def _best_parseable_slice(text: str) -> tuple[str, str | None, bool]:
    """从自由文本中找出可解析的 Python 代码片段。

    策略：先整体试；失败则从第一个"像代码"的行开始，在顶层语句边界处
    从长到短尝试截断（优先保留更完整的代码）。

    Returns:
        (code, error, trimmed)：trimmed 表示发生了截断（说明原文有残缺部分，
        调用方应给出告警，否则"截断到只剩 import"会被误判为成功）。
    """
    candidate = _dedent(text).strip()
    error = syntax_error(candidate)
    if error is None:
        return candidate, None, False

    lines = candidate.splitlines()
    for start in range(len(lines)):
        if not _TOP_LEVEL_START.match(lines[start]):
            continue
        for end in reversed(_top_level_boundaries(lines, start)):
            chunk = "\n".join(lines[start:end]).rstrip()
            if not chunk:
                continue
            if syntax_error(chunk) is None:
                return chunk, None, True
    return candidate, error, False


def _merge_blocks(blocks: list[str]) -> str:
    """合并多个代码块。

    必须**逐块先 dedent 再 strip**：先 strip 会把首行缩进抹掉，导致后续
    dedent 找不到公共缩进前缀而整体失效（模型在列表里给出的代码常被统一缩进）。
    """
    prepared = [_dedent(block).strip() for block in blocks if block.strip()]
    return "\n\n".join(prepared)


def extract_python_code(text: str) -> tuple[str, list[str]]:
    """从模型输出中提取 Python 代码。

    Returns:
        (code, warnings)。提取失败时 code 为尽力而为的文本，由调用方决定是否重试。
    """
    warnings: list[str] = []
    if not text or not text.strip():
        return "", ["模型输出为空"]

    blocks = list(_FENCED_BLOCK.finditer(text))
    if not blocks:
        warnings.append("未检测到 Markdown 代码块，已按启发式提取代码")
        code, error, trimmed = _best_parseable_slice(text)
        if trimmed:
            warnings.append("输出中含无法解析的片段，已截断保留可解析部分")
        if error:
            warnings.append(f"提取结果仍无法解析：{error}")
        return code, warnings

    # 只保留 Python 代码块；若模型的围栏没写语言，则全部视为候选
    python_blocks = [
        match.group("body")
        for match in blocks
        if match.group("lang").lower() in _PYTHON_LANGS
    ]
    if not python_blocks:
        warnings.append("代码块未标注 python，已按全部代码块处理")
        python_blocks = [match.group("body") for match in blocks]

    if len(python_blocks) > 1:
        warnings.append(f"检测到 {len(python_blocks)} 个代码块，已合并")
    if len(blocks) > len(python_blocks):
        warnings.append("输出中存在非 Python 代码块，已忽略")

    # 优先合并全部 Python 块：模型常把 `import pytest` 与测试函数分成两块，
    # 只取"含 test_ 的块"会把 import 丢掉，导致生成结果无法运行。
    merged = _merge_blocks(python_blocks)
    code, error, trimmed = _best_parseable_slice(merged)
    if error is not None:
        # 整体不可解析时，退一步只取含测试的块再试一次
        test_blocks = [block for block in python_blocks if "def test_" in block]
        if test_blocks:
            code, error, trimmed = _best_parseable_slice(_merge_blocks(test_blocks))

    if trimmed:
        warnings.append("输出中含无法解析的片段，已截断保留可解析部分")
    if error:
        warnings.append(f"代码块无法解析：{error}")
    return code, warnings


def python_test_functions(code: str) -> list[str]:
    """列出代码中的 pytest 测试函数名（含类内方法）。"""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # 语法错误时退化为正则，至少给出可用信息
        return re.findall(r"^\s*def\s+(test_\w+)", code, re.MULTILINE)

    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
            "test_"
        ):
            names.append(node.name)
    return names


def check_generated_test(code: str) -> list[str]:
    """校验生成的测试代码，返回问题列表（空列表表示通过）。

    检查项刻意只覆盖"必然导致无法运行"的问题，不做风格审查——
    风格问题交给提示词约束，避免 Agent 反复重试却改不出结果。
    """
    problems: list[str] = []
    error = syntax_error(code)
    if error:
        problems.append(f"代码无法解析：{error}")
        return problems

    if not python_test_functions(code):
        problems.append("未找到任何 test_ 开头的测试函数")

    tree = ast.parse(code)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    # 使用了 pytest 特性却没有 import pytest，运行必然 NameError
    source = code
    uses_pytest = any(
        token in source for token in ("pytest.", "pytest.mark", "@pytest")
    )
    if uses_pytest and "pytest" not in imported:
        problems.append("使用了 pytest 但未 `import pytest`")

    return problems


def extract_json_object(text: str) -> dict[str, Any] | None:
    """从模型输出里抠出 JSON 对象。

    模型经常不守规矩：套 markdown 代码块、前后加解释、甚至用单引号。
    因此按「直接解析 → 去代码块 → 取第一个 {...} 区间」三级降级尝试。

    参数:
        text: 模型返回的原始文本（可能夹着解释文字与代码围栏）。

    返回:
        解析出的 dict；完全解析不出来时返回 None（调用方据此走降级分支，
        而不是让整个请求失败）。
    """
    if not text or not text.strip():
        return None

    candidates: list[str] = [text.strip()]

    # 去掉 ```json ... ``` 包裹
    fenced = re.search(r"```(?:json)?\s*(?P<body>.*?)```", text, re.DOTALL)
    if fenced:
        candidates.append(fenced.group("body").strip())

    # 取第一个 { 到最后一个 } 之间的内容（容忍前后有解释文字）
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


__all__ = [
    "check_generated_test",
    "extract_json_object",
    "extract_python_code",
    "python_test_functions",
    "syntax_error",
]
