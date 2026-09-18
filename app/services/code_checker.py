"""AI 自动检测程序：先做本地语法检查，再让 AI 做深度检测。

--------------------------------------------------------------------------
整体流程（两阶段，顺序不能反）
--------------------------------------------------------------------------

    一段代码 + 语言
        │
        ├─ 阶段 1：本地静态检查（不联网、不花钱、毫秒级）
        │    1. 语法：Python 用内置 ast，C/Java 用 tree-sitter 的错误节点
        │       —— 拿到「第几行第几列、什么错」，这是给学生看的关键信息
        │    2. 结构：函数/类清单、圈复杂度、最长函数（复用仓库解析器）
        │    3. 风格：缩进是否混用、行尾空格、超长行、注释比例、命名规范
        │    4. 风险：Python 的可变默认参数 / 裸 except / `== None`，
        │       C 的 gets/strcpy/sprintf、Java 的 == 比较字符串等经典坑
        │
        └─ 阶段 2：AI 深度检测（需要配置大模型）
             把阶段 1 的**结论和统计一起塞进 Prompt**，让模型
               · 不要重复报告本地已经发现的问题（省 token，也避免刷屏）
               · 把精力放在本地规则查不出来的地方：逻辑错误、边界条件、
                 算法是否写对、有没有更好的写法
             要求它输出四类内容：错误 / 风格 / 风险 / 学习建议 + 评分

为什么要有阶段 1（这是本模块最重要的设计决定）：
    1. **语法错误必须准**。模型报"第几行出错"经常是错的，而 ast/tree-sitter 不会错。
       学生拿着一份"跑都跑不起来"的代码，最需要的就是精确的报错位置。
    2. **省钱、省时间**。缩进、行尾空格这类问题根本不需要动用大模型。
    3. **没配模型也能用**。阶段 1 的结果本身就有价值，接口不会因为缺 Key 而全废。

--------------------------------------------------------------------------
AI Prompt 的设计思路（详见 DEEP_CHECK_SYSTEM 上方的注释）
--------------------------------------------------------------------------

一句话总结：**把模型当成一位有耐心、会讲人话的助教，而不是一个代码评审机器人。**
具体做法是九条约束：角色与受众、注入本地结论、四类输出分界、写作规范、
评分锚点、禁止编造、严格 JSON、反面示例、语言适配。

--------------------------------------------------------------------------
说明：与「AI 代码导师」的关系
--------------------------------------------------------------------------
`app/agents/tutor_agent.py` 的 `check()` 是**给前端三个按钮用的轻量检测**；
本模块是**独立的深度检测**：多了本地静态检查阶段、问题按四类分开、
多了面向学生的学习建议、评分规则也被本地检查校准过。
两者共用同一个 `LLMClient`，Prompt 与输出结构不同，互不影响。
"""

from __future__ import annotations

import ast
import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from app.agents.extraction import extract_json_object
from app.core.config import Settings
from app.core.llm_client import (
    MISSING_API_KEY_HINT,
    MISSING_API_KEY_NOTE,
    LLMClient,
    LLMUsage,
    llm_client_for_request,
)
from app.core.trace import trace_span
from app.services.repo.language import LANGUAGE_LABELS, get_profile, load_grammar
from app.services.repo.parser import parse_source

logger = logging.getLogger(__name__)

# Prompt 版本号：改动提示词时同步 +1，便于把"结果变差"归因到哪一版
PROMPT_VERSION = "code-check/v1"

# 只支持这三种语言：与「AI 代码导师」面向的学习语言一致。
# 想加语言要同时确认三件事：仓库解析器有 profile、语法包已装、本地风格规则写好了。
SUPPORTED_LANGUAGES: tuple[str, ...] = ("c", "java", "python")

# 用户可能怎么写语言名 -> 标准标识。写成别名表而不是 if-else，
# 是为了让"支持哪些写法"一眼可见，加语言时也只改一处。
LANGUAGE_ALIASES: dict[str, str] = {
    "c": "c",
    "c99": "c",
    "c11": "c",
    "java": "java",
    "python": "python",
    "python3": "python",
    "py": "python",
}

# 严重程度：error=会出错/跑不起来，warning=有隐患，info=可以更好
Severity = Literal["error", "warning", "info"]
# 问题分类：syntax=语法，logic=逻辑，style=风格，risk=潜在 bug/风险
Category = Literal["syntax", "logic", "style", "risk"]

# 单次检测的代码上限：学生作业不会很长，超过基本是误贴了整个文件
MAX_CODE_CHARS = 200_000

# 本地风格规则的阈值。全放在这里而不是散落在函数里，
# 调整尺度（比如"行太长"放宽到 140 字符）时不必翻实现。
STYLE_LIMITS = {
    "max_line_length": 120,       # 超过算超长行
    "long_function_lines": 50,    # 单个函数超过就算太长
    "max_complexity": 10,         # 圈复杂度超过就提示拆分
    "min_comment_ratio": 0.05,    # 注释行占比低于 5% 且代码足够长时提示
    "comment_check_min_lines": 30,  # 少于这么多行就不提"注释太少"，避免小题大做
}


# ===========================================================================
# 一、AI Prompt（本模块最需要讲清楚的部分）
# ===========================================================================
# 设计思路逐条说明——每一条都对应一个真实会翻车的点：
#
# 【1】角色 + 受众写在最前面
#     模型对"你现在是谁、说给谁听"极其敏感。只写"检查代码"它会用代码评审的
#     口吻（"建议重构该模块"），学生看不懂。写明"面向大一新生的助教"之后，
#     输出会自动口语化。
#
# 【2】把本地检查结果注入 Prompt，并明确要求"不要重复报"
#     这是两阶段设计的价值所在：
#       · 语法错误位置由 ast/tree-sitter 给出，比模型猜的准；
#       · 让模型跳过缩进、行尾空格这类本地已经查完的事，把 token 花在
#         逻辑、边界、算法上——同样的调用次数，深度明显提升；
#       · 用"已知问题清单"而不是"已发现问题"，措辞上避免模型无脑附和。
#
# 【3】四类输出必须给定义和边界
#     不划边界，模型会把"命名不优雅"报成 error，学生一看就慌。所以逐条写清
#     error / style / risk 各自的判断标准，并强调"不许把风格问题升级成错误"。
#
# 【4】写作规范：通俗 + 具体 + 短
#     "通俗"要求术语第一次出现时用生活化比喻解释；
#     "具体"要求每次都给可执行的改法（给出改后的代码片段）；
#     "短"限制每段 1-2 句——不限制的话模型会写小作文，学生不会读完。
#
# 【5】评分锚点表
#     不给锚点，模型对大学生的代码一律给 80 分，分数就失去意义。
#     给出分档标准（还能不能跑、错在哪一层）后，分布明显拉开。
#
# 【6】禁止编造（幻觉约束）
#     行号必须真实存在于代码里、不确定就填 null；引用代码必须原样摘抄；
#     不要提到代码里没有的函数名/变量名。这三条覆盖了模型最常见的三类编造。
#
# 【7】严格 JSON + 字段说明
#     只输出一个 JSON 对象，不要 markdown 围栏。真不听话时由
#     `extract_json_object` 兜底（三级降级解析），不因为格式问题让整次请求失败。
#
# 【8】给反面示例
#     模型对"不要这样写"的示例比"要这样写"更敏感。明确禁止
#     "你的代码有问题""建议优化"这类空话，并要求每条建议都能直接照着改。
#
# 【9】语言适配
#     把语言名传进去，并要求建议符合该语言习惯（Python 的 snake_case、
#     Java 的 camelCase、C 的手动内存管理），避免给出"用 Pythonic 写法"
#     这种对着 C 代码说的话。
DEEP_CHECK_SYSTEM = """你是一位有耐心的编程入门课助教，正在帮大一新生检查他刚写完的代码。

## 你的读者
一名刚开始学编程的学生。他看得懂基本语法，但对"为什么这样写会出问题"还不熟。
请用**通俗的中文**解释，不要堆术语；必须用到术语时，用一句生活化的比喻解释它。

## 检查重点（按优先级从高到低）
1. **语法错误**：写错的关键字、缺分号/括号、类型不匹配
2. **逻辑错误**：条件写反、循环边界差一、变量用错、返回值不对、算法思路有误
3. **潜在 Bug 与风险**：数组越界、空指针、除零、资源未释放、边界条件没处理
4. **代码风格**：命名、缩进、注释、魔法数字、重复代码

## 已知的本地检查结果
系统已经用 AST / tree-sitter 和静态规则查过下面这些内容，**不要重复报告**，
把它们当作前提，把精力放在它们查不出来的地方（逻辑、算法、边界条件、更好的写法）：
<<LOCAL_FINDINGS>>

## 输出四类内容，边界要分清
- `errors`（语法/逻辑错误）：会让程序**跑不起来或结果不对**的问题。
  命名不好看、缩进不统一，**绝不能**放进这一类。
- `style`（代码风格建议）：命名规范、缩进、注释、可读性。即使不改也能跑。
- `risks`（潜在 Bug 与风险）：现在能跑，但在某些输入或情况下会出问题，
  例如数组越界、除零、空指针、内存泄漏、未关闭的文件。
- `advice`（学习建议）：**面向这名学生**的 2-4 条建议，不是针对某一行代码，
  而是告诉他这类问题以后怎么写、该补哪个知识点。用鼓励的语气。

## 写作规范
- 每条的 `detail` 用 1-2 句话讲清"为什么这是问题"，别写小作文
- 每条的 `suggestion` 必须**可以直接照着改**，尽量给出改后的代码写法
- 不要出现"你的代码有问题""建议优化""不够优雅"这类空话
- 不要提代码里根本不存在的函数名或变量名

## 评分（0-100 整数），请严格对照下面的锚点
- 90-100：逻辑正确、边界也考虑了，可以直接交作业
- 75-89：能跑、结果对，但有小毛病（命名、注释、少量重复代码）
- 60-74：能跑但存在明显问题（某类输入会出错、边界没处理、代码难懂）
- 40-59：跑不起来（语法/编译错误），或逻辑明显写错
- 0-39：大面积错误，建议重新理清思路再写

## 输出格式
只输出一个 JSON 对象，不要任何解释文字，不要用 markdown 代码块包裹：
{
  "score": 78,
  "summary": "一句话总体评价，用学生能听懂的话",
  "errors": [
    {"line": 12, "severity": "error", "title": "循环多跑了一次", "detail": "…", "suggestion": "…"}
  ],
  "style": [
    {"line": null, "severity": "info", "title": "变量名 n 含义不清",
     "detail": "…", "suggestion": "…"}
  ],
  "risks": [
    {"line": 20, "severity": "warning", "title": "没有判断除数是否为 0",
     "detail": "…", "suggestion": "…"}
  ],
  "advice": ["先把「边界情况」当成习惯：写完循环就想想 i 的起点和终点", "…"],
  "highlights": ["变量命名清晰", "缩进规范"]
}

规则：
- `line` 只能是代码里真实存在的行号（从 1 开始）；拿不准就填 null，**不要猜**
- `severity` 只能是 "error" / "warning" / "info"
- 某一类没有内容就填空数组 []，不要为了凑数编问题
- `highlights` 最多 3 条：学生做对的地方也要说出来
"""


# 提示词里的占位符。
# **不要用 str.format() 填这个洞**：提示词里含 JSON 输出示例，其中的 `{` `}`
# 会被 format 当成格式化字段，直接抛 `KeyError: '\n  "score"'`（实测踩过）。
# 用 <<...>> 这种不会和 JSON 冲突的记号 + str.replace()，改提示词时也不会踩坑。
LOCAL_FINDINGS_PLACEHOLDER = "<<LOCAL_FINDINGS>>"


def build_user_message(
    code: str, *, language: str, filename: str, local_findings: str
) -> str:
    """构造发给模型的「用户消息」。

    参数:
        code:           学生代码原文（原样放进围栏，不做任何改写）。
        language:       语言标识（c / java / python），用于选择代码围栏与语言习惯。
        filename:       文件名，给模型一点上下文（例如 .c 还是 .h）。
        local_findings: 本地检查结论的文本摘要，会填进系统提示里的占位符。

    返回:
        拼接好的用户消息文本。

    关键逻辑:
        把「已知问题」放在系统提示里、把「代码」放在用户消息里，是刻意的：
        系统提示描述"你要怎么干活"，用户消息才是"这次要看的材料"。
        代码用 ``` 围栏包起来并写明语言，可以让模型少犯"把 C 当 Python 看"的错。
    """
    return (
        f"请检查下面这段 {language_label(language)} 代码"
        f"（文件名：{filename}）。\n\n"
        f"```{language}\n{code}\n```\n\n"
        f"再提醒一次本地检查已经查过的内容，请勿重复报告：\n{local_findings}"
    )


def build_messages(
    code: str, *, language: str, filename: str, local_findings: str
) -> list[Any]:
    """组装完整的消息列表（系统 + 用户）。

    单独抽成函数的好处：测试里可以直接断言"提示词里确实包含了本地结论"
    "确实要求了不要重复报告"，而不必真的去调模型。
    """
    from app.core.llm_client import LLMMessage

    return [
        LLMMessage(
            role="system",
            content=DEEP_CHECK_SYSTEM.replace(LOCAL_FINDINGS_PLACEHOLDER, local_findings),
        ),
        LLMMessage(
            role="user",
            content=build_user_message(
                code, language=language, filename=filename, local_findings=local_findings
            ),
        ),
    ]


def language_label(language: str) -> str:
    """语言标识 -> 展示名（C / Java / Python）；未知的原样返回。"""
    return LANGUAGE_LABELS.get(language, language)


def normalize_language(value: str | None, filename: str | None = None) -> str:
    """把用户传来的语言名归一成标准标识。

    参数:
        value:    用户写的语言，如 "C" / "Python" / "py"；可为空。
        filename: 语言为空时的兜底：按文件后缀猜（snippet.py -> python）。

    返回:
        标准语言标识（c / java / python）。

    异常:
        UnsupportedLanguageError: 既没给语言、也没法从文件名推断，或语言不支持。

    关键逻辑:
        大小写不敏感、支持常见别名（py / python3 / c11），
        这样学生在界面上下拉选错大小写也不会报错。
    """
    text = (value or "").strip().lower()
    if not text and filename:
        from app.services.repo.language import detect_language

        text = (detect_language(filename) or "").strip().lower()
    resolved = LANGUAGE_ALIASES.get(text)
    if resolved is None:
        raise UnsupportedLanguageError(
            f"不支持的语言：{value or '(空)'}。"
            f"目前支持：" + "、".join(SUPPORTED_LANGUAGES)
        )
    return resolved


class UnsupportedLanguageError(ValueError):
    """语言不在支持范围内。调用方应转成 400。"""


class CodeCheckError(RuntimeError):
    """检测过程中的可预期错误（调用方应转成 4xx）。"""


# ===========================================================================
# 二、阶段 1：本地静态检查
# ===========================================================================
@dataclass(slots=True)
class Issue:
    """一条问题。本地规则与 AI 结论共用同一个结构，前端才好统一渲染。"""

    title: str
    detail: str = ""
    suggestion: str = ""
    line: int | None = None
    severity: Severity = "info"
    category: Category = "style"
    # 来源：local=本地规则/解析器，ai=模型。让学生（和我们）能分辨结论从哪来
    source: Literal["local", "ai"] = "local"

    def to_dict(self) -> dict[str, Any]:
        return {
            "line": self.line,
            "severity": self.severity,
            "category": self.category,
            "title": self.title,
            "detail": self.detail,
            "suggestion": self.suggestion,
            "source": self.source,
        }


@dataclass(slots=True)
class SyntaxErrorInfo:
    """语法错误的精确位置。"""

    line: int | None
    column: int | None
    message: str
    tool: str          # ast / tree-sitter —— 让学生知道这个结论是谁给的
    raw: str           # 原始错误字符串，便于排查工具本身的问题


@dataclass(slots=True)
class CodeMetrics:
    """代码统计指标。这些数字既展示给学生，也喂给模型当上下文。"""

    total_lines: int = 0
    code_lines: int = 0
    comment_lines: int = 0
    blank_lines: int = 0
    comment_ratio: float = 0.0
    max_line_length: int = 0
    long_lines: int = 0
    trailing_whitespace_lines: int = 0
    mixed_indent: bool = False
    uses_tabs: bool = False
    function_count: int = 0
    class_count: int = 0
    max_complexity: int = 0
    longest_function: int = 0
    longest_function_name: str = ""
    symbols: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_lines": self.total_lines,
            "code_lines": self.code_lines,
            "comment_lines": self.comment_lines,
            "blank_lines": self.blank_lines,
            "comment_ratio": round(self.comment_ratio, 4),
            "max_line_length": self.max_line_length,
            "long_lines": self.long_lines,
            "trailing_whitespace_lines": self.trailing_whitespace_lines,
            "mixed_indent": self.mixed_indent,
            "uses_tabs": self.uses_tabs,
            "function_count": self.function_count,
            "class_count": self.class_count,
            "max_complexity": self.max_complexity,
            "longest_function": self.longest_function,
            "longest_function_name": self.longest_function_name,
            "symbols": self.symbols,
        }


@dataclass(slots=True)
class LocalCheckResult:
    """阶段 1 的全部产出。"""

    language: str
    syntax_error: SyntaxErrorInfo | None = None
    metrics: CodeMetrics = field(default_factory=CodeMetrics)
    issues: list[Issue] = field(default_factory=list)
    duration_ms: float = 0.0

    @property
    def syntax_ok(self) -> bool:
        return self.syntax_error is None

    def findings_text(self) -> str:
        """把本地结论整理成一段文字，注入 AI 提示词。

        刻意写成"清单 + 统计数字"而不是 JSON：
        模型读自然语言清单更稳，而统计数字（多少行、多长、注释率）
        能帮它建立"这是份什么水平的代码"的判断。
        """
        metrics = self.metrics
        lines = [
            f"- 语言：{language_label(self.language)}；共 {metrics.total_lines} 行"
            f"（代码 {metrics.code_lines} 行、注释 {metrics.comment_lines} 行、"
            f"空行 {metrics.blank_lines} 行）",
            f"- 结构：函数 {metrics.function_count} 个、类 {metrics.class_count} 个；"
            f"最高圈复杂度 {metrics.max_complexity}",
        ]
        if self.syntax_error is not None:
            where = f"第 {self.syntax_error.line} 行" if self.syntax_error.line else "位置未知"
            lines.append(
                f"- 语法：**不通过**，{where} 有语法错误（{self.syntax_error.tool} 报告："
                f"{self.syntax_error.message}）"
            )
        else:
            lines.append("- 语法：通过（AST / tree-sitter 未发现语法错误）")

        if self.issues:
            lines.append("- 本地规则已发现（请勿重复报告）：")
            for item in self.issues:
                pos = f"第 {item.line} 行 " if item.line else ""
                lines.append(f"    · {pos}{item.title}")
        else:
            lines.append("- 本地规则未发现风格与常见风险问题")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 各语言的注释写法（用于统计注释行）
# ---------------------------------------------------------------------------
_LINE_COMMENT = {"python": "#", "c": "//", "java": "//"}
_BLOCK_COMMENT = {"c": ("/*", "*/"), "java": ("/*", "*/")}


def count_lines_by_kind(code: str, language: str) -> tuple[int, int, int]:
    """统计代码行 / 注释行 / 空行。

    参数:
        code:     源码文本。
        language: 语言标识，决定注释符号。

    返回:
        `(代码行数, 注释行数, 空行数)`。

    关键逻辑:
        · Python 额外把 **docstring（三引号文档字符串）算作注释行**（用 ast 精确定位）：
          它在语法上是字符串，但对学生来说它就是文档。
          不这么算的话，一份每个函数都写了 docstring 的好代码会被规则
          判成"几乎没有注释"，那是最冤枉的误报。
        · C/Java 跟踪 `/* ... */` 块注释的起止。
        · 行内注释（`x = 1  # 说明`）按代码行计——它首先是代码。
          这样统计出的"注释率"才是学生理解的"我写了多少行注释"。
    """
    lines = code.splitlines()
    blank = comment = 0
    in_block = False
    marker = _LINE_COMMENT.get(language, "#")
    block = _BLOCK_COMMENT.get(language)

    # Python：先把 docstring 占用的行记下来，避免下面按行判断时把它们当成代码
    docstring_lines: set[int] = set()
    if language == "python":
        docstring_lines = _python_docstring_lines(code)

    for index, raw in enumerate(lines, start=1):
        stripped = raw.strip()
        if not stripped:
            blank += 1
            continue
        if in_block:
            comment += 1
            if block and block[1] in stripped:
                in_block = False
            continue
        if block and stripped.startswith(block[0]):
            comment += 1
            # 同一行就闭合的情况（/* xxx */）不算未闭合
            if block[1] not in stripped[len(block[0]) :]:
                in_block = True
            continue
        if stripped.startswith(marker):
            comment += 1
            continue
        if index in docstring_lines:
            comment += 1
            continue

    total = len(lines)
    return total - blank - comment, comment, blank


def _python_docstring_lines(code: str) -> set[int]:
    """找出所有 docstring 占据的行号（模块/类/函数的第一个字符串字面量）。

    参数:
        code: Python 源码。

    返回:
        行号集合（从 1 开始）。语法错误时返回空集合——
        这时由语法检查单独报错，注释统计退回按 `#` 判断即可。
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return set()

    spans: set[int] = set()
    holders = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if not isinstance(node, holders):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if not (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            continue
        start = first.lineno
        end = getattr(first, "end_lineno", start) or start
        spans.update(range(start, end + 1))
    return spans


def compute_metrics(code: str, language: str, parsed: Any) -> CodeMetrics:
    """算出展示与提示词都要用的统计指标。

    参数:
        code:     源码文本。
        language: 语言标识。
        parsed:   仓库解析器的结果（`ParsedFile`），提供函数/类清单与复杂度。

    返回:
        `CodeMetrics`。

    关键逻辑:
        解析器（ast / tree-sitter）负责"结构化"的信息，本函数只做文本层面的统计，
        两者合起来才是一份完整的"体检报告"。
        缩进混用（同一文件里既有 Tab 又有空格）是初学者最常见的坑之一，
        在 Python 里会直接 IndentationError，所以单独标出来。
    """
    code_lines, comment_lines, blank_lines = count_lines_by_kind(code, language)
    total_lines = len(code.splitlines())

    rows = code.splitlines()
    lengths = [len(row) for row in rows]
    long_lines = sum(1 for item in lengths if item > STYLE_LIMITS["max_line_length"])
    trailing = sum(1 for row in rows if row != row.rstrip() and row.strip())

    tab_indented = sum(1 for row in rows if row.startswith("\t"))
    space_indented = sum(1 for row in rows if row.startswith(" ") and row.strip())

    metrics = CodeMetrics(
        total_lines=total_lines,
        code_lines=code_lines,
        comment_lines=comment_lines,
        blank_lines=blank_lines,
        comment_ratio=(comment_lines / total_lines) if total_lines else 0.0,
        max_line_length=max(lengths) if lengths else 0,
        long_lines=long_lines,
        trailing_whitespace_lines=trailing,
        mixed_indent=tab_indented > 0 and space_indented > 0,
        uses_tabs=tab_indented > 0,
    )

    symbols = getattr(parsed, "symbols", None) or []
    functions = [item for item in symbols if item.kind in ("function", "method")]
    metrics.function_count = len(functions)
    metrics.class_count = sum(1 for item in symbols if item.kind == "class")
    metrics.max_complexity = max((item.complexity for item in symbols), default=0)

    if functions:
        longest = max(functions, key=lambda item: item.end_line - item.start_line)
        metrics.longest_function = longest.end_line - longest.start_line + 1
        metrics.longest_function_name = longest.name
    metrics.symbols = [
        {
            "name": item.name,
            "qualified_name": item.qualified_name,
            "kind": item.kind,
            "start_line": item.start_line,
            "end_line": item.end_line,
            "complexity": item.complexity,
            # 签名（含参数名）要带上：注释生成功能靠它写"参数说明"，
            # 少了它就只能让模型自己猜参数叫什么，很容易张冠李戴。
            "signature": item.signature,
        }
        for item in symbols
    ]
    return metrics


# ---------------------------------------------------------------------------
# 风险模式：只在「把注释和字符串挖掉之后」的代码上匹配
# ---------------------------------------------------------------------------
# 为什么必须先挖掉注释与字符串：学生代码里经常出现
# `// 这里用 strcpy 会有问题` 这样的注释，直接对原文正则匹配就会报出假问题。
# 挖空之后只匹配真实代码，误报率明显下降；但仍是"模式匹配"，
# 因此这些结论一律标 severity=warning/info，且 source=local，提示需人工确认。
_C_RISK_PATTERNS = [
    (
        r"\bgets\s*\(",
        "使用了不安全的 gets()",
        "gets() 不检查长度，输入稍长就会冲掉内存，"
        "改用 fgets(buf, sizeof(buf), stdin)",
    ),
    (
        r"\bstrcpy\s*\(",
        "使用了不检查长度的 strcpy()",
        "strcpy() 不限制长度，源字符串偏长就会缓冲区溢出，"
        "改用 strncpy 并手动补 '\\0'",
    ),
    (
        r"\bsprintf\s*\(",
        "使用了 sprintf()",
        "sprintf() 同样不限制长度，建议改用 snprintf(buf, sizeof(buf), ...)",
    ),
    (
        r"\bscanf\s*\(\s*\"%s\"",
        "scanf(\"%s\") 没有限制长度",
        "加上宽度限制，例如 scanf(\"%19s\", buf)，否则输入过长会溢出",
    ),
    (
        r"\bmalloc\s*\(",
        "使用了 malloc，记得检查返回值并释放",
        "malloc 可能返回 NULL，用完要 free，否则会内存泄漏",
    ),
]
_JAVA_RISK_PATTERNS = [
    (
        r"==\s*\"",
        "用 == 比较字符串",
        "Java 里 == 比较的是对象地址，判断内容是否相同要用 a.equals(b)",
    ),
    (
        r"catch\s*\([^)]*\)\s*\{\s*\}",
        "空的 catch 块",
        "空 catch 会把异常悄悄吞掉；至少打印日志，或把错误告诉用户",
    ),
    (
        r"System\.exit\s*\(",
        "调用了 System.exit()",
        "在一般方法里直接退出程序会让调用方无法处理，建议改成抛异常或返回错误码",
    ),
]
_PYTHON_RISK_HINTS = {
    "bare_except": (
        "使用了裸 except",
        "`except:` 会连键盘中断一起吞掉，改成 `except Exception as exc:` 并处理它",
    ),
    "mutable_default": (
        "函数默认参数用了可变对象",
        "默认值只在定义时创建一次，会被多次调用共享；"
        "改成 `def f(items=None):` 再在函数体内 `items = items or []`",
    ),
    "none_compare": (
        "用 == 比较 None",
        "判断空值应用 `is None`，`==` 会被自定义的 __eq__ 影响",
    ),
}


def _mask_comments_and_strings(code: str, language: str) -> str:
    """把注释与字符串内容替换成等长空格，保留换行与行号。

    参数:
        code:     源码文本。
        language: 语言标识。

    返回:
        与原文**行数、每行长度完全一致**的掩码文本。

    关键逻辑:
        保留长度与换行使后面的正则匹配仍能报告正确的行号；
        把内容换成空格则避免匹配到注释/字符串里提到的关键字。
        C/Java 处理 // 与 /* */，Python 处理 # 与三引号字符串。
    """
    if language == "python":
        # Python 用 tokenize 才严谨，但这里的目标只是"降低误报"，
        # 因此用简化规则：逐行找 #（忽略字符串里的 # 代价可接受）
        masked_lines = []
        for row in code.splitlines():
            index = _find_python_comment(row)
            masked_lines.append(row if index is None else row[:index] + " " * (len(row) - index))
        return "\n".join(masked_lines)

    out: list[str] = []
    in_block = False
    for row in code.splitlines():
        chars = list(row)
        index = 0
        in_string: str | None = None
        while index < len(chars):
            two = row[index : index + 2]
            if in_block:
                chars[index] = " "
                if two == "*/":
                    chars[index + 1] = " "
                    in_block = False
                    index += 2
                    continue
                index += 1
                continue
            if in_string:
                if row[index] == "\\":       # 转义字符，跳过下一个
                    chars[index] = " "
                    if index + 1 < len(chars):
                        chars[index + 1] = " "
                    index += 2
                    continue
                if row[index] == in_string:
                    # 收尾的引号**保留**：后面靠"引号"判断"这里用了字符串字面量"
                    # （例如 Java 的 `a == "x"`），把引号也抹掉就查不出来了
                    in_string = None
                    index += 1
                    continue
                chars[index] = " "
                index += 1
                continue
            if two == "//":
                for rest in range(index, len(chars)):
                    chars[rest] = " "
                break
            if two == "/*":
                chars[index] = chars[index + 1] = " "
                in_block = True
                index += 2
                continue
            if row[index] in "\"'":
                # 起始引号同样保留，只把字符串内容抹成空格
                in_string = row[index]
            index += 1
        out.append("".join(chars))
    return "\n".join(out)


def _find_python_comment(row: str) -> int | None:
    """找出一行 Python 代码里注释的起始位置；没有注释返回 None。

    简化处理：跳过被引号包住的部分，因此 `print("# 井号")` 不会被当成注释。
    """
    quote: str | None = None
    index = 0
    while index < len(row):
        char = row[index]
        if quote:
            if char == "\\":
                index += 2
                continue
            if char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char == "#":
            return index
        index += 1
    return None


# ---------------------------------------------------------------------------
# 命名规范：按语言约定检查，措辞尽量温和（这是风格，不是错误）
# ---------------------------------------------------------------------------
def check_naming(metrics: CodeMetrics, language: str) -> list[Issue]:
    """检查命名是否符合该语言的常见约定。

    参数:
        metrics:  含 symbols 的统计结果。
        language: 语言标识。

    返回:
        风格类问题列表。

    关键逻辑:
        只查"几乎公认"的几条，且一律标成 info：
          · Python 函数名应 snake_case、类名首字母大写
          · Java 类名首字母大写、方法名首字母小写
          · C 函数名通常全小写
        刻意不查"变量名是否有意义"这类主观项——那是 AI 阶段更擅长的。
    """
    issues: list[Issue] = []
    for symbol in metrics.symbols:
        name = symbol["name"]
        kind = symbol["kind"]

        if language == "python":
            if kind in ("function", "method") and any(char.isupper() for char in name):
                issues.append(
                    Issue(
                        line=symbol["start_line"],
                        title=f"函数名 {name} 用了大写字母",
                        detail="Python 的惯例是函数名全小写、单词之间用下划线连接。",
                        suggestion=f"建议改成 {_to_snake(name)}",
                        category="style",
                    )
                )
            elif kind == "class" and name[:1].islower():
                issues.append(
                    Issue(
                        line=symbol["start_line"],
                        title=f"类名 {name} 没有首字母大写",
                        detail="Python 的类名惯例是每个单词首字母大写（CamelCase）。",
                        suggestion=f"建议改成 {name[:1].upper() + name[1:]}",
                        category="style",
                    )
                )
        elif language == "java":
            if kind == "class" and name[:1].islower():
                issues.append(
                    Issue(
                        line=symbol["start_line"],
                        title=f"类名 {name} 没有首字母大写",
                        detail="Java 的类名必须是首字母大写的驼峰写法。",
                        suggestion=f"建议改成 {name[:1].upper() + name[1:]}",
                        category="style",
                    )
                )
            elif kind == "method" and name[:1].isupper():
                issues.append(
                    Issue(
                        line=symbol["start_line"],
                        title=f"方法名 {name} 首字母大写",
                        detail="Java 的方法名惯例是小写开头的驼峰写法，首字母大写通常留给类名。",
                        suggestion=f"建议改成 {name[:1].lower() + name[1:]}",
                        category="style",
                    )
                )
        elif language == "c" and kind == "function" and name[:1].isupper():
            issues.append(
                Issue(
                    line=symbol["start_line"],
                    title=f"函数名 {name} 以大写字母开头",
                    detail="C 语言里函数名通常全小写（首字母大写一般留给结构体类型名）。",
                    suggestion="建议改成全小写，例如 " + name[:1].lower() + name[1:],
                    category="style",
                )
            )
    return issues


def _to_snake(name: str) -> str:
    """把驼峰命名转成下划线命名，仅用于给出建议（不修改学生代码）。"""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


# ---------------------------------------------------------------------------
# 风险模式匹配
# ---------------------------------------------------------------------------
def check_risk_patterns(code: str, language: str) -> list[Issue]:
    """检查各语言里最经典的坑（模式匹配）。

    参数:
        code:     源码原文。
        language: 语言标识。

    返回:
        风险类问题列表（severity=warning，source=local）。

    关键逻辑:
        先 `_mask_comments_and_strings` 再匹配，避免把注释里的说明当成代码；
        **字符串引号会保留**（只抹内容），这样 `a == "x"` 这种"用 == 比较字符串"
        的写法才查得出来。
        每条都给出"为什么危险 + 怎么改"，而不是只报一个函数名。
        Python 的几条用 AST 精确判断（见 `check_python_ast_risks`），
        因为 ast 比正则可靠得多；这里只保留 C/Java 的正则。
    """
    issues: list[Issue] = []
    if language == "python":
        return issues  # Python 交给 AST 版本，见 check_python_ast_risks

    masked = _mask_comments_and_strings(code, language)
    patterns = _C_RISK_PATTERNS if language == "c" else _JAVA_RISK_PATTERNS
    for pattern, title, suggestion in patterns:
        for match in re.finditer(pattern, masked):
            line = masked[: match.start()].count("\n") + 1
            issues.append(
                Issue(
                    line=line,
                    severity="warning",
                    category="risk",
                    title=title,
                    detail="这条由本地静态规则给出（模式匹配），请结合上下文确认。",
                    suggestion=suggestion,
                )
            )
    return issues


def check_python_ast_risks(code: str) -> list[Issue]:
    """用 ast 检查 Python 里几个"一看就是 bug"的写法。

    参数:
        code: Python 源码（语法错误时返回空列表，由语法检查单独报告）。

    返回:
        风险类问题列表。

    关键逻辑:
        这几条用 AST 判断比正则准得多：
          · 裸 `except:` / `except Exception:` 吞掉所有异常
          · 可变对象当默认参数（列表/字典/集合）——多次调用会共享同一个对象
          · 用 `== None` 而不是 `is None`
        AST 能拿到精确行号，因此这些结论可以直接给学生看。
    """
    issues: list[Issue] = []
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return issues

    for node in ast.walk(tree):
        # 1) 可变默认参数
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for default in list(node.args.defaults) + [
                item for item in node.args.kw_defaults if item is not None
            ]:
                if isinstance(default, (ast.List, ast.Dict, ast.Set)) or (
                    isinstance(default, ast.Call)
                    and getattr(default.func, "id", "") in {"list", "dict", "set"}
                ):
                    title, suggestion = _PYTHON_RISK_HINTS["mutable_default"]
                    issues.append(
                        Issue(
                            line=getattr(default, "lineno", node.lineno),
                            severity="warning",
                            category="risk",
                            title=f"{node.name} 的默认参数用了可变对象",
                            detail=title
                            + "：默认值只在函数定义时创建一次，多次调用会共用同一个对象。",
                            suggestion=suggestion,
                        )
                    )
        # 2) 裸 except
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            title, suggestion = _PYTHON_RISK_HINTS["bare_except"]
            issues.append(
                Issue(
                    line=node.lineno,
                    severity="warning",
                    category="risk",
                    title=title,
                    detail="它会把所有异常（包括 Ctrl+C 的中断）都吃掉，出错了也查不出原因。",
                    suggestion=suggestion,
                )
            )
        # 3) == None
        if isinstance(node, ast.Compare) and any(
            isinstance(op, (ast.Eq, ast.NotEq)) for op in node.ops
        ):
            for comparator in [node.left, *node.comparators]:
                if isinstance(comparator, ast.Constant) and comparator.value is None:
                    title, suggestion = _PYTHON_RISK_HINTS["none_compare"]
                    issues.append(
                        Issue(
                            line=node.lineno,
                            severity="info",
                            category="risk",
                            title=title,
                            detail="`==` 比较的是值，而 None 应该用身份判断，这是 Python 的惯例。",
                            suggestion=suggestion,
                        )
                    )
    return issues


def check_style_rules(metrics: CodeMetrics, language: str) -> list[Issue]:
    """基于统计指标给出风格建议。

    参数:
        metrics:  统计结果。
        language: 语言标识。

    返回:
        风格类问题列表。

    关键逻辑:
        只报"确实值得说一句"的：
          · 缩进 Tab / 空格混用（Python 下会直接报错，C/Java 下也不该出现）
          · 行尾多余空格（数量多时才提，少于一行的琐碎问题不打扰学生）
          · 超长行（超过阈值，且给出具体行号）
          · 注释太少（仅在代码足够长时才提，避免小题大做）
          · 函数太长、圈复杂度太高
    """
    issues: list[Issue] = []

    if metrics.mixed_indent:
        issues.append(
            Issue(
                title="缩进混用了 Tab 和空格",
                detail="同一份代码里两种缩进混用，Python 会直接报 IndentationError，"
                       "其它语言也会在不同编辑器里显示错乱。",
                suggestion="统一用 4 个空格（VS Code 右下角可以一键转换）",
                severity="warning",
                category="style",
            )
        )
    elif metrics.uses_tabs and language == "python":
        issues.append(
            Issue(
                title="Python 代码用了 Tab 缩进",
                detail="Python 官方建议用 4 个空格缩进；Tab 在不同编辑器宽度不一致，容易看错层级。",
                suggestion="改成 4 个空格缩进",
                category="style",
            )
        )

    if metrics.trailing_whitespace_lines >= 3:
        issues.append(
            Issue(
                title=f"有 {metrics.trailing_whitespace_lines} 行行尾多了空格",
                detail="行尾空格不影响运行，但会让版本对比时出现无意义的差异。",
                suggestion="编辑器里开启「保存时删除行尾空格」即可自动处理",
                category="style",
            )
        )

    if metrics.long_lines:
        issues.append(
            Issue(
                title=f"有 {metrics.long_lines} 行超过 {STYLE_LIMITS['max_line_length']} 个字符",
                detail=(
                    f"最长的一行有 {metrics.max_line_length} 个字符，"
                    "横向滚动才能看全，容易看漏内容。"
                ),
                suggestion="把长表达式拆成几行，或者提取成中间变量",
                category="style",
            )
        )

    if (
        metrics.total_lines >= STYLE_LIMITS["comment_check_min_lines"]
        and metrics.comment_ratio < STYLE_LIMITS["min_comment_ratio"]
    ):
        issues.append(
            Issue(
                title="关键逻辑几乎没有注释",
                detail=(
                    f"这份代码注释占比只有 {metrics.comment_ratio:.1%}。"
                    "过一阵子回头看会很吃力。"
                ),
                suggestion="在每个函数开头写一句它是干什么的，复杂判断前写一句思路",
                category="style",
            )
        )

    if metrics.longest_function > STYLE_LIMITS["long_function_lines"]:
        issues.append(
            Issue(
                title=f"函数 {metrics.longest_function_name} 有 {metrics.longest_function} 行",
                detail="单个函数太长时，里面的每一步都难单独检查，出错了也不好定位。",
                suggestion="按「一件事一个函数」拆成几个小函数，每个只做一步",
                category="style",
            )
        )

    if metrics.max_complexity > STYLE_LIMITS["max_complexity"]:
        issues.append(
            Issue(
                title=f"有一处逻辑的分支复杂度达到 {metrics.max_complexity}",
                detail="条件嵌套太多（if/for/while 层层叠加）时，很难确认每种组合都对。",
                suggestion="把嵌套的条件提前 return，或者拆成小函数",
                category="style",
            )
        )
    return issues


# ===========================================================================
# 三、CodeChecker：编排两个阶段
# ===========================================================================
@dataclass(slots=True)
class CheckOutcome:
    """一次检测的完整结果（阶段 1 + 阶段 2 合并后）。"""

    filename: str
    language: str
    local: LocalCheckResult
    errors: list[Issue] = field(default_factory=list)
    style: list[Issue] = field(default_factory=list)
    risks: list[Issue] = field(default_factory=list)
    advice: list[str] = field(default_factory=list)
    highlights: list[str] = field(default_factory=list)
    summary: str = ""
    score: float = 0.0
    score_reason: str = ""
    ai_score: float | None = None
    ai_available: bool = False
    model: str = ""
    usage: LLMUsage = field(default_factory=LLMUsage)
    warnings: list[str] = field(default_factory=list)
    note: str = ""
    duration_ms: float = 0.0
    ai_duration_ms: float | None = None
    checked_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def all_issues(self) -> list[Issue]:
        """三类问题合在一起，便于统计与写库。"""
        return [*self.errors, *self.style, *self.risks]

    @property
    def level(self) -> str:
        """分数 -> 等级（与 AI 导师模块保持同一套口径）。"""
        if self.score >= 90:
            return "优秀"
        if self.score >= 75:
            return "良好"
        if self.score >= 60:
            return "及格"
        return "待改进"


class CodeChecker:
    """AI 自动检测：本地静态检查 + 大模型深度检测。"""

    # 类名以 Code 开头不会被 pytest 当成测试类；仍显式声明一次更保险
    __test__ = False

    def __init__(self, settings: Settings, llm: LLMClient) -> None:
        """构造检测器。

        参数:
            settings: 全局配置（多模型表、默认超时等都在里面）。
            llm: **默认**客户端（通常来自 `.env` 配置或测试注入的假客户端）。
                 真正调用时会先用 `llm_client_for_request()` 看看这次请求
                 有没有"用户自己选的模型 + 自己的 Key"，有就用那个。
        """
        self._settings = settings
        self._llm = llm

    @property
    def llm(self) -> LLMClient:
        return self._llm

    # ------------------------------------------------------------------
    # 阶段 1
    # ------------------------------------------------------------------
    def local_check(self, code: str, *, language: str, filename: str) -> LocalCheckResult:
        """本地静态检查：不联网、不花钱，语法 + 结构 + 风格 + 常见风险。

        参数:
            code:     源码文本。
            language: 标准语言标识（已由 normalize_language 归一）。
            filename: 文件名（用于报错信息与语言推断展示）。

        返回:
            `LocalCheckResult`。

        关键逻辑:
            1. 复用仓库解析器 `parse_source`：Python 走 ast、其它走 tree-sitter，
               顺带拿到函数/类清单与圈复杂度——这部分代码已在 Step 2 被大量测试覆盖，
               不重复造轮子。
            2. 语法错误的**精确位置**单独补一次：
               `parse_source` 返回的 parse_error 是给人看的字符串，
               而学生需要的是"第几行第几列"。Python 直接读 SyntaxError 的
               lineno/offset；C/Java 用 tree-sitter 找第一个 ERROR/MISSING 节点。
            3. 任何一步出错都不抛异常：本地检查失败只应导致"少几条建议"，
               不应该让整个接口 500。
        """
        started = time.perf_counter()
        result = LocalCheckResult(language=language)

        raw = code.encode("utf-8", errors="replace")
        try:
            parsed = parse_source(raw, filename, language)
            result.metrics = compute_metrics(code, language, parsed)
        except Exception:  # noqa: BLE001 - 解析器异常不该让检测整体失败
            logger.warning("本地解析失败，降级为纯文本统计: %s", filename, exc_info=True)
            result.metrics = compute_metrics(code, language, None)

        result.syntax_error = locate_syntax_error(code, language=language, filename=filename)

        issues: list[Issue] = []
        issues.extend(check_style_rules(result.metrics, language))
        issues.extend(check_naming(result.metrics, language))
        if language == "python":
            issues.extend(check_python_ast_risks(code))
        else:
            issues.extend(check_risk_patterns(code, language))
        result.issues = _dedupe_issues(issues)

        result.duration_ms = (time.perf_counter() - started) * 1000
        return result

    # ------------------------------------------------------------------
    # 阶段 2
    # ------------------------------------------------------------------
    async def check(
        self,
        code: str,
        *,
        language: str,
        filename: str,
        session_id: str | None = None,
        model_id: str | None = None,
        api_key: str | None = None,
    ) -> CheckOutcome:
        """完整检测：先本地，再 AI，并把两边结论合并成一份报告。

        参数:
            code:     源码文本。
            language: 标准语言标识。
            filename: 文件名。
            session_id: 网页带来的会话号（`X-Session-Id`）。给了它就会用
                **用户自己存的 Key** 去调模型，而不是 `.env` 里那把。
            model_id: 网页下拉框选中的模型 ID（如 `deepseek`）。
            api_key: 本次请求直接带来的 Key（优先级最高）。

        返回:
            `CheckOutcome`。**AI 不可用时也会正常返回**，只是
            `ai_available=False`、`note` 里说明原因（含"请先在网页上输入 API Key"），
            问题清单只有本地结论。

        异常:
            CodeCheckError: 代码为空或过长。

        关键逻辑（为什么 AI 失败不算致命错误）:
            阶段 1 已经给出了语法错误的精确位置和风格/风险清单，
            这些恰恰是初学者最需要的。如果因为"没配 API Key"就返回 503，
            等于把已经算出来的结果扔掉。因此这里把 AI 缺失降级为
            "少一部分结论 + 一句明确提示"，而不是整体失败。

            另外：**密钥由 `llm_client_for_request` 统一解析**——
            网页上填了就优先用网页的，没填才回落到 `.env` 的默认模型。
        """
        if not code.strip():
            raise CodeCheckError("代码内容为空，没有可检测的内容")
        if len(code) > MAX_CODE_CHARS:
            raise CodeCheckError(
                f"代码过长（{len(code)} 字符），上限为 {MAX_CODE_CHARS} 字符"
            )

        started = time.perf_counter()
        # ---- 阶段 1 ----
        local = self.local_check(code, language=language, filename=filename)

        outcome = CheckOutcome(filename=filename, language=language, local=local)
        # 本地发现的问题直接进报告：即使后面 AI 挂了，学生也看得到这些结论
        _split_into_buckets(outcome, local.issues)

        # ---- 阶段 2：先确定这次用哪个模型 / 哪把 Key ----
        async with llm_client_for_request(
            self._settings.llm,
            session_id=session_id,
            model_id=model_id,
            api_key=api_key,
            fallback=self._llm,
        ) as llm:
            await self._run_ai_stage(
                code, language=language, filename=filename, local=local,
                outcome=outcome, llm=llm, started=started,
            )
        return outcome

    async def _run_ai_stage(
        self,
        code: str,
        *,
        language: str,
        filename: str,
        local: LocalCheckResult,
        outcome: CheckOutcome,
        llm: LLMClient,
        started: float,
    ) -> None:
        """跑 AI 深度检测阶段，并把结果合并进 `outcome`（原地修改）。

        单独抽出来是因为"选客户端"要放在 `async with` 里：
        用户 Key 用完即关，不能把连接池留在全局。
        """
        # ---- 没配 Key（页面上没填、后端也没有）：给出那句明确提示，跳过 AI ----
        if not llm.configured:
            outcome.note = MISSING_API_KEY_NOTE.format(extra="检测")
            outcome.warnings.append(f"{MISSING_API_KEY_HINT}，已跳过 AI 深度检测")
            _apply_local_score(outcome)
            outcome.duration_ms = (time.perf_counter() - started) * 1000
            return

        outcome.ai_available = True
        try:
            payload, usage, model, ai_duration = await self._ask_ai(
                code, language=language, filename=filename, local=local, llm=llm
            )
        except Exception as exc:  # noqa: BLE001 - AI 失败也要给出本地结论
            logger.warning("AI 深度检测失败，仅返回本地结论: %s", exc, exc_info=True)
            outcome.ai_available = False
            outcome.note = f"AI 深度检测调用失败（{exc}），以下仅为本地静态检查结果。"
            outcome.warnings.append(f"AI 调用失败：{exc}")
            _apply_local_score(outcome)
            outcome.duration_ms = (time.perf_counter() - started) * 1000
            return

        outcome.usage = usage
        outcome.model = model
        outcome.ai_duration_ms = ai_duration
        self._merge_ai(outcome, payload)

        outcome.duration_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "代码检测完成: 文件=%s 语言=%s 评分=%s 错误=%s 风格=%s 风险=%s AI=%s",
            filename, language, outcome.score,
            len(outcome.errors), len(outcome.style), len(outcome.risks),
            outcome.ai_available,
        )

    async def _ask_ai(
        self,
        code: str,
        *,
        language: str,
        filename: str,
        local: LocalCheckResult,
        llm: LLMClient | None = None,
    ) -> tuple[dict[str, Any], LLMUsage, str, float]:
        """调用大模型做深度检测，返回 (解析后的 JSON, 用量, 模型名, 耗时)。

        参数:
            llm: 本次请求要用的客户端（由 `check()` 按用户选的模型/Key 传进来）；
                 不传则用构造时注入的那个（老行为）。

        关键逻辑:
            把本地结论（`local.findings_text()`）同时写进系统提示与用户消息，
            并要求模型不要重复报告——这是"省钱又更深"的关键。
            模型输出不守规矩时用 `extract_json_object` 三级降级兜底。
        """
        client = llm or self._llm
        findings = local.findings_text()
        messages = build_messages(
            code, language=language, filename=filename, local_findings=findings
        )

        # 注意：trace_span 是**同步**上下文管理器（见 app/core/trace.py），
        # 所以这里用 with 而不是 async with，内部的 await 仍然在事件循环里跑。
        with trace_span(
            "service.code_checker",
            kind="agent",
            payload={
                "language": language,
                "filename": filename,
                "code_chars": len(code),
                "prompt_version": PROMPT_VERSION,
            },
            metadata={"model": client.model},
        ) as span:
            response = await client.chat(messages)
            payload = extract_json_object(response.content)
            if payload is None:
                span.set_metadata(parsed=False)
                raise CodeCheckError("模型返回的内容不是合法 JSON")
            span.set_metadata(parsed=True, model=response.model)
            span.set_output({"keys": sorted(payload.keys())})
            return payload, response.usage, response.model or client.model, response.latency_ms

    # ------------------------------------------------------------------
    # 合并
    # ------------------------------------------------------------------
    def _merge_ai(self, outcome: CheckOutcome, payload: dict[str, Any]) -> None:
        """把模型的 JSON 结论并进报告，并做一轮字段清洗。

        参数:
            outcome: 已有本地结论的报告对象（原地修改）。
            payload: 模型返回的 JSON。

        关键逻辑:
            模型输出永远不能直接信，这里做四件事：
              1. 四类字段各自校验类型，非列表一律当空；
              2. severity 只允许三档，其它值收敛为 info；
              3. 行号必须是整数且在代码范围内，否则置 None（模型很爱编行号）；
              4. 分类字段由我们按桶赋值，不接受模型自己写的 category。
        """
        for key, bucket, category in (
            ("errors", "errors", "error"),
            ("style", "style", "style"),
            ("risks", "risks", "risk"),
        ):
            raw_items = payload.get(key)
            if not isinstance(raw_items, list):
                continue
            # 分类由我们按"桶"赋值，不接受模型自己写的 category：
            # 它经常把风格问题塞进 errors 里，那会让学生以为代码跑不起来。
            # errors 桶再细分一次：有语法错误时归为 syntax，否则是逻辑问题。
            bucket_category: Category = category  # type: ignore[assignment]
            if key == "errors":
                bucket_category = "syntax" if outcome.local.syntax_error else "logic"

            parsed: list[Issue] = []
            for item in raw_items:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title") or "").strip()
                if not title:
                    continue
                parsed.append(
                    Issue(
                        title=title,
                        detail=str(item.get("detail") or "").strip(),
                        suggestion=str(item.get("suggestion") or "").strip(),
                        line=_safe_line(item.get("line"), outcome.local.metrics.total_lines),
                        severity=_safe_severity(item.get("severity")),
                        category=bucket_category,
                        source="ai",
                    )
                )
            getattr(outcome, bucket).extend(parsed)

        outcome.advice = _string_list(payload.get("advice"))
        outcome.highlights = _string_list(payload.get("highlights"))
        outcome.summary = str(payload.get("summary") or "").strip()

        # ---- 评分：以模型的分数为基础，用本地检查校准 ----
        # 注意用"可选"解析：模型偶尔会漏掉 score 字段，
        # 那种情况应该退回本地评分，而不是当成 0 分把学生吓一跳。
        outcome.ai_score = _safe_optional_float(payload.get("score"))
        if outcome.ai_score is None:
            outcome.warnings.append("模型没有给出评分，已改用本地规则评分")
        _apply_local_score(outcome)

    # ------------------------------------------------------------------
    # 便捷入口
    # ------------------------------------------------------------------
    async def check_text(
        self,
        code: str,
        *,
        language: str | None,
        filename: str | None,
        session_id: str | None = None,
        model_id: str | None = None,
        api_key: str | None = None,
    ) -> CheckOutcome:
        """面向接口的入口：先归一语言与文件名，再走完整检测。

        参数:
            code:     源码文本。
            language: 用户传入的语言（可为空，会按文件名推断）。
            filename: 文件名（可为空，会按语言给个默认名）。
            session_id / model_id / api_key: 见 `check()`——网页上填的 Key 从这里进来。

        异常:
            UnsupportedLanguageError: 语言不支持（调用方转 400）。
        """
        resolved_filename = (filename or "").strip() or f"snippet.{_default_suffix(language)}"
        resolved_language = normalize_language(language, resolved_filename)
        return await self.check(
            code,
            language=resolved_language,
            filename=resolved_filename,
            session_id=session_id,
            model_id=model_id,
            api_key=api_key,
        )


def _default_suffix(language: str | None) -> str:
    """语言为空时给文件名一个合理的默认后缀，便于展示与推断。"""
    text = (language or "").strip().lower()
    return {"c": "c", "java": "java", "python": "py"}.get(text, "py")


# ---------------------------------------------------------------------------
# 小工具：字段清洗
# ---------------------------------------------------------------------------
def _safe_severity(value: Any) -> Severity:
    """把模型给的严重程度收敛到三档合法值。"""
    text = str(value or "").strip().lower()
    if text in ("error", "warning", "info"):
        return text  # type: ignore[return-value]
    return "info"


def _safe_line(value: Any, total_lines: int) -> int | None:
    """校验行号：必须是落在代码范围内的整数，否则返回 None。

    模型编造行号是高频问题（它会"顺手"给一个看起来合理的数字），
    而这个数字会直接显示给学生并指错地方，所以宁可置空也不能带病展示。
    """
    if isinstance(value, bool):  # bool 是 int 的子类，单独挡掉
        return None
    if isinstance(value, int) and 1 <= value <= max(total_lines, 1):
        return value
    return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    """把任意值安全地转成 float 并夹到 0-100。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(100.0, number))


def _safe_optional_float(value: Any) -> float | None:
    """把任意值转成 0-100 的 float；不是数字时返回 None。

    与 `_safe_float` 的区别：这里区分"模型给了 0 分"和"模型压根没给分"。
    前者是有效结论，后者要退回本地评分。
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(100.0, number))


def _string_list(value: Any, limit: int = 8) -> list[str]:
    """把模型给的数组收敛成"非空字符串列表"，并限制条数。"""
    if not isinstance(value, list):
        return []
    items = [str(item).strip() for item in value if str(item).strip()]
    return items[:limit]


def _dedupe_issues(issues: list[Issue]) -> list[Issue]:
    """去掉重复的本地问题。

    典型的重复来源：同一行既触发"缩进混用"又触发"命名不规范"，
    或者模式匹配在多行命中同一种危险函数。按 (标题, 行号) 去重即可。
    """
    seen: set[tuple[str, int | None]] = set()
    unique: list[Issue] = []
    for item in issues:
        key = (item.title, item.line)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _split_into_buckets(outcome: CheckOutcome, issues: list[Issue]) -> None:
    """把本地问题按 category 分到 errors / style / risks 三个桶里。"""
    for item in issues:
        if item.category in ("syntax", "logic"):
            outcome.errors.append(item)
        elif item.category == "risk":
            outcome.risks.append(item)
        else:
            outcome.style.append(item)


def _apply_local_score(outcome: CheckOutcome) -> None:
    """决定最终评分，并给出一句"为什么是这个分"。

    规则（刻意做得简单、可解释，学生能对上号）：
        1. 有 AI 分数时以它为基础；
        2. **有语法错误时把分数压到 45 以下**——代码都跑不起来，
           谈风格和可读性没有意义，这也是防止模型"语法错误还给 80 分"；
        3. 没有 AI 时用本地规则打分：100 分起，语法错误 -55、
           每个 error 级问题 -8、每个风险 -4、每个风格问题 -2，最低 20 分；
           再**封顶到 85 分**（见下面的说明）。

    为什么本地分要封顶到 85：
        本地规则查不出"逻辑对不对"——`for i in range(len(a) + 1)` 这种越界循环，
        静态规则看不出问题。实测一份带着真实逻辑错误的代码，纯本地会给到 92 分
        （优秀），这对学生是误导。因此本地模式不给优秀档，并在评分依据里
        说明"逻辑层面未检查"，鼓励配好模型再看一次。

    为什么要自己算一套本地分：
        没配模型时接口依然要给一个分数，且这个分数必须**可解释**——
        上表就是它的全部规则，学生问"为什么扣分"能答得上来。
    """
    if outcome.ai_score is not None:
        score = outcome.ai_score
        reason = "评分来自 AI 深度检测"
    else:
        error_count = len(outcome.errors)
        risk_count = len(outcome.risks)
        style_count = len(outcome.style)
        score = 100.0
        score -= 55 if outcome.local.syntax_error else 0
        score -= 8 * error_count
        score -= 4 * risk_count
        score -= 2 * style_count
        score = max(20.0, score)
        reason = (
            f"未启用 AI，按本地规则评分：100 分起，语法错误 -55、"
            f"错误 {error_count} 条 -{8 * error_count}、"
            f"风险 {risk_count} 条 -{4 * risk_count}、"
            f"风格 {style_count} 条 -{2 * style_count}"
        )
        if score > 85:
            score = 85.0
            reason += "；本地检查看不出逻辑对错，因此最高只给 85 分"

    # 语法不通过时封顶：这是硬规则，不交给模型自觉
    if outcome.local.syntax_error is not None and score > 45:
        score = 45.0
        reason += "；因存在语法错误，分数上限压到 45"

    outcome.score = round(score, 1)
    outcome.score_reason = reason


def locate_syntax_error(
    code: str, *, language: str, filename: str
) -> SyntaxErrorInfo | None:
    """定位语法错误，返回精确的行列号；没有语法错误返回 None。

    参数:
        code:     源码文本。
        language: 语言标识。
        filename: 文件名（Python 的 ast 解析需要，报错信息里会带上）。

    返回:
        `SyntaxErrorInfo` 或 None。

    关键逻辑:
        · Python：`ast.parse` 抛的 SyntaxError 自带 lineno/offset/msg，直接取用，
          这是最权威的位置信息；
        · C/Java：tree-sitter 是容错解析，不会抛异常，而是把出错的地方标成
          ERROR 节点、缺失的部分标成 MISSING。这里深度优先找**第一个**这样的
          节点，用它给出行号——比"语法树包含错误节点"这句话有用得多。
        · 语法包缺失时返回 None 并记日志：宁可说"没查出语法问题"，
          也不能因为环境缺包就把好代码判成错的。
    """
    if language == "python":
        try:
            ast.parse(code, filename=filename)
        except SyntaxError as exc:
            message = f"{exc.msg}"
            if exc.text and exc.text.strip():
                message += f"：{exc.text.strip()}"
            return SyntaxErrorInfo(
                line=exc.lineno,
                column=exc.offset,
                message=message,
                tool="ast",
                raw=f"SyntaxError: {exc.msg} (line {exc.lineno}, column {exc.offset})",
            )
        except ValueError as exc:  # 源码含 NUL 字节等
            return SyntaxErrorInfo(
                line=None, column=None, message=str(exc), tool="ast", raw=f"ValueError: {exc}"
            )
        return None

    # C / Java：tree-sitter
    if get_profile(language) is None or load_grammar(language) is None:
        logger.info("语法包不可用，跳过 %s 的语法检查", language)
        return None

    try:
        import tree_sitter

        parser = tree_sitter.Parser(load_grammar(language))
        tree = parser.parse(code.encode("utf-8", errors="replace"))
    except Exception:  # noqa: BLE001 - 语法包版本差异兜底
        logger.warning("tree-sitter 解析失败: %s", language, exc_info=True)
        return None

    node = _first_error_node(tree.root_node)
    if node is None:
        return None
    row, column = node.start_point
    kind = "缺失的内容" if node.is_missing else "无法解析的内容"
    return SyntaxErrorInfo(
        line=row + 1,
        column=column + 1,
        message=f"第 {row + 1} 行附近有{kind}（语法错误）",
        tool="tree-sitter",
        raw=f"tree-sitter: {node.type} at line {row + 1}, column {column + 1}",
    )


def _first_error_node(root: Any) -> Any | None:
    """在语法树里找第一个 ERROR / MISSING 节点。

    用队列做广度优先：先看浅层节点，因此拿到的是"最外层、最先出现"的那个错误
    —— 报错位置必须稳定可复现，随机挑一个子孙节点会让同一份代码两次报出不同行号。
    tree-sitter 容错解析的特点：出错处标成 ERROR 节点，缺失的部分标成 MISSING，
    两者都是我们要找的信号。
    """
    queue = deque([root])
    while queue:
        node = queue.popleft()
        if node.type == "ERROR" or node.is_missing:
            return node
        queue.extend(node.children)
    return None


__all__ = [
    "DEEP_CHECK_SYSTEM",
    "LANGUAGE_ALIASES",
    "MAX_CODE_CHARS",
    "PROMPT_VERSION",
    "STYLE_LIMITS",
    "SUPPORTED_LANGUAGES",
    "CheckOutcome",
    "CodeCheckError",
    "CodeChecker",
    "CodeMetrics",
    "Issue",
    "LocalCheckResult",
    "SyntaxErrorInfo",
    "UnsupportedLanguageError",
    "build_messages",
    "build_user_message",
    "check_naming",
    "check_python_ast_risks",
    "check_risk_patterns",
    "check_style_rules",
    "compute_metrics",
    "count_lines_by_kind",
    "language_label",
    "locate_syntax_error",
    "normalize_language",
]
