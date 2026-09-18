"""代码注释生成：为学生的代码补上「文件级 / 函数级 / 关键行内」三层中文注释。

===========================================================================
整体流程（三步，与「检测」「改错」保持一致的分层）
===========================================================================

    一段代码 + 语言
        │
        ├─ 第 1 步：本地分析（不联网，复用 code_checker 的本地检查）
        │    解析出文件结构：有哪些函数/方法、各自的起止行、参数与返回值签名，
        │    以及原代码里已经有几行注释。
        │    这份结构既喂给模型（告诉它"需要给哪几个函数写注释"），
        │    也是第 3 步检查"注释覆盖率"的依据。
        │
        ├─ 第 2 步：AI 生成带注释的完整代码
        │    Prompt 里把三层注释的**位置和格式**写死（见 FIX_COMMENT_SYSTEM）：
        │      · 文件级：说明这个文件是做什么的
        │      · 函数级：功能 + 参数 + 返回值，按语言规范（Python docstring /
        │        Java Javadoc / C 块注释）
        │      · 行内：只解释难懂的关键逻辑，不逐行加
        │
        └─ 第 3 步：本地复检（本模块最关键的把关）
             AI 很容易在"加注释"时顺手把代码改了、或者把三引号写坏。
             这里做四项客观检查：
               ① 生成的代码还能不能解析（语法是否被注释破坏）
               ② **代码逻辑有没有被改动**（这是最要命的一项）
                  · Python：比对 ast.dump（去掉 docstring 后），语义级判断
                  · C/Java：去掉注释、压缩空白后比对，能发现任何 token 变化
               ③ 每个函数是否都拿到了注释（覆盖率）
               ④ 文件级注释是否存在
             前两项不通过就必须明确告诉学生，绝不能把"被改过的代码"当成
             "只是加了注释"交回去。

===========================================================================
AI Prompt 的设计思路（完整模板见 COMMENT_SYSTEM，逐条说明见其上方注释）
===========================================================================

一句话：**让模型当一个"只写注释、不碰代码"的助教。**

十条约束：角色与受众 / 注入本地结构 / 三层注释各写在哪 / 语言规范对照 /
只加注释不改代码 / 不逐行注释 / 讲"为什么"而非复述语法 / 不编造参数含义 /
保留原有注释 / 严格 JSON + 反面示例。

===========================================================================
与项目里其它"注释"功能的区别
===========================================================================
  · `app/agents/tutor_agent.py` 的 `comment()`：前端「生成注释」按钮用的轻量版，
    只要求"加通俗中文注释"；
  · 本模块：把三层注释的位置与格式写进 Prompt，并在生成后**客观复检**
    （代码有没有被改、函数有没有漏注释、语法有没有被破坏），
    同时把原代码与带注释代码一起存库供对比。
"""

from __future__ import annotations

import ast
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.agents.extraction import extract_json_object
from app.core.config import Settings
from app.core.llm_client import (
    MISSING_API_KEY_HINT,
    MISSING_API_KEY_NOTE,
    LLMClient,
    LLMMessage,
    LLMUsage,
    llm_client_for_request,
)
from app.core.trace import trace_span
from app.services.code_checker import (
    MAX_CODE_CHARS,
    CodeChecker,
    LocalCheckResult,
    SyntaxErrorInfo,
    count_lines_by_kind,
    language_label,
    locate_syntax_error,
    normalize_language,
)

logger = logging.getLogger(__name__)

# Prompt 版本号：改动提示词时同步 +1，便于把"结果变差"归因到哪一版
PROMPT_VERSION = "comment-gen/v1"

# 各语言的注释规范说明——直接拼进 Prompt。
# 写成"按语言给出样例"而不是一句"遵循该语言规范"，
# 是因为后者模型理解得很含糊：实测不加样例时，Java 常常给不出 @param/@return，
# C 也经常只写成一行 //。
STYLE_GUIDE: dict[str, str] = {
    "python": (
        "Python 规范（PEP 257 + Google 风格）：\n"
        "  · 文件级：文件第一行用三引号文档字符串，例如\n"
        '      """学生成绩统计：计算平均分与最高分。"""\n'
        "  · 函数级：函数定义下一行用三引号文档字符串，写清功能、参数、返回值，例如\n"
        '      """求平均分。\\n\\n'
        "      Args:\\n          scores: 分数列表\\n\\n"
        '      Returns:\\n          平均分；列表为空时返回 0.0\\n      """\n'
        "      （没有参数就不写 Args，没有返回值就不写 Returns）\n"
        "  · 行内：用 # 号，写在被解释的那一行**上方**"
    ),
    "java": (
        "Java 规范（Javadoc）：\n"
        "  · 文件级：文件最上方用 /** ... */ 块注释说明这个文件做什么\n"
        "  · 方法级：方法定义**上方**用 /** ... */，写功能描述，并用标签说明参数与返回值，例如\n"
        "      /**\n"
        "       * 求最高分。\n"
        "       * @param scores 分数数组\n"
        "       * @return 最高分；数组为空时返回 0\n"
        "       */\n"
        "  · 行内：用 // 或 /* */，写在被解释的那一行**上方**"
    ),
    "c": (
        "C 规范：\n"
        "  · 文件级：文件最上方用 /* ... */ 块注释说明这个文件做什么\n"
        "  · 函数级：函数定义**上方**用 /* ... */ 块注释，写功能说明，"
        "参数与返回值各占一行（用 @param / @return 或中文「参数：」「返回：」都可以），例如\n"
        "      /*\n"
        "       * 冒泡排序：把数组从小到大排好序。\n"
        "       * 参数：arr 待排序的数组，n 元素个数\n"
        "       * 返回：无\n"
        "       */\n"
        "  · 行内：用 /* */ 或 //，写在被解释的那一行**上方**"
    ),
}


# ===========================================================================
# 一、AI Prompt
# ===========================================================================
# 设计思路逐条说明——每条都对应一个真实会翻车的点：
#
# 【1】角色 + 受众放最前面
#     "耐心的助教 + 读者是初学者"，模型的语言会从"文档工程师"切到"讲题"。
#
# 【2】注入本地解析出的文件结构
#     告诉它"这份代码里有哪几个函数、各自几行到几行"，
#     比让它自己数要可靠得多，也保证不会漏掉某个函数。
#     同时把"原代码已有几行注释"也告诉它，便于它保留而不是重写。
#
# 【3】三层注释必须写清"写在哪、写成什么样"
#     只说"加文件级注释"是不够的——实测模型会把文件级注释塞在 import 之后，
#     或者把 Java 的函数注释写成一行 //。所以 STYLE_GUIDE 里给了分语言的样例，
#     连"写在函数定义上方"这种位置都写死。
#
# 【4】只加注释，一个字符都不许改代码
#     这是本模块最硬的约束。实测模型非常爱"顺手"改掉一些东西：
#     把 for i in range(len(a)) 改成 for x in a、把变量名改得更"规范"、
#     甚至把明显的 bug 修掉。对"加注释"这个任务来说这些都是越界行为：
#     学生要的是"我的代码 + 注释"，不是"另一个人重写的代码"。
#     光靠 Prompt 约束不够，第 3 步还会用 AST 比对兜底。
#
# 【5】不要逐行加注释
#     不写这条，模型会给每一行都配一句"这是赋值语句"，代码被淹没。
#     明确要求"只在难懂的地方加"，并给出判断标准（复杂条件、易错边界、
#     不直观的算法步骤）。
#
# 【6】讲"为什么"而不是复述语法
#     在 Prompt 里直接给好/坏对照：
#       好："判断分数是否及格，60 分以上输出及格"
#       坏："if 语句，判断 score 是否大于等于 60"
#     模型对具体示例的敏感度远高于抽象要求。
#
# 【7】不编造参数含义
#     参数说明必须能从代码里推出来；推不出来就描述它的用途，
#     不要猜取值范围——编造的错误说明比没有注释更糟。
#
# 【8】保留学生原有的注释
#     那是他自己的思考痕迹，被覆盖掉会让人很恼火，也不利于复习。
#
# 【9】严格 JSON，代码里的换行用 \n 转义
#     真不听话时由 `extract_json_object` 三级降级兜底解析。
#
# 【10】给反面示例（不要写什么）
#     把"这是循环语句""定义了一个变量"这类废话点名禁止。
COMMENT_SYSTEM = """你是一位有耐心的编程入门课助教，正在给大一新生的代码写中文注释。

## 你的读者
一名刚开始学编程的学生，过两周回头看自己的代码时会忘掉当时的思路。
你的注释要能让他**重新看懂**这段代码，所以重点是讲清「这段在干什么、为什么这么写」。

## 这份代码的结构（系统已经解析出来了）
<<STRUCTURE>>

## 三层注释，位置和格式都要按要求来
### 1. 文件级注释（整个文件一份）
用一句话说明这个文件是做什么的、解决什么问题。
<<STYLE_GUIDE>>

### 2. 函数/方法级注释（上面结构里列出的每一个函数都要有）
说明三件事：这个函数做什么、参数分别是什么含义、返回什么。
一定要**按上面给出的语言规范写**（Python 用文档字符串、Java 用 Javadoc、C 用块注释）。

### 3. 关键逻辑行内注释（只加在难懂的地方）
判断标准（满足任意一条才加）：
  · 条件判断的逻辑不那么直观（例如边界值、多个条件组合）
  · 涉及边界处理的循环（起点、终点、是否包含末尾）
  · 算法步骤本身需要解释（例如交换、递归、累加的含义）
  · 容易看错的变量或单位

## 硬性要求（非常重要）
1. **只加注释，不要修改代码本身**：不要改变量名、不要调整代码结构、
   不要"顺手"修 bug、不要格式化代码。一个字都不要改。
2. **不要逐行加注释**：像 "x = 1  # 把 1 赋给 x" 这种复述语法的话不要写。
3. **讲"为什么"而不是复述语法**：
   好注释："判断分数是否及格，60 分以上算通过"
   坏注释："if 语句，判断 score 是否大于等于 60"
4. **保留学生原有的注释**，不要删掉，也不要改写他的原话。
5. **不要编造**：参数的含义必须能从代码里看出来；看不出来就描述它的用途，
   不要瞎猜取值范围。
6. 注释一律用**通俗的中文**，不要堆术语。

## 输出格式
只输出一个 JSON 对象，不要解释文字，不要用 markdown 代码块包裹：
{
  "commented_code": "加了注释的完整代码，换行用 \\n 表示",
  "summary": "一句话说明你在哪些地方加了注释
              （例如：加了文件说明、3 个函数的文档注释、5 处关键逻辑注释）",
  "coverage": {
    "file_comment": true,
    "functions_documented": ["average", "main"],
    "inline_comments": 5
  }
}

注意：`commented_code` 必须是**完整代码**（从第一行到最后一行），
不能省略任何部分，也不能写成 "…（其余不变）" 这种形式。
"""

# 提示词里的占位符。
# **不要用 str.format()**：上面的 JSON 示例含 `{` `}`，会被当成格式化字段并抛
# `KeyError`（在检测模块里实测踩过这个坑）。用不会与 JSON 冲突的记号 + replace。
STRUCTURE_PLACEHOLDER = "<<STRUCTURE>>"
STYLE_GUIDE_PLACEHOLDER = "<<STYLE_GUIDE>>"

# 判断"函数有没有注释"时，向上/向下看多少行。给 3 行是为了容忍
# `/** ... */` 多行块注释与 Python 的三引号文档字符串。
_COMMENT_LOOKBACK = 3


# ===========================================================================
# 二、数据结构
# ===========================================================================
@dataclass(slots=True)
class CommentVerification:
    """第 3 步本地复检的结论。"""

    syntax_ok: bool = False
    syntax_error: SyntaxErrorInfo | None = None
    # 代码逻辑有没有被改动。None 表示"无法判定"（例如原代码本身有语法错误，
    # 解析不了自然也就没法比对），这种情况必须如实说明，不能默认当成"没改"。
    code_unchanged: bool | None = None
    unchanged_note: str = ""
    # 注释覆盖情况
    functions_total: int = 0
    functions_covered: int = 0
    file_comment: bool = False
    # 注释行数
    comment_lines_before: int = 0
    comment_lines_after: int = 0
    inline_comment_lines: int = 0
    verified: bool = False
    note: str = ""

    @property
    def added_comment_lines(self) -> int:
        """新增的注释行数（可能为负——模型删掉了原有注释）。"""
        return self.comment_lines_after - self.comment_lines_before

    @property
    def coverage_ratio(self) -> float:
        """函数注释覆盖率，0-1；没有函数时视为 1（没有可漏的）。"""
        if self.functions_total == 0:
            return 1.0
        return self.functions_covered / self.functions_total

    def to_dict(self) -> dict[str, Any]:
        return {
            "syntax_ok": self.syntax_ok,
            "code_unchanged": self.code_unchanged,
            "unchanged_note": self.unchanged_note,
            "functions_total": self.functions_total,
            "functions_covered": self.functions_covered,
            "coverage_ratio": round(self.coverage_ratio, 4),
            "file_comment": self.file_comment,
            "comment_lines_before": self.comment_lines_before,
            "comment_lines_after": self.comment_lines_after,
            "added_comment_lines": self.added_comment_lines,
            "inline_comment_lines": self.inline_comment_lines,
            "verified": self.verified,
            "note": self.note,
        }


@dataclass(slots=True)
class CommentOutcome:
    """一次注释生成的完整结果。"""

    filename: str
    language: str
    original_code: str
    commented_code: str = ""
    summary: str = ""
    verification: CommentVerification = field(default_factory=CommentVerification)
    local: LocalCheckResult | None = None
    functions_total: int = 0
    ai_available: bool = False
    model: str = ""
    usage: LLMUsage = field(default_factory=LLMUsage)
    warnings: list[str] = field(default_factory=list)
    note: str = ""
    duration_ms: float = 0.0
    ai_duration_ms: float | None = None
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class CommentGeneratorError(RuntimeError):
    """注释生成过程中的可预期错误（调用方转 4xx）。"""


# ===========================================================================
# 三、CommentGenerator
# ===========================================================================
class CommentGenerator:
    """代码注释生成：本地分析 → AI 写注释 → 本地复检。"""

    # 类名不会被 pytest 误认成测试类；仍显式声明一次更保险
    __test__ = False

    def __init__(
        self, settings: Settings, llm: LLMClient, checker: CodeChecker | None = None
    ) -> None:
        self._settings = settings
        self._llm = llm
        # 复用检测模块的本地解析能力：文件结构（函数清单/签名/行号）与注释统计
        # 都从那里来，保证「检测」「改错」「注释」三个功能对同一份代码的理解一致。
        self._checker = checker or CodeChecker(settings, llm)

    @property
    def llm(self) -> LLMClient:
        return self._llm

    # ------------------------------------------------------------------
    # 第 1 步：本地分析
    # ------------------------------------------------------------------
    def analyze_locally(self, code: str, *, language: str, filename: str) -> LocalCheckResult:
        """本地分析：拿到函数/类清单、签名与已有注释的统计（不联网）。

        参数:
            code:     源码。
            language: 标准语言标识（已归一）。
            filename: 文件名。

        返回:
            `LocalCheckResult`（与检测模块同一结构，含 symbols 与 metrics）。
        """
        return self._checker.local_check(code, language=language, filename=filename)

    def describe_structure(self, local: LocalCheckResult, *, language: str) -> str:
        """把本地解析出的结构整理成一段文字，注入 AI 提示词。

        参数:
            local:    本地分析结果（要有 symbols）。
            language: 语言标识。

        返回:
            供模型阅读的结构说明。

        关键逻辑:
            写"哪个函数在第几行到第几行、签名是什么"，而不是只报函数名：
            模型据此知道该给谁写注释、参数有哪些（参数名直接从签名里读，
            不用它猜），从而显著减少"参数说明张冠李戴"的情况。
            没有函数时也明说一句，免得模型以为漏解析了。
        """
        metrics = local.metrics
        functions = [item for item in metrics.symbols if item["kind"] in ("function", "method")]
        classes = [item for item in metrics.symbols if item["kind"] == "class"]

        lines = [
            f"- 语言：{language_label(language)}",
            f"- 总行数：{metrics.total_lines}（其中已有注释 {metrics.comment_lines} 行）",
            f"- 函数/方法：{len(functions)} 个；类：{len(classes)} 个",
        ]
        if functions:
            lines.append("- 需要写函数级注释的清单：")
            for item in functions:
                lines.append(
                    f"    · {item['name']}（{item['kind']}，第 {item['start_line']}-"
                    f"{item['end_line']} 行）签名：{_signature_of(item)}"
                )
        else:
            lines.append("- 这份代码里没有定义函数，只需要写文件级注释和关键行内注释")
        if metrics.comment_lines:
            lines.append(
                f"- 注意：代码里已经有 {metrics.comment_lines} 行注释，请原样保留，不要改写"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 第 3 步：本地复检
    # ------------------------------------------------------------------
    def verify(
        self,
        original: str,
        commented: str,
        *,
        language: str,
        filename: str,
        local: LocalCheckResult,
    ) -> CommentVerification:
        """复检 AI 生成的结果：语法、代码一致性、注释覆盖率。

        参数:
            original:  学生原来的代码。
            commented: AI 返回的带注释代码。
            language:  语言标识。
            filename:  文件名。
            local:     本地分析结果（提供函数清单与原有注释行数）。

        返回:
            `CommentVerification`。

        关键逻辑（四项检查，逐项说明"能查出什么、查不出什么"）:
            ① 语法：用与检测模块相同的解析器。注释写坏三引号、
               把 /** */ 嵌套进字符串等都会在这里暴露。
            ② **代码一致性**（最重要）：
               · Python 用 `ast.dump` 比对，并且先剥掉文档字符串——
                 因为"加 docstring"正是本任务的目标，它不该被判成"改了代码"。
                 AST 比对是**语义级**的：改变量名、调换语句顺序、
                 改动常量都会被查出来；只改缩进风格不会（那不算改逻辑）。
               · C/Java 没有现成的 AST 比对工具链，改为"去掉注释 +
                 压缩空白后逐字符比对"：能查出任何 token 层面的改动，
                 但查不出纯空白/换行的重排（在 C/Java 里那不影响语义）。
               · 原代码本身有语法错误时无法比对，返回 None 并说明原因——
                 不能默默当成"没改"。
            ③ 函数注释覆盖率：对原代码里的每个函数，检查生成后的代码中
               函数定义前是否紧邻注释（C/Java），或函数体内首行是否是三引号
               文档字符串（Python）。漏掉的函数会列出来。
            ④ 文件级注释：Python 看模块文档字符串；C/Java 看文件开头是否为注释。

            `verified` 的口径：语法通过 + 代码没有被判定为改动过 +
            **确实新增了注释**。三项缺一不可，否则学生可能拿到一份
            "什么都没变"或"被偷偷改过"的结果。
        """
        check = CommentVerification()

        # ---- ① 语法 ----
        check.syntax_error = locate_syntax_error(commented, language=language, filename=filename)
        check.syntax_ok = check.syntax_error is None
        if not check.syntax_ok:
            where = f"第 {check.syntax_error.line} 行" if check.syntax_error.line else "未知位置"
            check.note = (
                f"生成的代码在 {where} 处有语法错误，注释可能没有被正确闭合"
                "（例如三引号或多行注释没有配对）。建议重新生成一次。"
            )
            return check

        # ---- ② 代码一致性 ----
        check.code_unchanged, check.unchanged_note = compare_code(original, commented, language)

        # ---- ③ 函数注释覆盖率 ----
        functions = [
            item
            for item in local.metrics.symbols
            if item["kind"] in ("function", "method")
        ]
        check.functions_total = len(functions)
        covered, missing = count_covered_functions(
            commented, functions, language=language, filename=filename
        )
        check.functions_covered = covered

        # ---- ④ 文件级注释 + 注释行数 ----
        check.file_comment = has_file_comment(commented, language)
        check.comment_lines_before = local.metrics.comment_lines
        _, comment_after, _ = count_lines_by_kind(commented, language)
        check.comment_lines_after = comment_after
        # 第三层注释：既不是文件级、也不是函数级/文档字符串的那些（两种写法都算）
        check.inline_comment_lines = count_inline_comments(
            commented, language, filename=filename
        )

        # ---- 汇总 ----
        problems: list[str] = []
        if check.code_unchanged is False:
            problems.append("生成的代码**改动了原有逻辑**（本任务只应加注释）")
        if missing:
            problems.append(f"有 {len(missing)} 个函数没写注释：{'、'.join(missing[:5])}")
        if not check.file_comment:
            problems.append("缺少文件级注释")
        if check.added_comment_lines <= 0:
            problems.append("没有新增注释行")

        check.verified = (
            check.syntax_ok
            and check.code_unchanged is not False
            and check.added_comment_lines > 0
        )

        if check.verified and not problems:
            check.note = (
                f"注释已加好：新增 {check.added_comment_lines} 行注释，"
                f"{check.functions_covered}/{check.functions_total} 个函数有注释，"
                f"原有代码未被改动。"
            )
        elif check.verified:
            check.note = "注释已加好，但有几处可以再完善：" + "；".join(problems)
        else:
            check.note = "生成结果有问题：" + "；".join(problems)
        return check

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    async def generate(
        self,
        code: str,
        *,
        language: str,
        filename: str,
        session_id: str | None = None,
        model_id: str | None = None,
        api_key: str | None = None,
    ) -> CommentOutcome:
        """完整流程：本地分析 → AI 写注释 → 本地复检。

        参数:
            code:     源码。
            language: 标准语言标识。
            filename: 文件名。
            session_id: 网页带来的会话号（`X-Session-Id`）：给了就用
                **用户自己存的 Key** 调模型，而不是 `.env` 里那把。
            model_id: 网页上选的模型 ID。
            api_key: 本次请求直接带来的 Key（优先级最高）。

        返回:
            `CommentOutcome`。AI 不可用时**不抛异常**，返回
            `ai_available=False` + 原文（保证学生至少能拿回自己的代码）+ 明确提示
            （含"请先在网页上输入 API Key"）。

        异常:
            CommentGeneratorError: 代码为空或过长。

        关键逻辑:
            与其它模块一致的取舍：AI 挂了也不能把学生的代码弄丢，
            所以 `commented_code` 默认就是原代码，只有 AI 成功返回且通过复检
            才会被替换。这样任何失败路径下，学生复制到的都是能跑的代码。
        """
        if not code.strip():
            raise CommentGeneratorError("代码内容为空，没有可加注释的内容")
        if len(code) > MAX_CODE_CHARS:
            raise CommentGeneratorError(
                f"代码过长（{len(code)} 字符），上限为 {MAX_CODE_CHARS} 字符"
            )

        started = time.perf_counter()
        # ---- 第 1 步 ----
        local = self.analyze_locally(code, language=language, filename=filename)
        functions = [
            item for item in local.metrics.symbols if item["kind"] in ("function", "method")
        ]

        outcome = CommentOutcome(
            filename=filename,
            language=language,
            original_code=code,
            # 默认返回原代码：任何失败路径下都不能把学生的代码弄丢
            commented_code=code,
            local=local,
            functions_total=len(functions),
        )

        # ---- 第 2 步：先确定"这次用哪个模型 / 哪把 Key"，再调 AI ----
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
        outcome: CommentOutcome,
        llm: LLMClient,
        started: float,
    ) -> None:
        """跑 AI 注释生成 + 本地复检，把结果写进 `outcome`（原地修改）。

        抽成单独方法是为了让"选客户端"留在 `async with` 里：用户 Key 用完即关。
        """
        # ---- 没有可用 Key（页面没填、后端也没配）：给出那句明确提示，跳过 AI ----
        if not llm.configured:
            outcome.note = MISSING_API_KEY_NOTE.format(extra="注释生成")
            outcome.warnings.append(f"{MISSING_API_KEY_HINT}，已跳过 AI 注释生成")
            outcome.verification = self.verify(
                code, code, language=language, filename=filename, local=local
            )
            outcome.duration_ms = (time.perf_counter() - started) * 1000
            return

        outcome.ai_available = True
        try:
            payload, usage, model, latency = await self._ask_ai(
                code, language=language, filename=filename, local=local, llm=llm
            )
        except Exception as exc:  # noqa: BLE001 - AI 失败也要把原代码还给学生
            logger.warning("AI 注释生成失败，返回原代码: %s", exc, exc_info=True)
            outcome.ai_available = False
            outcome.note = f"AI 注释生成调用失败（{exc}），已返回你的原代码。"
            outcome.warnings.append(f"AI 调用失败：{exc}")
            outcome.verification = self.verify(
                code, code, language=language, filename=filename, local=local
            )
            outcome.duration_ms = (time.perf_counter() - started) * 1000
            return

        outcome.usage = usage
        outcome.model = model
        outcome.ai_duration_ms = latency

        # ---- 第 3 步 ----
        self._merge_ai(outcome, payload)
        outcome.verification = self.verify(
            code,
            outcome.commented_code,
            language=language,
            filename=filename,
            local=local,
        )
        self._cross_check(outcome)

        outcome.duration_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "注释生成完成: 文件=%s 语言=%s 新增注释=%s 行 覆盖率=%s/%s 代码未改=%s",
            filename, language, outcome.verification.added_comment_lines,
            outcome.verification.functions_covered, outcome.verification.functions_total,
            outcome.verification.code_unchanged,
        )

    async def generate_text(
        self,
        code: str,
        *,
        language: str | None,
        filename: str | None,
        session_id: str | None = None,
        model_id: str | None = None,
        api_key: str | None = None,
    ) -> CommentOutcome:
        """面向接口的入口：先归一语言与文件名，再走完整流程。

        参数:
            code:     源码。
            language: 用户传入的语言（可为空，按文件名推断）。
            filename: 文件名（可为空，给个默认名）。
            session_id / model_id / api_key: 见 `generate()`（网页填的 Key 从这些参数进来）。

        异常:
            UnsupportedLanguageError: 语言不支持（调用方转 400）。
        """
        resolved_filename = (filename or "").strip() or f"snippet.{_default_suffix(language)}"
        resolved_language = normalize_language(language, resolved_filename)
        return await self.generate(
            code,
            language=resolved_language,
            filename=resolved_filename,
            session_id=session_id,
            model_id=model_id,
            api_key=api_key,
        )

    # ------------------------------------------------------------------
    # AI 调用与字段清洗
    # ------------------------------------------------------------------
    async def _ask_ai(
        self,
        code: str,
        *,
        language: str,
        filename: str,
        local: LocalCheckResult,
        llm: LLMClient | None = None,
    ) -> tuple[dict[str, Any], LLMUsage, str, float]:
        """调用模型生成带注释的代码，返回 (解析后的 JSON, 用量, 模型名, 耗时)。

        参数:
            llm: 本次请求要用的客户端（由 `generate()` 按用户选的模型/Key 传进来）；
                 不传则用构造时注入的那个（老行为）。
        """
        client = llm or self._llm
        messages = build_messages(
            code,
            language=language,
            filename=filename,
            structure=self.describe_structure(local, language=language),
        )

        # trace_span 是同步上下文管理器（见 app/core/trace.py），所以用 with
        with trace_span(
            "service.comment_generator",
            kind="agent",
            payload={
                "language": language,
                "filename": filename,
                "code_chars": len(code),
                "functions": len(local.metrics.symbols),
                "prompt_version": PROMPT_VERSION,
            },
            metadata={"model": client.model},
        ) as span:
            response = await client.chat(messages)
            payload = extract_json_object(response.content)
            if payload is None:
                span.set_metadata(parsed=False)
                raise CommentGeneratorError("模型返回的内容不是合法 JSON")
            span.set_metadata(parsed=True, model=response.model)
            span.set_output({"keys": sorted(payload.keys())})
            return payload, response.usage, response.model or client.model, response.latency_ms

    @staticmethod
    def _merge_ai(outcome: CommentOutcome, payload: dict[str, Any]) -> None:
        """把模型的 JSON 结果并进报告，并做字段清洗。

        参数:
            outcome: 已有本地分析的报告对象（原地修改）。
            payload: 模型返回的 JSON。

        关键逻辑:
            模型输出永远不能直接信，这里做四件事：
              1. `commented_code` 必须是非空字符串，且**行数不能明显少于原文**——
                 模型偶尔会用 "…（其余不变）" 省略中间部分，这种残缺代码
                 交回去等于毁掉学生的文件，必须拦下来并退回原代码；
              2. `summary` 去空白；
              3. 模型自称的 `coverage` 只做参考（真值由第 3 步本地复检给出），
                 不直接采用——自评往往偏乐观；
              4. 记录一条"模型自评 vs 本地复检"的差异，供排查提示词问题。
        """
        commented = payload.get("commented_code")
        if not isinstance(commented, str) or not commented.strip():
            outcome.warnings.append("模型没有返回带注释的代码，已回退为原代码")
            outcome.commented_code = outcome.original_code
            return

        # 残缺检测：生成结果比原文少了 30% 以上的行，基本可以判定是省略写法
        original_lines = len(outcome.original_code.splitlines())
        commented_lines = len(commented.splitlines())
        if original_lines >= 5 and commented_lines < original_lines * 0.7:
            outcome.warnings.append(
                f"模型返回的代码只有 {commented_lines} 行（原文 {original_lines} 行），"
                "疑似省略了部分内容，已回退为原代码"
            )
            outcome.commented_code = outcome.original_code
            outcome.summary = str(payload.get("summary") or "").strip()
            return

        outcome.commented_code = commented
        outcome.summary = str(payload.get("summary") or "").strip()

        # 模型自评的覆盖率：只用于对照，不作为结论
        claimed = payload.get("coverage")
        if isinstance(claimed, dict):
            documented = claimed.get("functions_documented")
            claimed_count = len(documented) if isinstance(documented, list) else None
            if claimed_count is not None and claimed_count != outcome.functions_total:
                logger.info(
                    "模型自评函数注释数 %s，实际函数数 %s（以本地复检为准）",
                    claimed_count, outcome.functions_total,
                )

    @staticmethod
    def _cross_check(outcome: CommentOutcome) -> None:
        """把复检发现的问题明确写成给学生看的提示。

        参数:
            outcome: 已完成复检的报告对象（原地修改）。

        关键逻辑:
            最需要点破的是"代码被改动了"：学生要的是"我的代码 + 注释"，
            如果模型顺手改了他的逻辑，而他没发现，之后照着这份代码去理解
            自己的程序，学到的东西就是错的。所以这条必须进 warnings 并写进 note。
        """
        check = outcome.verification
        if check.code_unchanged is False:
            outcome.warnings.append(
                "生成的代码与你的原代码在逻辑上有差异（模型可能顺手改了代码），"
                "已在校验结果中标注，建议只复制注释部分，或重新生成一次"
            )
            outcome.note = (
                outcome.note
                or "注意：AI 返回的代码改动了原有逻辑，本功能应当只添加注释。"
            )
        if check.code_unchanged is None and check.unchanged_note:
            outcome.warnings.append(check.unchanged_note)

        if check.functions_total and check.functions_covered < check.functions_total:
            missing = check.functions_total - check.functions_covered
            outcome.warnings.append(f"有 {missing} 个函数没有拿到注释，可重新生成一次")


# ---------------------------------------------------------------------------
# Prompt 构造
# ---------------------------------------------------------------------------
def build_user_message(
    code: str, *, language: str, filename: str, structure: str
) -> str:
    """构造「用户消息」：这次要加注释的代码 + 结构清单再提醒一次。

    参数:
        code:      学生代码原文（原样放进围栏，不做任何改写）。
        language:  语言标识，决定代码围栏与注释规范。
        filename:  文件名。
        structure: 本地解析出的结构说明。

    返回:
        拼接好的用户消息。

    关键逻辑:
        结尾再贴一次函数清单，是为了让模型"照着单子干活"——
        实测只把清单放在系统提示里时，模型偶尔会漏掉最后一个函数。
    """
    return (
        f"请给下面这段 {language_label(language)} 代码加上中文注释"
        f"（文件名：{filename}）。\n\n"
        f"```{language}\n{code}\n```\n\n"
        f"再次确认需要写函数级注释的清单（一个都不要漏）：\n{structure}\n\n"
        "请记住：只加注释，不要修改代码本身。"
    )


def build_messages(
    code: str, *, language: str, filename: str, structure: str
) -> list[LLMMessage]:
    """组装完整消息列表（系统 + 用户）。

    单独抽成函数：测试里可以直接断言"提示词里确实包含了函数清单"
    "确实要求了只加注释不改代码"，而不必真的去调模型。
    """
    style_guide = STYLE_GUIDE.get(language, STYLE_GUIDE["python"])
    system = COMMENT_SYSTEM.replace(STRUCTURE_PLACEHOLDER, structure).replace(
        STYLE_GUIDE_PLACEHOLDER, style_guide
    )
    return [
        LLMMessage(role="system", content=system),
        LLMMessage(
            role="user",
            content=build_user_message(
                code, language=language, filename=filename, structure=structure
            ),
        ),
    ]


# ===========================================================================
# 四、本地复检用到的纯函数（单独拆出来便于测试）
# ===========================================================================
def compare_code(
    original: str, commented: str, language: str
) -> tuple[bool | None, str]:
    """比对两份代码在逻辑上是否一致（忽略注释与空白）。

    参数:
        original:  原代码。
        commented: 加了注释的代码。
        language:  语言标识。

    返回:
        `(是否一致, 说明)`。一致返回 True，有差异返回 False，
        **无法判定时返回 None**（例如原代码本身有语法错误）。

    关键逻辑:
        · Python：用 `ast.dump` 做**语义级**比对，并且先剥掉文档字符串
          （加 docstring 正是本任务的目标，不该算改动）。
          能查出：改变量名、改常量、调换语句顺序、删掉一句；
          查不出：缩进风格、空行、行内注释位置——这些不影响逻辑，不该报错。
        · C/Java：没有现成的 AST 比对，改为"去掉注释 + 压缩连续空白"后
          逐字符比对。能查出任何 token 改动；
          查不出纯换行/缩进的重排（在 C/Java 里不影响语义）。
    """
    if language == "python":
        return _compare_python(original, commented)
    return _compare_token_stream(original, commented, language)


def _compare_python(original: str, commented: str) -> tuple[bool | None, str]:
    """Python 的语义级比对（ast.dump，剥掉文档字符串）。"""
    try:
        before = ast.dump(_strip_docstrings(ast.parse(original)))
    except (SyntaxError, ValueError) as exc:
        return None, f"原代码本身无法解析（{type(exc).__name__}），无法自动确认代码是否被改动"
    try:
        after = ast.dump(_strip_docstrings(ast.parse(commented)))
    except (SyntaxError, ValueError) as exc:
        return False, f"生成的代码无法解析（{type(exc).__name__}），代码可能被破坏"

    if before == after:
        return True, "已比对语法树：除注释外，代码逻辑与原文完全一致"
    return (
        False,
        "已比对语法树：生成的代码与原文在**逻辑上有差异**（不只是多了注释）",
    )


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    """剥掉模块/类/函数的第一条字符串字面量（文档字符串）。

    参数:
        tree: 已解析的语法树。

    返回:
        同一个树对象（原地修改）。

    关键逻辑:
        文档字符串在 AST 里是一个普通的 `Expr(Constant(str))` 节点，
        直接比对会把"加了 docstring"误判成"改了代码"。
        这里把这类节点去掉，只留下真正的逻辑语句。
        注意只剥**第一条**：函数体中间的裸字符串可能是有意义的（虽然少见），
        不做处理更安全。
    """
    holders = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if not isinstance(node, holders):
            continue
        target = node.body
        if not target:
            continue
        first = target[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            del target[0]
    return tree


def _strip_comments(code: str, language: str) -> str:
    """去掉注释，但**保留字符串字面量**（字符串里的 // 不是注释）。

    参数:
        code:     源码。
        language: 语言标识。

    返回:
        去掉注释后的代码。

    关键逻辑:
        逐字符扫描并跟踪"是否在字符串里"，与检测模块的掩码函数思路一致，
        但这里是把注释**整段删除**而不是替换成空格——因为后续要压缩空白比对，
        留着空格反而干扰。
        Python 用 `#` 到行尾；C/Java 用 `//` 到行尾与 `/* ... */` 块注释。
    """
    if language == "python":
        return "\n".join(_strip_python_line_comment(row) for row in code.splitlines())

    out: list[str] = []
    in_block = False
    for row in code.splitlines():
        chars: list[str] = []
        index = 0
        in_string: str | None = None
        while index < len(row):
            char = row[index]
            two = row[index : index + 2]
            if in_block:
                if two == "*/":
                    in_block = False
                    index += 2
                    continue
                index += 1
                continue
            if in_string:
                chars.append(char)
                if char == "\\":                 # 转义：下一个字符原样保留
                    if index + 1 < len(row):
                        chars.append(row[index + 1])
                    index += 2
                    continue
                if char == in_string:
                    in_string = None
                index += 1
                continue
            if two == "//":
                break                            # 行注释：本行到此为止
            if two == "/*":
                in_block = True
                index += 2
                continue
            if char in "\"'":
                in_string = char
            chars.append(char)
            index += 1
        out.append("".join(chars))
    return "\n".join(out)


def _strip_python_line_comment(row: str) -> str:
    """去掉一行 Python 代码里的注释部分（跳过字符串内部）。"""
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
            return row[:index]
        index += 1
    return row


def _compare_token_stream(
    original: str, commented: str, language: str
) -> tuple[bool | None, str]:
    """C/Java 的比对：去注释 + 压缩空白后逐字符比较。"""
    before = re.sub(r"\s+", " ", _strip_comments(original, language)).strip()
    after = re.sub(r"\s+", " ", _strip_comments(commented, language)).strip()

    if not before:
        return None, "原代码去掉注释后没有内容，无法比对"
    if before == after:
        return True, "已比对去掉注释后的代码：与原文完全一致（空白与换行不影响语义）"
    return (
        False,
        "已比对去掉注释后的代码：与原文存在差异，模型可能改动了代码本身",
    )


def count_covered_functions(
    commented: str,
    functions: list[dict[str, Any]],
    *,
    language: str,
    filename: str = "<generated>",
) -> tuple[int, list[str]]:
    """统计有多少个函数拿到了注释。

    参数:
        commented: 加了注释的代码。
        functions: **原代码**里解析出的函数清单（含 name）。
        language:  语言标识。
        filename:  文件名（重新解析生成代码时用，只影响报错信息）。

    返回:
        `(有注释的函数数, 没注释的函数名列表)`。

    关键逻辑（这里踩过一个真实的坑，别改回去）:
        不能拿**原代码的行号**去生成后的代码里找函数——加了文件级注释和
        文档字符串之后，所有函数的行号都会整体下移，按老行号去找必然错位，
        结果是"明明每个函数都有注释，却报 0/2 覆盖"。
        所以这里**重新解析生成后的代码**，按函数名把两边对上：
          · Python：用 ast 找函数节点，直接判断它有没有文档字符串
            （比按行扫描文本更准，也不会把"函数体第一行恰好是注释"算成文档字符串）；
          · C/Java：用同一个解析器重新拿函数定义行号，再往上找紧邻的注释。
        解析失败时退回文本判断，宁可少算也不错算。
    """
    lines = commented.splitlines()
    covered = 0
    missing: list[str] = []

    # ---- Python：AST 精确判断文档字符串 ----
    if language == "python":
        documented = _python_documented_names(commented)
        if documented is not None:
            for item in functions:
                name = str(item.get("name") or "")
                if name and name in documented:
                    covered += 1
                else:
                    missing.append(name or "(未知函数)")
            return covered, missing

    # ---- C/Java（以及 Python 解析失败时的兜底）：重新解析生成代码 ----
    try:
        from app.services.repo.parser import parse_source

        parsed = parse_source(commented.encode("utf-8", errors="replace"), filename, language)
        lines_by_name = {
            symbol.name: symbol.start_line
            for symbol in parsed.symbols
            if symbol.kind in ("function", "method")
        }
    except Exception:  # noqa: BLE001 - 解析失败不该让整个复检崩掉
        logger.warning("生成代码重新解析失败，退回按名称扫描: %s", filename, exc_info=True)
        lines_by_name = {}

    for item in functions:
        name = str(item.get("name") or "")
        start = lines_by_name.get(name)
        if start is None:
            # 解析不到就退回"按名字在文本里找函数定义行"
            start = _guess_definition_line(lines, name, language)
        if start is None:
            missing.append(name or "(未知函数)")
            continue
        if _has_leading_comment(lines, start, language):
            covered += 1
        else:
            missing.append(name or "(未知函数)")

    return covered, missing


def _python_documented_names(code: str) -> set[str] | None:
    """返回带文档字符串的函数/方法名集合；解析失败返回 None。

    参数:
        code: Python 源码。

    返回:
        有文档字符串的函数名集合；代码有语法错误时返回 None，
        让调用方退回文本判断（而不是直接报"全都没注释"）。
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None

    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = node.body
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            names.add(node.name)
    return names


def _guess_definition_line(
    lines: list[str], name: str, language: str
) -> int | None:
    """在文本里猜函数定义所在行（只在解析器给不出行号时兜底用）。

    参数:
        lines:    代码的每一行。
        name:     函数名。
        language: 语言标识。

    返回:
        1 基行号；找不到返回 None。
    """
    if not name:
        return None
    for index, row in enumerate(lines, start=1):
        stripped = row.strip()
        if language == "python":
            if stripped.startswith((f"def {name}(", f"def {name} (", f"async def {name}(")):
                return index
        elif re.search(rf"\b{re.escape(name)}\s*\(", stripped) and "(" in stripped:
            return index
    return None


def _has_leading_comment(lines: list[str], start_line: int, language: str) -> bool:
    """判断某个函数（定义在第 start_line 行）是否带注释。

    参数:
        lines:      代码的每一行。
        start_line: 函数定义所在行（1 基）。
        language:   语言标识。

    返回:
        有注释返回 True。

    关键逻辑:
        · Python：定义行之后紧邻的非空行以三引号开头 → 有文档字符串。
          （更精确的判断在 `_python_documented_names` 里用 ast 做，
          这个文本版本只在解析失败时兜底。）
        · C/Java：定义行往上最多 `_COMMENT_LOOKBACK` 行内出现注解结束标记
          或 `//` 行 → 有块注释；一旦遇到真正的代码行就停止向上找，
          避免把上一个函数的注释算到这一个头上。
    """
    index = start_line - 1   # 转成 0 基

    if language == "python":
        # 文档字符串：定义行的下一个非空行
        for offset in range(1, _COMMENT_LOOKBACK + 2):
            position = index + offset
            if position >= len(lines):
                break
            stripped = lines[position].strip()
            if not stripped:
                continue
            return stripped.startswith(('"""', "'''", 'r"""', "r'''"))
        return False

    # C/Java：往上找紧邻的注释行
    for offset in range(1, _COMMENT_LOOKBACK + 1):
        position = index - offset
        if position < 0:
            break
        stripped = lines[position].strip()
        if not stripped:
            continue
        if stripped.startswith("//") or stripped.endswith("*/") or stripped.startswith("/*"):
            return True
        # 遇到第一行真正的代码就停止向上找
        break
    return False


def has_file_comment(code: str, language: str) -> bool:
    """判断代码开头是否有文件级注释。

    参数:
        code:     源码。
        language: 语言标识。

    返回:
        有文件级注释返回 True。

    关键逻辑:
        Python 看模块文档字符串（第一个非空行是否以三引号开头）；
        C/Java 看第一个非空行是否是注释（`/*` 或 `//`）。
        允许前面有若干空行——学生文件开头留空行很常见。
    """
    for row in code.splitlines():
        stripped = row.strip()
        if not stripped:
            continue
        if language == "python":
            return stripped.startswith(('"""', "'''"))
        return stripped.startswith("/*") or stripped.startswith("//")
    return False


def count_inline_comments(
    code: str, language: str, *, filename: str = "<generated>"
) -> int:
    """统计「关键逻辑注释」的行数——三层注释里的第三层。

    参数:
        code:     源码。
        language: 语言标识。
        filename: 文件名（解析 C/Java 函数位置时用）。

    返回:
        关键逻辑注释的行数。

    关键逻辑（这里踩过一个坑，定义不能想当然）:
        最初只统计"行尾注释"（`x = 1  # 说明`），结果实测模型把关键逻辑注释
        写成了**代码上方独立一行**（`# 累加总分` 换行 `total += score`）——
        这同样是在解释具体逻辑，却被统计成 0 行，白白冤枉了模型。
        所以这里的口径是**排除法**：

            关键逻辑注释 = 全部注释行 − 文件级注释行 − 函数级注释行

        其中"全部注释行"同时包含整行注释与行尾注释两种写法；
        "文件级"指文件开头的说明块，"函数级"指 Python 的文档字符串
        或 C/Java 紧邻函数定义的注释块。剩下的自然就是"针对某段逻辑的注释"。
    """
    whole_line = count_lines_by_kind(code, language)[1]
    trailing = _count_trailing_comments(code, language)
    reserved = _reserved_comment_lines(code, language, filename=filename)
    return max(0, whole_line - reserved) + trailing


def _count_trailing_comments(code: str, language: str) -> int:
    """统计行尾注释（代码后面跟注释）的行数。

    参数:
        code:     源码。
        language: 语言标识。

    返回:
        行尾注释行数。

    关键逻辑:
        判据是"去掉注释后仍有代码"——整行注释去掉后就空了，因此不会被重复计数。
    """
    count = 0
    for row in code.splitlines():
        if not row.strip():
            continue
        if language == "python":
            stripped = _strip_python_line_comment(row)
        else:
            stripped = _strip_comments(row, language)
        if stripped.strip() and stripped != row:
            count += 1
    return count


def _reserved_comment_lines(code: str, language: str, *, filename: str) -> int:
    """统计"属于文件级或函数级"的注释行数（这些不该算进关键逻辑注释）。

    参数:
        code:     源码。
        language: 语言标识。
        filename: 文件名。

    返回:
        应被排除的注释行数。

    关键逻辑:
        · Python：用 ast 精确找出模块/类/函数的文档字符串所占行数；
        · C/Java：找出文件开头的注释块，以及紧邻每个函数定义的注释块——
          函数位置由同一个解析器给出，与"注释覆盖率"用的是同一套判据，
          两个指标不会互相打架。
        解析失败时返回 0（宁可把注释多算成"关键逻辑注释"，也不要少算成 0）。
    """
    if language == "python":
        # 只排除"非空"的文档字符串行：注释计数器从不把空行算作注释，
        # 若这里把文档字符串内部的空行也算成"已排除"，两边口径就会差几行，
        # 导致明明有 2 条关键逻辑注释却报 1 条（实测踩过）。
        lines = code.splitlines()
        return sum(
            1
            for line_no in _python_docstring_lines(code)
            if 1 <= line_no <= len(lines) and lines[line_no - 1].strip()
        )

    lines = code.splitlines()
    reserved: set[int] = set()

    # ① 文件开头连续出现的注释块
    for index, row in enumerate(lines):
        stripped = row.strip()
        if not stripped:
            continue
        if stripped.startswith("/*") or stripped.startswith("//") or stripped.startswith("*"):
            reserved.add(index)
            if stripped.endswith("*/"):
                break
            continue
        break  # 遇到第一行代码就停

    # ② 紧邻函数定义的注释块
    try:
        from app.services.repo.parser import parse_source

        parsed = parse_source(code.encode("utf-8", errors="replace"), filename, language)
        starts = [
            symbol.start_line
            for symbol in parsed.symbols
            if symbol.kind in ("function", "method")
        ]
    except Exception:  # noqa: BLE001 - 解析失败就只按文件级注释排除
        starts = []

    for start in starts:
        index = start - 2                     # 定义行上一行的 0 基下标
        for _ in range(_COMMENT_LOOKBACK):
            if index < 0:
                break
            stripped = lines[index].strip()
            if not stripped:
                index -= 1
                continue
            if stripped.startswith("//") or stripped.startswith("*") or stripped.endswith("*/"):
                reserved.add(index)
                index -= 1
                continue
            if stripped.startswith("/*"):
                reserved.add(index)
                break
            break

    return len(reserved)


def _python_docstring_lines(code: str) -> set[int]:
    """找出 Python 里所有文档字符串占用的行号（模块/类/函数）。

    参数:
        code: Python 源码。

    返回:
        行号集合（1 基）；语法错误时返回空集合。
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


def _signature_of(symbol: dict[str, Any]) -> str:
    """从本地解析结果里取函数签名；没有就给出一个可读的兜底描述。

    参数:
        symbol: 本地解析出的符号字典。

    返回:
        形如 `def average(scores)` 的签名字符串。

    关键逻辑:
        签名（尤其是参数名）是模型写"参数说明"的唯一依据。
        本地解析器把签名存在 `signature` 字段里；一旦为空，
        明确写成"（签名未解析出来）"而不是编一个，
        免得模型照着假签名写出张冠李戴的参数说明。
    """
    signature = str(symbol.get("signature") or "").strip()
    return signature or "（签名未解析出来，请以代码为准）"


def _default_suffix(language: str | None) -> str:
    """语言为空时给文件名一个合理的默认后缀。"""
    text = (language or "").strip().lower()
    return {"c": "c", "java": "java", "python": "py"}.get(text, "py")


__all__ = [
    "COMMENT_SYSTEM",
    "PROMPT_VERSION",
    "STYLE_GUIDE",
    "CommentGenerator",
    "CommentGeneratorError",
    "CommentOutcome",
    "CommentVerification",
    "build_messages",
    "build_user_message",
    "compare_code",
    "count_covered_functions",
    "count_inline_comments",
    "has_file_comment",
]
