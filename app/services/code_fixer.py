"""代码改错：AI 分析错误 → 输出修正后的完整代码 → 用通俗语言解释怎么改、以后怎么避免。

===========================================================================
错误分析逻辑（这是本模块的核心，分三步，顺序不能反）
===========================================================================

    一段有错误的代码 + 语言
        │
        ├─ 第 1 步：本地分析（不联网、毫秒级，复用 code_checker 的本地检查）
        │    · 语法错误：Python 用 ast、C/Java 用 tree-sitter，
        │      拿到**精确到行列**的报错，而不是让模型去猜"大概在第几行"
        │    · 静态风险：可变默认参数、裸 except、gets/strcpy、== 比字符串……
        │    · 风格提示：缩进混用、行尾空格、命名规范
        │    这些结论会**一起塞进 Prompt**，见下面"Prompt 设计"第 2 条
        │
        ├─ 第 2 步：AI 分析并给出修正后的完整代码
        │    每条修改必须回答四个问题（对应学生的认知路径）：
        │      ① 原来错在哪里（定位）→ ② 为什么错（理解）
        │      → ③ 怎么改（修正）→ ④ 以后如何避免（迁移）
        │    只给"改前/改后"而不讲"为什么"和"以后怎么避免"，
        │    学生下次还会犯同样的错——这是本模块和普通"自动修复"最大的区别。
        │
        └─ 第 3 步：本地复检（本模块最值得说的一步）
             AI 说改好了，不等于真的改好了。这里把修正后的代码**再跑一遍语法检查**：
               · 改前有语法错误、改后没有了  → verified=True，可以放心用
               · 改后仍有语法错误           → verified=False，明确告诉学生"仍需人工确认"
               · 改前本来就没有语法错误      → verified=True，但必须说明
                 "逻辑正确性本地无法验证，请自己跑一遍"
             没有第 3 步的话，"AI 给的代码"和"能跑的代码"之间的差距就没人把关了。

===========================================================================
AI Prompt 的设计（完整模板见 FIX_SYSTEM，逐条说明见其上方注释）
===========================================================================

一句话：**把模型当作"会讲题的助教"，而不是"代码修复工具"。**

十条约束：角色与受众 / 注入本地结论 / 四问结构 / 必须给完整代码 /
只改该改的 / 没错误就明说 / 禁止编造行号 / 保留原注释 / 严格 JSON /
反面示例（不要"这里有语法错误"这种不说明在哪、为什么的话）。

===========================================================================
与项目里其它"改错"功能的区别
===========================================================================
  · `app/agents/fix_agent.py`：面向**给开源项目提 PR**，输出 unified diff，简洁工程化；
  · `app/agents/tutor_agent.py` 的 `fix()`：前端「自动改错」按钮用的轻量版；
  · 本模块：面向**学生学编程**，多了本地分析前置、本地复检、四问式讲解，
    并把原代码/新代码/说明一起存库供对比复习。
"""

from __future__ import annotations

import difflib
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

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
    language_label,
    locate_syntax_error,
    normalize_language,
)

logger = logging.getLogger(__name__)

# Prompt 版本号：改动提示词时同步 +1，便于把"结果变差"归因到哪一版
PROMPT_VERSION = "code-fix/v1"

# 错误类型：与检测模块保持同一套分类，学生在两个功能里看到的词是一致的
ChangeCategory = Literal["syntax", "logic", "risk", "style"]

# diff 最多保留多少行：几百行的 diff 没人看，还容易把响应撑大
MAX_DIFF_LINES = 200
# 统一 diff 的上下文行数
DIFF_CONTEXT_LINES = 3


# ===========================================================================
# 一、AI Prompt
# ===========================================================================
# 设计思路逐条说明——每条都对应一个真实会翻车的点：
#
# 【1】角色 + 受众放最前面
#     "有耐心的助教 + 读者是初学者"，模型的语言会立刻从"代码评审"切到"讲题"。
#
# 【2】把本地分析结论注入 Prompt
#     语法错误的**行列号**由 ast/tree-sitter 给出，比模型目测准得多；
#     同时明确要求"不要重复描述这些已知信息，把它们当事实"，
#     让模型把篇幅花在"为什么错"和"怎么避免"上——这正是学生最需要的部分。
#
# 【3】强制四问结构（what / why / how / avoid）
#     这是本模块的灵魂。只让模型"给出修正后的代码"，它会只给代码；
#     只让它"说明改了什么"，它会写"修正了循环边界"这种空话。
#     把四个问题拆成四个必填字段后，输出被迫具体：
#       ① what  原来错在哪里  —— 定位到具体这一行、这个写法
#       ② why   为什么错      —— 讲清楚后果（越界、死循环、结果不对）
#       ③ how   怎么改        —— 给出可直接替换的代码
#       ④ avoid 以后如何避免   —— 把这次错误抽象成一条可复用的经验
#     第 ④ 条是"改错"和"学会"的分界线，绝大多数自动修复工具都不给。
#
# 【4】必须给完整代码（而不是片段）
#     学生要能直接复制去运行；只给片段的话，他自己拼接时很容易接错。
#
# 【5】只改真正有问题的地方，不许顺手重构
#     这条是实践得出的：不加约束，模型会把变量名、代码风格一起改掉，
#     学生拿到一份"认不出是自己写的"代码，反而更困惑。
#
# 【6】没发现错误就明说，不要为了改而改
#     否则模型会硬凑出几条"改进建议"，让学生以为自己的代码到处是错。
#
# 【7】禁止编造
#     行号必须真实存在；`original` 必须是代码里**原文摘抄**；
#     不要提代码里没有的函数名——这三条覆盖了模型最常编造的三类内容。
#
# 【8】保留学生原有的注释
#     注释是学生自己的思考痕迹，被删掉会让人很恼火，也不利于复习。
#
# 【9】严格 JSON，不要 markdown 围栏
#     真不听话时由 `extract_json_object` 三级降级兜底解析。
#
# 【10】给反面示例
#     "这里有语法错误" / "建议优化一下" 这类话必须禁止，要求每条都能指出
#     具体写法与具体后果。
FIX_SYSTEM = """你是一位有耐心的编程入门课助教，正在帮大一新生改正代码里的错误。

## 你的读者
一名刚开始学编程的学生。他看得懂基本语法，但常常"知道错了却不知道为什么错"。
请用**通俗的中文**讲，不要堆术语；必须用术语时，用一句生活化的比喻解释它。

## 已知的本地分析结果
系统已经用 AST / tree-sitter 和静态规则分析过这份代码，结论如下（当作事实，不用重复描述）：
<<LOCAL_ANALYSIS>>

## 任务
1. 找出代码里**真正**的错误：语法错误、逻辑错误、以及初学者常踩的坑
2. 给出**修正后的完整代码**（学生要能直接复制运行，不要只给片段）
3. 对每一处修改，用四个字段讲清楚：
   - `what`：原来错在哪里（指出具体写法，例如"循环条件写成了 i <= n"）
   - `why`：为什么这是错的（会造成什么后果，例如"会读到数组外面"）
   - `how`：怎么改（给出改后的代码写法）
   - `avoid`：以后如何避免（提炼成一条可复用的经验，例如"写完循环先数一遍
     最后一个合法下标是多少"）

## 硬性要求
- **只改真正有问题的地方**：不要顺手重构、不要改变量名、不要调整代码风格
- **保留学生原有的注释**，不要删掉他的思考痕迹
- 如果代码其实没有错误，不要为了改而改：
  `had_error` 填 false，`fixed_code` 原样返回学生代码，`changes` 填空数组
- 不要写"这里有语法错误""建议优化"这类不指出具体位置和原因的话
- `original` 必须是代码里**原文摘抄**（一个字都不要改）
- `line` 只能是代码里真实存在的行号（从 1 开始）；拿不准就填 null，**不要猜**
- 不要提代码里根本不存在的函数名或变量名

## 输出格式
只输出一个 JSON 对象，不要解释文字，不要用 markdown 代码块包裹：
{
  "had_error": true,
  "summary": "一句话说明这份代码的主要问题",
  "fixed_code": "修正后的完整代码，换行用 \\n 表示",
  "changes": [
    {
      "line": 12,
      "category": "logic",
      "what": "循环条件写成了 i <= n",
      "why": "数组下标最大只到 n-1，i == n 时会读到数组外面，结果不可预测甚至崩溃",
      "how": "改成 for (int i = 0; i < n; i++)",
      "avoid": "写完循环先数一遍：最后一个合法下标是多少？循环变量会不会取到它？",
      "original": "for (int i = 0; i <= n; i++) {",
      "fixed": "for (int i = 0; i < n; i++) {"
    }
  ]
}

`category` 只能取这四个值之一：
- "syntax"：语法/编译错误
- "logic"：逻辑错误（能跑，但结果不对）
- "risk"：潜在风险（现在能跑，某些输入下会出问题）
- "style"：风格问题（不影响运行）
"""

# 提示词里的占位符。
# **不要用 str.format()**：上面的 JSON 示例含 `{` `}`，会被当成格式化字段并抛
# `KeyError`（在检测模块里实测踩过这个坑）。用不会与 JSON 冲突的记号 + replace。
LOCAL_ANALYSIS_PLACEHOLDER = "<<LOCAL_ANALYSIS>>"

# 允许的错误类型；模型给了别的值就收敛成 logic（绝大多数真问题是逻辑错）
_ALLOWED_CATEGORIES = ("syntax", "logic", "risk", "style")


# ===========================================================================
# 二、数据结构
# ===========================================================================
@dataclass(slots=True)
class CodeChange:
    """一处修改的说明（四问结构）。"""

    what: str = ""      # 原来错在哪里
    why: str = ""       # 为什么错
    how: str = ""       # 怎么改
    avoid: str = ""     # 以后如何避免
    line: int | None = None
    category: ChangeCategory = "logic"
    original: str = ""  # 改前的代码（原文摘抄）
    fixed: str = ""     # 改后的代码

    def to_dict(self) -> dict[str, Any]:
        return {
            "line": self.line,
            "category": self.category,
            "what": self.what,
            "why": self.why,
            "how": self.how,
            "avoid": self.avoid,
            "original": self.original,
            "fixed": self.fixed,
        }


@dataclass(slots=True)
class DiffStats:
    """两份代码的差异统计，供前端显示"改了多少"。"""

    added: int = 0
    removed: int = 0
    changed: int = 0
    unchanged: int = 0

    @property
    def total_changed(self) -> int:
        return self.added + self.removed + self.changed

    def to_dict(self) -> dict[str, int]:
        return {
            "added": self.added,
            "removed": self.removed,
            "changed": self.changed,
            "unchanged": self.unchanged,
            "total_changed": self.total_changed,
        }


@dataclass(slots=True)
class FixVerification:
    """第 3 步本地复检的结论。"""

    syntax_before: SyntaxErrorInfo | None = None
    syntax_after: SyntaxErrorInfo | None = None
    verified: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "verified": self.verified,
            "note": self.note,
            "syntax_before": _syntax_to_dict(self.syntax_before),
            "syntax_after": _syntax_to_dict(self.syntax_after),
        }


def _syntax_to_dict(info: SyntaxErrorInfo | None) -> dict[str, Any] | None:
    """把语法错误信息转成可序列化的字典（None 原样返回）。"""
    if info is None:
        return None
    return {
        "line": info.line,
        "column": info.column,
        "message": info.message,
        "tool": info.tool,
        "raw": info.raw,
    }


@dataclass(slots=True)
class FixOutcome:
    """一次改错的完整结果。"""

    filename: str
    language: str
    original_code: str
    fixed_code: str = ""
    changes: list[CodeChange] = field(default_factory=list)
    had_error: bool = False
    summary: str = ""
    diff: str = ""
    diff_stats: DiffStats = field(default_factory=DiffStats)
    verification: FixVerification = field(default_factory=FixVerification)
    local: LocalCheckResult | None = None
    ai_available: bool = False
    model: str = ""
    usage: LLMUsage = field(default_factory=LLMUsage)
    warnings: list[str] = field(default_factory=list)
    note: str = ""
    duration_ms: float = 0.0
    ai_duration_ms: float | None = None
    fixed_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def change_count(self) -> int:
        """修改处数。"""
        return len(self.changes)

    @property
    def categories(self) -> dict[str, int]:
        """各类错误的条数，例如 {"logic": 2, "style": 1}。"""
        counts: dict[str, int] = {}
        for item in self.changes:
            counts[item.category] = counts.get(item.category, 0) + 1
        return counts


class CodeFixerError(RuntimeError):
    """改错过程中的可预期错误（调用方转 4xx）。"""


# ===========================================================================
# 三、CodeFixer
# ===========================================================================
class CodeFixer:
    """代码改错：本地分析 → AI 修正 → 本地复检。"""

    # 类名以 Code 开头，不会被 pytest 当成测试类；仍显式声明一次更保险
    __test__ = False

    def __init__(
        self, settings: Settings, llm: LLMClient, checker: CodeChecker | None = None
    ) -> None:
        self._settings = settings
        self._llm = llm
        # 直接复用检测模块的本地检查能力：语法定位、风格规则、风险模式都在那里，
        # 两个功能对"什么算问题"的判断必须一致，否则学生会困惑"为什么这里说对那里说错"。
        self._checker = checker or CodeChecker(settings, llm)

    @property
    def llm(self) -> LLMClient:
        return self._llm

    # ------------------------------------------------------------------
    # 第 1 步：本地分析
    # ------------------------------------------------------------------
    def analyze_locally(self, code: str, *, language: str, filename: str) -> LocalCheckResult:
        """本地分析：语法错误位置 + 风格/风险清单（不联网）。

        参数:
            code:     待分析的源码。
            language: 标准语言标识（已归一）。
            filename: 文件名，用于报错信息。

        返回:
            `LocalCheckResult`（与检测模块同一结构）。

        关键逻辑:
            直接委托给 `CodeChecker.local_check`，不另写一套规则。
            理由有两条：
              1. 两个功能对"什么算问题"必须是同一标准，否则学生会在
                 「检测」说没问题、「改错」说有问题之间反复横跳；
              2. 语法定位这块已经在检测模块被大量用例覆盖，重写等于重复踩坑。
        """
        return self._checker.local_check(code, language=language, filename=filename)

    # ------------------------------------------------------------------
    # 第 3 步：本地复检
    # ------------------------------------------------------------------
    def verify_fix(
        self, fixed_code: str, *, language: str, filename: str, before: SyntaxErrorInfo | None
    ) -> FixVerification:
        """复检 AI 给出的修正代码：语法是否真的通过了。

        参数:
            fixed_code: AI 返回的修正后代码。
            language:   语言标识。
            filename:   文件名。
            before:     修改前的语法错误（本地分析阶段得到的）。

        返回:
            `FixVerification`，含复检结论与一句人话说明。

        关键逻辑（`verified` 的含义必须说清楚，否则会误导学生）:
            本函数**只能验证语法**，不能验证逻辑。所以：
              · 改前有语法错误、改后没有了 → verified=True
                "修正后的代码已通过语法检查"
              · 改后有语法错误 → verified=False
                "修正后的代码仍然有语法错误，请把报错发给我再看看"
              · 改前本来就没有语法错误（典型是逻辑错误）→ verified=True
                但必须补一句"逻辑正确性本地无法验证，请自己跑一组测试"
            最后这条最容易被忽略也最容易误导：一个"verified=True"如果被理解成
            "逻辑也对了"，学生就会直接交作业。所以 note 里必须写明验证的边界。
        """
        if not fixed_code.strip():
            return FixVerification(
                syntax_before=before,
                syntax_after=None,
                verified=False,
                note="模型没有返回可用的代码，无法复检。",
            )

        after = locate_syntax_error(fixed_code, language=language, filename=filename)

        if after is not None:
            where = f"第 {after.line} 行" if after.line else "未知位置"
            return FixVerification(
                syntax_before=before,
                syntax_after=after,
                verified=False,
                note=(
                    f"修正后的代码**仍然有语法错误**（{where}：{after.message}）。"
                    "请把这份代码和报错信息再发我一次，或自己对照报错行检查。"
                ),
            )

        if before is not None:
            return FixVerification(
                syntax_before=before,
                syntax_after=None,
                verified=True,
                note=(
                    f"修改前第 {before.line} 行的语法错误已经消除，"
                    "修正后的代码通过了语法检查，可以直接复制去运行。"
                ),
            )

        return FixVerification(
            syntax_before=None,
            syntax_after=None,
            verified=True,
            note=(
                "修改前后都能通过语法检查（这类问题通常是逻辑错误），"
                "**逻辑是否正确本地无法自动验证**：请自己造几组输入跑一遍，"
                "尤其试试空数据、只有一条数据、以及最大值/最小值。"
            ),
        )

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    async def fix(
        self,
        code: str,
        *,
        language: str,
        filename: str,
        session_id: str | None = None,
        model_id: str | None = None,
        api_key: str | None = None,
    ) -> FixOutcome:
        """完整改错流程：本地分析 → AI 修正 → 本地复检。

        参数:
            code:     有错误的源码。
            language: 标准语言标识。
            filename: 文件名。
            session_id: 网页带来的会话号（`X-Session-Id`）：给了就用
                **用户自己存的 Key** 调模型，而不是 `.env` 里那把。
            model_id: 网页上选的模型 ID。
            api_key: 本次请求直接带来的 Key（优先级最高）。

        返回:
            `FixOutcome`。AI 不可用时**不抛异常**，而是返回
            `ai_available=False` + 本地分析结论 + 一句明确提示
            （含"请先在网页上输入 API Key"）。

        异常:
            CodeFixerError: 代码为空或过长。

        关键逻辑:
            与检测模块同样的取舍：AI 挂了不能把已经算出来的本地分析一起丢掉。
            没配模型时，学生至少能看到"第几行有语法错误、错在哪"。
        """
        if not code.strip():
            raise CodeFixerError("代码内容为空，没有可修正的内容")
        if len(code) > MAX_CODE_CHARS:
            raise CodeFixerError(
                f"代码过长（{len(code)} 字符），上限为 {MAX_CODE_CHARS} 字符"
            )

        started = time.perf_counter()
        # ---- 第 1 步：本地分析 ----
        local = self.analyze_locally(code, language=language, filename=filename)

        outcome = FixOutcome(
            filename=filename,
            language=language,
            original_code=code,
            local=local,
            # 默认先把原代码原样放上：万一 AI 完全失败，返回的也是"能跑的原代码"，
            # 而不是空字符串——空字符串会让前端显示成一片空白，像是把学生的代码弄丢了
            fixed_code=code,
        )

        # ---- 第 2、3 步：先确定"这次用哪个模型 / 哪把 Key"，再调 AI 并复检 ----
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
        outcome: FixOutcome,
        llm: LLMClient,
        started: float,
    ) -> None:
        """跑 AI 修正 + 本地复检，把结果写进 `outcome`（原地修改）。

        抽成单独方法是为了让"选客户端"留在 `async with` 里：用户 Key 用完即关。
        """
        # ---- 没有可用 Key（页面没填、后端也没配）：给出那句明确提示，跳过 AI ----
        if not llm.configured:
            outcome.note = MISSING_API_KEY_NOTE.format(extra="改错")
            outcome.warnings.append(f"{MISSING_API_KEY_HINT}，已跳过 AI 改错")
            outcome.verification = self.verify_fix(
                code, language=language, filename=filename, before=local.syntax_error
            )
            outcome.diff_stats = diff_stats(code, code)
            outcome.duration_ms = (time.perf_counter() - started) * 1000
            return

        outcome.ai_available = True
        try:
            payload, usage, model, latency = await self._ask_ai(
                code, language=language, filename=filename, local=local, llm=llm
            )
        except Exception as exc:  # noqa: BLE001 - AI 失败也要给出本地结论
            logger.warning("AI 改错失败，仅返回本地分析: %s", exc, exc_info=True)
            outcome.ai_available = False
            outcome.note = f"AI 改错调用失败（{exc}），以下仅为本地分析结果。"
            outcome.warnings.append(f"AI 调用失败：{exc}")
            outcome.verification = self.verify_fix(
                code, language=language, filename=filename, before=local.syntax_error
            )
            outcome.diff_stats = diff_stats(code, code)
            outcome.duration_ms = (time.perf_counter() - started) * 1000
            return

        outcome.usage = usage
        outcome.model = model
        outcome.ai_duration_ms = latency

        # ---- 第 3 步：合并 AI 结论 + 本地复检 ----
        self._merge_ai(outcome, payload)
        outcome.verification = self.verify_fix(
            outcome.fixed_code,
            language=language,
            filename=filename,
            before=local.syntax_error,
        )
        outcome.diff = build_unified_diff(code, outcome.fixed_code, filename=filename)
        outcome.diff_stats = diff_stats(code, outcome.fixed_code)
        self._cross_check(outcome)

        outcome.duration_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "代码改错完成: 文件=%s 语言=%s 有错=%s 修改=%s 处 复检=%s 耗时=%.0fms",
            filename, language, outcome.had_error, outcome.change_count,
            outcome.verification.verified, outcome.duration_ms,
        )

    async def fix_text(
        self,
        code: str,
        *,
        language: str | None,
        filename: str | None,
        session_id: str | None = None,
        model_id: str | None = None,
        api_key: str | None = None,
    ) -> FixOutcome:
        """面向接口的入口：先归一语言与文件名，再走完整流程。

        参数:
            code:     有错误的源码。
            language: 用户传入的语言（可为空，会按文件名推断）。
            filename: 文件名（可为空，会给个默认名）。
            session_id / model_id / api_key: 见 `fix()`（网页填的 Key 从这些参数进来）。

        异常:
            UnsupportedLanguageError: 语言不支持（调用方转 400）。
        """
        resolved_filename = (filename or "").strip() or f"snippet.{_default_suffix(language)}"
        resolved_language = normalize_language(language, resolved_filename)
        return await self.fix(
            code,
            language=resolved_language,
            filename=resolved_filename,
            session_id=session_id,
            model_id=model_id,
            api_key=api_key,
        )

    # ------------------------------------------------------------------
    # AI 调用
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
        """调用模型做错误分析与修正，返回 (解析后的 JSON, 用量, 模型名, 耗时)。

        参数:
            llm: 本次请求要用的客户端（由 `fix()` 按用户选的模型/Key 传进来）；
                 不传则用构造时注入的那个（老行为）。
        """
        client = llm or self._llm
        messages = build_messages(
            code, language=language, filename=filename, local_analysis=local.findings_text()
        )

        # trace_span 是同步上下文管理器（见 app/core/trace.py），所以用 with；
        # 内部的 await 依然跑在事件循环里。
        with trace_span(
            "service.code_fixer",
            kind="agent",
            payload={
                "language": language,
                "filename": filename,
                "code_chars": len(code),
                "syntax_ok_before": local.syntax_ok,
                "prompt_version": PROMPT_VERSION,
            },
            metadata={"model": client.model},
        ) as span:
            response = await client.chat(messages)
            payload = extract_json_object(response.content)
            if payload is None:
                span.set_metadata(parsed=False)
                raise CodeFixerError("模型返回的内容不是合法 JSON")
            span.set_metadata(parsed=True, model=response.model)
            span.set_output({"keys": sorted(payload.keys())})
            return payload, response.usage, response.model or client.model, response.latency_ms

    # ------------------------------------------------------------------
    # 字段清洗
    # ------------------------------------------------------------------
    def _merge_ai(self, outcome: FixOutcome, payload: dict[str, Any]) -> None:
        """把模型的 JSON 结果并进报告，并做一轮字段清洗。

        参数:
            outcome: 已有本地分析的报告对象（原地修改）。
            payload: 模型返回的 JSON。

        关键逻辑（模型输出永远不能直接信）:
            1. `fixed_code` 必须是非空字符串；缺失或为空时**保留原代码**并告警，
               绝不能把空字符串当成"修好了"返回给学生；
            2. 行号必须是代码范围内真实存在的整数，否则置 None（模型爱编行号）；
            3. `category` 收敛到四个合法值；
            4. `original` / `fixed` 缺失时按空串处理，前端会隐藏这一块而不是显示 null；
            5. 每条修改的四个字段（what/why/how/avoid）都做去空白处理，
               同时统计"四个字段都齐全"的比例 —— 缺 why/avoid 的条目
               对学生的价值大打折扣，这属于需要被告知的降级。
        """
        had_error = bool(payload.get("had_error", False))
        fixed_code = payload.get("fixed_code")

        if not isinstance(fixed_code, str) or not fixed_code.strip():
            if had_error:
                outcome.warnings.append("模型声称有错误，但没有返回修正后的代码，已保留原代码")
            fixed_code = outcome.original_code
        outcome.had_error = had_error
        outcome.fixed_code = fixed_code

        raw_changes = payload.get("changes")
        changes: list[CodeChange] = []
        if isinstance(raw_changes, list):
            for item in raw_changes:
                if not isinstance(item, dict):
                    continue
                what = str(item.get("what") or item.get("title") or "").strip()
                if not what:
                    continue          # 没说什么错在哪的条目没有价值，直接丢
                changes.append(
                    CodeChange(
                        what=what,
                        why=str(item.get("why") or item.get("reason") or "").strip(),
                        how=str(item.get("how") or item.get("suggestion") or "").strip(),
                        avoid=str(item.get("avoid") or item.get("prevent") or "").strip(),
                        line=_safe_line(item.get("line"), len(outcome.original_code.splitlines())),
                        category=_safe_category(item.get("category")),
                        original=str(item.get("original") or "").strip(),
                        fixed=str(item.get("fixed") or "").strip(),
                    )
                )
        outcome.changes = changes
        outcome.summary = str(payload.get("summary") or "").strip()

        incomplete = [
            item for item in changes
            if not (item.why and item.how and item.avoid)
        ]
        if incomplete:
            outcome.warnings.append(
                f"有 {len(incomplete)} 条修改说明不完整（缺少'为什么错'或'以后如何避免'）"
            )

    @staticmethod
    def _cross_check(outcome: FixOutcome) -> None:
        """交叉检查本地结论与 AI 结论是否矛盾，矛盾时如实告知。

        参数:
            outcome: 已合并 AI 结论的报告对象（原地修改）。

        关键逻辑:
            两种必须报出来的矛盾（都会直接影响学生的判断）:
              1. **本地查到语法错误，模型却说没问题**：语法是确定性的事实，
                 模型说"没错"就是它漏了。此时必须提醒，否则学生把代码原样交上去。
              2. **模型说有错误，但给出的代码与原文一字不差**：等于什么也没改，
                 学生看了半天以为改好了。这类"为了改而改"的输出必须点破。
        """
        syntax_before = outcome.local.syntax_error if outcome.local else None

        if syntax_before is not None and not outcome.had_error:
            where = f"第 {syntax_before.line} 行" if syntax_before.line else "未知位置"
            outcome.warnings.append(
                f"本地语法检查发现 {where} 有语法错误，但模型认为无需修改，请人工确认"
            )
            outcome.note = outcome.note or "模型与本地检查结论不一致，请以报错信息为准。"

        if outcome.had_error and outcome.fixed_code == outcome.original_code and outcome.changes:
            outcome.warnings.append(
                "模型给出的代码与原代码完全相同，修改说明可能没有对应的实际改动"
            )
            outcome.note = (
                outcome.note or "模型没有真正改动代码，建议重新提交一次或换个说法描述问题。"
            )

        if outcome.had_error and not outcome.changes:
            outcome.warnings.append("模型认为有错误，但没有给出任何修改说明")


# ---------------------------------------------------------------------------
# Prompt 构造
# ---------------------------------------------------------------------------
def build_user_message(
    code: str, *, language: str, filename: str, local_analysis: str
) -> str:
    """构造「用户消息」：这次要改的代码 + 再提醒一次本地已知信息。

    参数:
        code:           学生代码原文（原样放进围栏，不做任何改写）。
        language:       语言标识，用于选择代码围栏。
        filename:       文件名，给模型一点上下文。
        local_analysis: 本地分析结论的文本摘要。

    返回:
        拼接好的用户消息。

    关键逻辑:
        代码用 ``` 围栏包起来并写明语言，能减少"把 C 当 Python 看"这类错误；
        结尾再重申一次"本地已查过什么"，是为了让模型别在语法位置上浪费篇幅。
    """
    return (
        f"请帮我改正下面这段 {language_label(language)} 代码里的错误"
        f"（文件名：{filename}）。\n\n"
        f"```{language}\n{code}\n```\n\n"
        f"再提醒一次本地分析已经查过的内容，请勿重复描述：\n{local_analysis}"
    )


def build_messages(
    code: str, *, language: str, filename: str, local_analysis: str
) -> list[LLMMessage]:
    """组装完整消息列表（系统 + 用户）。

    单独抽成函数：测试里可以直接断言"提示词里确实包含了本地结论"
    "确实要求了四问结构"，而不必真的去调模型。
    """
    return [
        LLMMessage(
            role="system",
            content=FIX_SYSTEM.replace(LOCAL_ANALYSIS_PLACEHOLDER, local_analysis),
        ),
        LLMMessage(
            role="user",
            content=build_user_message(
                code, language=language, filename=filename, local_analysis=local_analysis
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------
def build_unified_diff(original: str, fixed: str, *, filename: str) -> str:
    """生成统一格式（unified diff）的差异文本，供学生逐行对比。

    参数:
        original: 修改前的代码。
        fixed:    修改后的代码。
        filename: 文件名，会出现在 diff 头部的 ---/+++ 行。

    返回:
        unified diff 文本；两份完全相同时返回空字符串。

    关键逻辑:
        · `lineterm=""` 配合 splitlines：避免 difflib 在每行末尾再补一个换行，
          否则前端显示时会出现"每隔一行一个空行"的错位；
        · 行数超过 MAX_DIFF_LINES 时截断并追加提示 —— 几百行的 diff 没人会看完，
          截断比让前端卡住更实用。
    """
    if original == fixed:
        return ""

    diff_lines = list(
        difflib.unified_diff(
            original.splitlines(),
            fixed.splitlines(),
            fromfile=f"a/{filename}（修改前）",
            tofile=f"b/{filename}（修改后）",
            n=DIFF_CONTEXT_LINES,
            lineterm="",
        )
    )
    if len(diff_lines) > MAX_DIFF_LINES:
        diff_lines = diff_lines[:MAX_DIFF_LINES]
        diff_lines.append(f"...（差异过长，仅显示前 {MAX_DIFF_LINES} 行）")
    return "\n".join(diff_lines)


def diff_stats(original: str, fixed: str) -> DiffStats:
    """统计两段代码的增/删/改行数。

    参数:
        original: 修改前的代码。
        fixed:    修改后的代码。

    返回:
        `DiffStats`。

    关键逻辑:
        用 `SequenceMatcher.get_opcodes()`：它把差异切成 equal/replace/delete/insert
        四类操作块。"改"（replace）按两边的行数取较大值计入 changed ——
        这样"1 行改成 3 行"显示为改了 3 行，比显示 1 行更符合直觉。
    """
    stats = DiffStats()
    matcher = difflib.SequenceMatcher(
        None, original.splitlines(), fixed.splitlines(), autojunk=False
    )
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            stats.unchanged += i2 - i1
        elif tag == "delete":
            stats.removed += i2 - i1
        elif tag == "insert":
            stats.added += j2 - j1
        else:  # replace
            stats.changed += max(i2 - i1, j2 - j1)
    return stats


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _safe_line(value: Any, total_lines: int) -> int | None:
    """校验行号：必须是落在代码范围内的整数，否则返回 None。

    模型编行号是高频问题，而这个数字会直接显示给学生并指错地方，
    所以宁可置空也不能带病展示。
    """
    if isinstance(value, bool):  # bool 是 int 的子类，单独挡掉
        return None
    if isinstance(value, int) and 1 <= value <= max(total_lines, 1):
        return value
    return None


def _safe_category(value: Any) -> ChangeCategory:
    """把模型给的类型收敛到四个合法值。"""
    text = str(value or "").strip().lower()
    if text in _ALLOWED_CATEGORIES:
        return text  # type: ignore[return-value]
    # 模型偶尔会写中文（"逻辑错误"）或别的词，统一按最常见的逻辑错误处理
    if "语法" in text or "syntax" in text:
        return "syntax"
    if "风格" in text or "style" in text or "规范" in text:
        return "style"
    if "风险" in text or "risk" in text:
        return "risk"
    return "logic"


def _default_suffix(language: str | None) -> str:
    """语言为空时给文件名一个合理的默认后缀。"""
    text = (language or "").strip().lower()
    return {"c": "c", "java": "java", "python": "py"}.get(text, "py")


__all__ = [
    "FIX_SYSTEM",
    "LOCAL_ANALYSIS_PLACEHOLDER",
    "MAX_DIFF_LINES",
    "PROMPT_VERSION",
    "CodeChange",
    "CodeFixer",
    "CodeFixerError",
    "DiffStats",
    "FixOutcome",
    "FixVerification",
    "build_messages",
    "build_unified_diff",
    "build_user_message",
    "diff_stats",
]
