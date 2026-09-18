"""AI 代码导师 Agent：代码检测、注释生成、自动改错。

与项目里已有的「修复 Agent」的区别：
- 修复 Agent 面向的是**给开源项目提 PR**，输出 unified diff，简洁、工程化；
- 本模块面向的是**学生学编程**，要求学生看得懂，因此：
  · 检测要给出「为什么这是问题 + 怎么改」，而不是只报错；
  · 注释要通俗，讲清"这段在干什么"，而不是复述语法；
  · 改错要逐条列出「原来怎么写 / 改成什么 / 为什么」，而不是只给一份新代码。

三个功能都只依赖 `LLMClient` 协议，因此可以用假客户端离线测试整条链路。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from app.agents.extraction import extract_json_object as _extract_json_object
from app.core.config import Settings
from app.core.llm_client import LLMClient, LLMUsage
from app.core.trace import trace_span

logger = logging.getLogger(__name__)

PROMPT_VERSION = "tutor/v1"

# 各语言的展示名，用于拼接提示词
_LANG_LABEL = {"python": "Python", "c": "C", "java": "Java",
               "javascript": "JavaScript", "go": "Go"}


def _lang_name(language: str) -> str:
    return _LANG_LABEL.get(language, language)


# ===========================================================================
# 一、代码检测
# ===========================================================================
CHECK_SYSTEM = """你是一位耐心细致的编程老师，正在帮学生检查代码。

你的任务：找出代码中的问题，并用学生能听懂的话告诉他怎么改。

检查范围（按重要性排序）：
1. **语法/编译错误**：写错的语法、类型不匹配、缺少分号等，这类问题最优先
2. **逻辑错误**：条件写反、循环边界错、变量用错、返回值不对
3. **运行时风险**：数组越界、空指针、除零、内存泄漏（C）、未关闭资源（Java）
4. **基础规范**：命名、缩进、魔法数字、重复代码

注意事项：
- 学生代码通常很短，不要把"没有写单元测试""没有写文档"这类当成错误
- 不要吹毛求疵地挑格式问题，重点放在真正影响程序正确性的地方
- 每条问题都要给出具体的修改建议，而不是泛泛而谈

只输出一个 JSON 对象，不要任何其他文字、不要用 markdown 代码块包裹。格式：
{
  "score": 85,
  "summary": "一句话总体评价",
  "issues": [
    {
      "line": 12,
      "severity": "error",
      "title": "除数为零会导致程序崩溃",
      "detail": "第 12 行 a / b 中的 b 可能是 0，运行时会抛出异常",
      "suggestion": "先判断 b 是否为 0：if (b == 0) { printf(\\"除数不能为0\\"); return -1; }"
    }
  ],
  "highlights": ["变量命名清晰易读", "缩进规范"]
}

规则：
- score 是 0-100 的整数，90+ 表示基本没问题，60 以下表示有严重错误
- severity 只能是 "error"（会出错）、"warning"（有隐患）、"info"（可以更好）
- 没有问题时代 issues 填空数组 []
- line 是问题所在的行号（从 1 开始）；说不准就填 null
- highlights 最多 3 条，没有就填空数组
"""


# ===========================================================================
# 二、生成注释
# ===========================================================================
COMMENT_SYSTEM = """你是一位编程老师，正在给学生的代码加注释，帮助他理解代码在做什么。

注释要求：
1. **用中文**，语言通俗，像老师讲给学生听
2. 解释「这段代码在干什么、为什么这么写」，而不是复述语法
   - 好："判断分数是否及格，60 分以上输出及格"
   - 差："if 语句，判断 score 是否大于等于 60"
3. 在关键位置加注释：函数开头说明功能与参数、复杂逻辑前说明思路、
   容易看错的变量说明含义
4. **不要给每一行都加注释**，只在真正需要解释的地方加
5. 保留学生原有的代码和注释，只做加法
6. 不要修改代码逻辑，一个字都不要改

只输出一个 JSON 对象，不要任何其他文字、不要用 markdown 代码块包裹。格式：
{
  "commented_code": "加了注释的完整代码，换行用 \\n 表示",
  "summary": "用一句话说明你在哪些地方加了注释"
}
"""


# ===========================================================================
# 三、自动改错
# ===========================================================================
FIX_SYSTEM = """你是一位编程老师，正在帮学生改正代码里的错误。

要求：
1. 先找出代码里的错误，然后给出修正后的**完整代码**
2. 逐条列出你改了什么、为什么这么改，语言要让学生看得懂
3. 如果代码其实没有错误，**不要为了改而改**：
   had_error 填 false，fixed_code 原样返回学生代码，changes 填空数组
4. 只改真正有问题的地方，不要顺手重构、不要改学生的代码风格
5. 保持学生原有的注释

只输出一个 JSON 对象，不要任何其他文字、不要用 markdown 代码块包裹。格式：
{
  "had_error": true,
  "summary": "一句话说明主要问题",
  "fixed_code": "修正后的完整代码，换行用 \\n 表示",
  "changes": [
    {
      "line": 8,
      "original": "return a - b;",
      "fixed": "return a + b;",
      "reason": "函数名叫 add（相加），但这里用了减号，应该改成加号"
    }
  ]
}
"""


@dataclass(slots=True)
class TutorResult:
    """三个功能共用的返回结构。"""

    payload: dict[str, Any] = field(default_factory=dict)
    model: str = ""
    usage: LLMUsage = field(default_factory=LLMUsage)
    duration_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """从模型输出里抠出 JSON 对象（实现已上移到 `app.agents.extraction`）。

    这里保留同名再导出：既有调用方（含测试）一直是从本模块 import 的，
    不动它们的导入路径，避免为了"整理目录"而制造无谓的改动。
    """
    return _extract_json_object(text)


class TutorAgent:
    """AI 代码导师 Agent（检测 / 注释 / 改错）。"""

    # 类名以 Tutor 开头，不会被 pytest 误认成测试类；仍显式声明一次更保险
    __test__ = False

    def __init__(self, settings: Settings, llm: LLMClient) -> None:
        self._settings = settings
        self._llm = llm

    @property
    def llm(self) -> LLMClient:
        return self._llm

    # -- 共用调用 ---------------------------------------------------------
    async def _ask(self, system: str, user: str, *, kind: str) -> TutorResult:
        """发起一次调用并解析 JSON，带 Trace 记录。"""
        started = time.perf_counter()
        with trace_span(
            "agent.tutor",
            kind="agent",
            payload={"kind": kind, "prompt_version": PROMPT_VERSION, "code_chars": len(user)},
            metadata={"model": self._llm.model},
        ) as span:
            from app.core.llm_client import LLMMessage

            response = await self._llm.chat(
                [
                    LLMMessage(role="system", content=system),
                    LLMMessage(role="user", content=user),
                ]
            )
            warnings: list[str] = []
            payload = extract_json_object(response.content)
            if payload is None:
                warnings.append("模型返回的内容不是合法 JSON，已按空结果处理")
                logger.warning("导师 Agent 输出解析失败，前 200 字：%s", response.content[:200])
                payload = {}

            duration = (time.perf_counter() - started) * 1000
            span.set_metadata(parsed=bool(payload), duration_ms=round(duration, 1))
            span.set_output({"keys": sorted(payload.keys())})

            return TutorResult(
                payload=payload,
                model=response.model or self._llm.model,
                usage=response.usage,
                duration_ms=duration,
                warnings=warnings,
            )

    # -- 1. 检测 ----------------------------------------------------------
    async def check(self, code: str, *, filename: str, language: str) -> TutorResult:
        """检查代码质量，返回评分、问题列表与亮点。"""
        user = (
            f"请检查下面这段 {_lang_name(language)} 代码（文件名：{filename}）。\n\n"
            f"```{language}\n{code}\n```"
        )
        result = await self._ask(CHECK_SYSTEM, user, kind="check")

        # 归一化：分数收敛到 0-100，缺失字段补默认值，避免前端拿到脏数据
        payload = result.payload
        try:
            score = float(payload.get("score", 0))
        except (TypeError, ValueError):
            score = 0.0
            result.warnings.append("模型给的评分不是数字，已置为 0")
        payload["score"] = max(0.0, min(100.0, score))

        issues = payload.get("issues")
        payload["issues"] = [item for item in issues if isinstance(item, dict)] if isinstance(
            issues, list
        ) else []
        for issue in payload["issues"]:
            issue.setdefault("severity", "info")
            if issue["severity"] not in ("error", "warning", "info"):
                issue["severity"] = "info"

        highlights = payload.get("highlights")
        payload["highlights"] = (
            [str(item) for item in highlights if str(item).strip()]
            if isinstance(highlights, list)
            else []
        )
        payload.setdefault("summary", "")
        return result

    # -- 2. 生成注释 ------------------------------------------------------
    async def comment(self, code: str, *, filename: str, language: str) -> TutorResult:
        """为代码生成通俗的中文注释。"""
        user = (
            f"请给下面这段 {_lang_name(language)} 代码加上通俗的中文注释"
            f"（文件名：{filename}）。\n\n```{language}\n{code}\n```"
        )
        result = await self._ask(COMMENT_SYSTEM, user, kind="comment")

        commented = result.payload.get("commented_code")
        if not isinstance(commented, str) or not commented.strip():
            # 模型没给出可用代码时，退回原代码并告警，前端据此提示学生重试
            result.warnings.append("模型没有返回带注释的代码，已回退为原代码")
            result.payload["commented_code"] = code
        result.payload.setdefault("summary", "")
        return result

    # -- 3. 自动改错 ------------------------------------------------------
    async def fix(self, code: str, *, filename: str, language: str) -> TutorResult:
        """找出错误并给出修正后的代码与逐条说明。"""
        user = (
            f"请检查并修正下面这段 {_lang_name(language)} 代码"
            f"（文件名：{filename}）。\n\n```{language}\n{code}\n```"
        )
        result = await self._ask(FIX_SYSTEM, user, kind="fix")

        payload = result.payload
        fixed = payload.get("fixed_code")
        if not isinstance(fixed, str) or not fixed.strip():
            result.warnings.append("模型没有返回修正后的代码，已回退为原代码")
            payload["fixed_code"] = code
        payload["had_error"] = bool(payload.get("had_error", False))

        changes = payload.get("changes")
        payload["changes"] = (
            [item for item in changes if isinstance(item, dict)]
            if isinstance(changes, list)
            else []
        )
        payload.setdefault("summary", "")
        return result


__all__ = ["PROMPT_VERSION", "TutorAgent", "TutorResult", "extract_json_object"]
