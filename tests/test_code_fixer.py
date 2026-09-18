"""代码改错测试：本地分析、AI 修正合并、本地复检、diff、SQLite 历史、API 契约。

大模型一律用可注入的假客户端替代，因此整条链路离线可测；
本地分析（ast / tree-sitter）与复检是真跑的——它们才是"改对了没有"的判据。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_code_fixer
from app.core.config import get_settings
from app.core.database import dispose_engine, get_session, init_db
from app.core.llm_client import LLMMessage, LLMResponse, LLMUsage
from app.main import create_app
from app.services.code_fix_service import CodeFixService
from app.services.code_fixer import (
    FIX_SYSTEM,
    MAX_DIFF_LINES,
    PROMPT_VERSION,
    CodeFixer,
    build_messages,
    build_unified_diff,
    diff_stats,
)

# ---------------------------------------------------------------------------
# 样例代码
# ---------------------------------------------------------------------------
BROKEN_PY = '''def average(scores):
    total = 0
    for i in range(len(scores) + 1):
        total += scores[i]
    return total / len(scores)
'''

SHELL_PY = '''def add(a, b)
    return a + b
'''

FIXED_SHELL_PY = '''def add(a, b):
    return a + b
'''

RISKY_C = '''#include <stdio.h>

int main(void) {
    char buf[8];
    gets(buf);
    printf("%s\\n", buf);
    return 0;
}
'''

STILL_BROKEN_C = '''#include <stdio.h>

int main(void) {
    char buf[8]
    return 0;
}
'''

GOOD_PY = '''def add(a, b):
    """求和。"""
    return a + b
'''

# AI 的模拟回复：覆盖"有错并修好"的完整四问结构
FIX_REPLY = json.dumps(
    {
        "had_error": True,
        "summary": "循环边界写错导致越界，另外没有处理空列表。",
        "fixed_code": (
            "def average(scores):\n"
            "    if not scores:\n"
            "        return 0.0\n"
            "    total = 0\n"
            "    for score in scores:\n"
            "        total += score\n"
            "    return total / len(scores)\n"
        ),
        "changes": [
            {
                "line": 3,
                "category": "logic",
                "what": "循环条件写成了 range(len(scores) + 1)，多跑一轮",
                "why": "scores 的下标最大只到 len(scores)-1，多出来的那一轮会读到列表外面，程序直接报 IndexError",
                "how": "改成 for score in scores:，直接遍历元素，既不会越界也更简洁",
                "avoid": "写完循环先数一遍：最后一个合法下标是多少？循环变量会不会取到它？",
                "original": "    for i in range(len(scores) + 1):",
                "fixed": "    for score in scores:",
            },
            {
                "line": 5,
                "category": "risk",
                "what": "没有判断 scores 是否为空",
                "why": "空列表时 len(scores) 是 0，除法会抛 ZeroDivisionError",
                "how": "在函数开头加 if not scores: return 0.0",
                "avoid": "凡是做除法，先问一句「分母会不会是 0」",
                "original": "    return total / len(scores)",
                "fixed": "    if not scores:\n        return 0.0",
            },
        ],
    },
    ensure_ascii=False,
)

NO_ERROR_REPLY = json.dumps(
    {
        "had_error": False,
        "summary": "这段代码没有发现错误。",
        "fixed_code": GOOD_PY,
        "changes": [],
    },
    ensure_ascii=False,
)

# 模型说改好了，但给出的代码仍然有语法错误（复检必须抓出来）
BAD_FIX_REPLY = json.dumps(
    {
        "had_error": True,
        "summary": "已修正。",
        "fixed_code": STILL_BROKEN_C,
        "changes": [
            {
                "line": 4,
                "category": "syntax",
                "what": "缺少分号",
                "why": "C 语句要以分号结尾",
                "how": "在行尾加上分号",
                "avoid": "写完一行就检查结尾符号",
                "original": "    char buf[8]",
                "fixed": "    char buf[8];",
            }
        ],
    },
    ensure_ascii=False,
)


class FixFakeLLM:
    """可编排的假 LLM：记录调用、按固定内容回复。"""

    def __init__(self, reply: str = FIX_REPLY, *, fail_with: Exception | None = None) -> None:
        self.reply = reply
        self.fail_with = fail_with
        self.calls: list[list[LLMMessage]] = []

    @property
    def model(self) -> str:
        return "fake-fixer"

    @property
    def configured(self) -> bool:
        return True

    @property
    def system_prompt(self) -> str:
        return self.calls[-1][0].content if self.calls else ""

    @property
    def user_prompt(self) -> str:
        return self.calls[-1][1].content if self.calls else ""

    async def chat(self, messages: list[LLMMessage], **_: object) -> LLMResponse:
        self.calls.append(list(messages))
        if self.fail_with is not None:
            raise self.fail_with
        return LLMResponse(
            content=self.reply,
            model="fake-fixer",
            finish_reason="stop",
            usage=LLMUsage(prompt_tokens=900, completion_tokens=400, total_tokens=1300),
            latency_ms=15.0,
        )

    async def aclose(self) -> None:
        return None


class OfflineLLM(FixFakeLLM):
    """自称"未配置"的假客户端，用于验证无 AI 时的降级路径。"""

    @property
    def configured(self) -> bool:
        return False


def _fixer(llm: FixFakeLLM) -> CodeFixer:
    return CodeFixer(get_settings(), llm)


def _run_async(coro):
    """跑一个协程（用于在同步测试里拿结果）。"""
    import asyncio

    return asyncio.run(coro)


def _run_db(scenario):
    """一个事件循环里跑完建表 → 场景 → 释放连接池。"""
    import asyncio

    async def main():
        await init_db()
        try:
            async with get_session() as session:
                return await scenario(session)
        finally:
            await dispose_engine()

    return asyncio.run(main())


# ===========================================================================
# 1. Prompt 构造（需求里点名要讲清的部分）
# ===========================================================================
def test_prompt_injects_local_analysis() -> None:
    """本地分析结论必须进提示词：这是"不让模型去猜语法错在哪"的前提。"""
    analysis = "- 语法：**不通过**，第 1 行有语法错误（ast 报告：expected ':'）"
    messages = build_messages(
        SHELL_PY, language="python", filename="a.py", local_analysis=analysis
    )
    assert "expected ':'" in messages[0].content
    assert "第 1 行" in messages[0].content


def test_prompt_asks_four_questions() -> None:
    """四问结构必须写进提示词，否则模型只会给代码不讲解。"""
    system = FIX_SYSTEM
    for field in ("`what`", "`why`", "`how`", "`avoid`"):
        assert field in system
    assert "原来错在哪里" in system
    assert "以后如何避免" in system


def test_prompt_contains_category_definitions() -> None:
    for category in ("syntax", "logic", "risk", "style"):
        assert f'"{category}"' in FIX_SYSTEM


def test_prompt_forbids_unrelated_refactor() -> None:
    """不许顺手重构：不加约束模型会把变量名、风格一起改掉，学生认不出自己的代码。"""
    assert "不要顺手重构" in FIX_SYSTEM
    assert "保留学生原有的注释" in FIX_SYSTEM


def test_prompt_forbids_fabrication() -> None:
    assert "原文摘抄" in FIX_SYSTEM
    assert "不要猜" in FIX_SYSTEM


def test_prompt_handles_no_error_case() -> None:
    """没错误时要明说，不能为了改而改。"""
    assert "不要为了改而改" in FIX_SYSTEM


def test_prompt_states_audience() -> None:
    assert "大一新生" in FIX_SYSTEM
    assert "通俗" in FIX_SYSTEM


def test_user_message_contains_code_and_language() -> None:
    messages = build_messages(
        BROKEN_PY, language="python", filename="avg.py", local_analysis="- 无"
    )
    user = messages[1].content
    assert "```python" in user
    assert "def average" in user
    assert "avg.py" in user


def test_prompt_template_has_placeholder_replaced() -> None:
    """占位符必须被真实替换掉——留着 <<LOCAL_ANALYSIS>> 会让模型看到模板标记。"""
    messages = build_messages(BROKEN_PY, language="python", filename="a.py", local_analysis="- 无")
    assert "<<LOCAL_ANALYSIS>>" not in messages[0].content
    assert "- 无" in messages[0].content


def test_prompt_version_defined() -> None:
    assert PROMPT_VERSION.startswith("code-fix/")


# ===========================================================================
# 2. 本地分析（第 1 步）
# ===========================================================================
def test_local_analysis_finds_syntax_error() -> None:
    fixer = _fixer(FixFakeLLM())
    local = fixer.analyze_locally(SHELL_PY, language="python", filename="a.py")

    assert local.syntax_ok is False
    assert local.syntax_error is not None
    assert local.syntax_error.line == 1
    assert local.syntax_error.tool == "ast"


def test_local_analysis_finds_static_risks() -> None:
    """本地风险规则与检测模块是同一套，两个功能不能给出互相矛盾的结论。"""
    fixer = _fixer(FixFakeLLM())
    local = fixer.analyze_locally(RISKY_C, language="c", filename="a.c")

    assert local.syntax_ok is True
    assert any("gets" in item.title for item in local.issues)
    assert all(item.source == "local" for item in local.issues)


def test_local_analysis_of_clean_code() -> None:
    fixer = _fixer(FixFakeLLM())
    local = fixer.analyze_locally(GOOD_PY, language="python", filename="a.py")
    assert local.syntax_ok is True


# ===========================================================================
# 3. 本地复检（第 3 步）—— 本模块最关键的诚实性设计
# ===========================================================================
def test_verification_passes_when_syntax_error_removed() -> None:
    fixer = _fixer(FixFakeLLM())
    local = fixer.analyze_locally(SHELL_PY, language="python", filename="a.py")

    result = fixer.verify_fix(
        FIXED_SHELL_PY, language="python", filename="a.py", before=local.syntax_error
    )

    assert result.verified is True
    assert result.syntax_after is None
    assert "语法检查" in result.note
    assert "第 1 行" in result.note


def test_verification_fails_when_fix_is_still_broken() -> None:
    """"AI 说改好了"不等于真的改好了——这就是本步骤存在的全部理由。"""
    fixer = _fixer(FixFakeLLM())
    local = fixer.analyze_locally(STILL_BROKEN_C, language="c", filename="a.c")

    result = fixer.verify_fix(
        STILL_BROKEN_C, language="c", filename="a.c", before=local.syntax_error
    )

    assert result.verified is False
    assert result.syntax_after is not None
    assert "仍然有语法错误" in result.note


def test_verification_note_states_logic_is_not_verified() -> None:
    """逻辑错误无法本地验证，必须在 note 里说清边界，否则学生以为可以直接交作业。"""
    fixer = _fixer(FixFakeLLM())

    result = fixer.verify_fix(BROKEN_PY, language="python", filename="a.py", before=None)

    assert result.verified is True
    assert "逻辑" in result.note
    assert "无法自动验证" in result.note


def test_verification_fails_on_empty_fix() -> None:
    fixer = _fixer(FixFakeLLM())
    result = fixer.verify_fix("   ", language="python", filename="a.py", before=None)

    assert result.verified is False
    assert "没有返回可用的代码" in result.note


# ===========================================================================
# 4. diff
# ===========================================================================
def test_diff_stats_counts_changes() -> None:
    stats = diff_stats("a\nb\nc\n", "a\nB\nc\nd\n")
    assert stats.changed == 1      # b -> B
    assert stats.added == 1        # 新增 d
    assert stats.removed == 0
    assert stats.unchanged == 2
    assert stats.total_changed == 2


def test_diff_stats_zero_for_identical_code() -> None:
    stats = diff_stats(GOOD_PY, GOOD_PY)
    assert stats.total_changed == 0
    assert stats.unchanged == len(GOOD_PY.splitlines())


def test_unified_diff_has_headers_and_changes() -> None:
    diff = build_unified_diff("x = 1\n", "x = 2\n", filename="a.py")
    assert "--- a/a.py" in diff
    assert "+++ b/a.py" in diff
    assert "-x = 1" in diff
    assert "+x = 2" in diff


def test_unified_diff_empty_when_identical() -> None:
    assert build_unified_diff(GOOD_PY, GOOD_PY, filename="a.py") == ""


def test_unified_diff_is_truncated() -> None:
    original = "\n".join(f"line{i}" for i in range(MAX_DIFF_LINES * 2))
    fixed = "\n".join(f"changed{i}" for i in range(MAX_DIFF_LINES * 2))

    diff = build_unified_diff(original, fixed, filename="a.py")

    assert "差异过长" in diff
    assert len(diff.splitlines()) <= MAX_DIFF_LINES + 2


# ===========================================================================
# 5. 完整流程：本地分析 + AI + 复检
# ===========================================================================
async def test_fix_returns_four_question_changes() -> None:
    fixer = _fixer(FixFakeLLM())
    outcome = await fixer.fix(BROKEN_PY, language="python", filename="avg.py")

    assert outcome.had_error is True
    assert outcome.change_count == 2
    assert outcome.ai_available is True
    assert outcome.model == "fake-fixer"

    first = outcome.changes[0]
    assert first.what and first.why and first.how and first.avoid
    assert first.category == "logic"
    assert first.line == 3
    assert first.original and first.fixed

    assert outcome.categories == {"logic": 1, "risk": 1}
    assert outcome.summary
    assert outcome.diff and outcome.diff_stats.total_changed > 0
    assert outcome.verification.verified is True


async def test_fix_verifies_the_model_output() -> None:
    """模型返回仍然有语法错误的代码时，复检必须拦下来。"""
    fixer = _fixer(FixFakeLLM(BAD_FIX_REPLY))
    outcome = await fixer.fix(RISKY_C, language="c", filename="a.c")

    assert outcome.verification.verified is False
    assert outcome.verification.syntax_after is not None
    assert "仍然有语法错误" in outcome.verification.note


async def test_fix_keeps_original_code_when_model_returns_nothing() -> None:
    """模型没给代码时不能返回空字符串——那会让前端显示成一片空白，像把学生的代码弄丢了。"""
    reply = json.dumps(
        {"had_error": True, "summary": "有问题", "fixed_code": "", "changes": []},
        ensure_ascii=False,
    )
    fixer = _fixer(FixFakeLLM(reply))
    outcome = await fixer.fix(BROKEN_PY, language="python", filename="a.py")

    assert outcome.fixed_code == BROKEN_PY
    assert any("没有返回修正后的代码" in item for item in outcome.warnings)


async def test_fix_reports_no_error_honestly() -> None:
    fixer = _fixer(FixFakeLLM(NO_ERROR_REPLY))
    outcome = await fixer.fix(GOOD_PY, language="python", filename="a.py")

    assert outcome.had_error is False
    assert outcome.changes == []
    assert outcome.fixed_code == GOOD_PY
    assert outcome.diff == ""


async def test_fix_without_ai_returns_local_analysis() -> None:
    """没配模型时也要能用：返回原代码 + 本地分析结论 + 明确提示。"""
    fixer = _fixer(OfflineLLM())
    outcome = await fixer.fix(SHELL_PY, language="python", filename="a.py")

    assert outcome.ai_available is False
    assert outcome.fixed_code == SHELL_PY          # 原代码原样返回
    assert outcome.changes == []
    # 提示语统一指向"网页上填 Key"
    assert "请先在网页上输入 API Key" in outcome.note
    assert outcome.local is not None and outcome.local.syntax_error is not None
    # 本地复检仍然要给出结论（对原代码来说：语法不通过）
    assert outcome.verification.verified is False


async def test_fix_survives_ai_failure() -> None:
    fixer = _fixer(FixFakeLLM(fail_with=RuntimeError("网关 502")))
    outcome = await fixer.fix(BROKEN_PY, language="python", filename="a.py")

    assert outcome.ai_available is False
    assert outcome.fixed_code == BROKEN_PY
    assert any("502" in item for item in outcome.warnings)
    assert "调用失败" in outcome.note


async def test_fix_survives_non_json_reply() -> None:
    fixer = _fixer(FixFakeLLM("这段代码看起来没问题。"))
    outcome = await fixer.fix(BROKEN_PY, language="python", filename="a.py")

    assert outcome.ai_available is False
    assert any("JSON" in item for item in outcome.warnings)


async def test_fix_drops_hallucinated_line_numbers() -> None:
    reply = json.dumps(
        {
            "had_error": True,
            "fixed_code": "x = 2\n",
            "changes": [{"line": 9999, "what": "越界行号"}],
            "summary": "",
        },
        ensure_ascii=False,
    )
    fixer = _fixer(FixFakeLLM(reply))
    outcome = await fixer.fix("x = 1\n", language="python", filename="a.py")
    assert outcome.changes[0].line is None


async def test_fix_normalizes_category() -> None:
    reply = json.dumps(
        {
            "had_error": True,
            "fixed_code": "x = 2\n",
            "changes": [
                {"line": 1, "category": "逻辑错误", "what": "中文类型"},
                {"line": 1, "category": "诡异的值", "what": "非法类型"},
            ],
            "summary": "",
        },
        ensure_ascii=False,
    )
    fixer = _fixer(FixFakeLLM(reply))
    outcome = await fixer.fix("x = 1\n", language="python", filename="a.py")
    assert outcome.changes[0].category == "logic"
    assert outcome.changes[1].category == "logic"


async def test_fix_drops_changes_without_what() -> None:
    reply = json.dumps(
        {
            "had_error": True,
            "fixed_code": "x = 2\n",
            "changes": [{"title": "  "}, "不是字典", {"what": "有效说明"}],
            "summary": "",
        },
        ensure_ascii=False,
    )
    fixer = _fixer(FixFakeLLM(reply))
    outcome = await fixer.fix("x = 1\n", language="python", filename="a.py")
    assert [item.what for item in outcome.changes] == ["有效说明"]


async def test_fix_warns_on_incomplete_explanation() -> None:
    """缺 why/avoid 的说明对学生价值大打折扣，必须告警。"""
    reply = json.dumps(
        {
            "had_error": True,
            "fixed_code": "x = 2\n",
            "changes": [{"line": 1, "what": "只说了错在哪"}],
            "summary": "",
        },
        ensure_ascii=False,
    )
    fixer = _fixer(FixFakeLLM(reply))
    outcome = await fixer.fix("x = 1\n", language="python", filename="a.py")
    assert any("不完整" in item for item in outcome.warnings)


async def test_fix_cross_checks_model_against_local_syntax_error() -> None:
    """本地查出语法错误、模型却说"没问题"时，必须点破（语法是确定性事实）。"""
    reply = json.dumps(
        {"had_error": False, "summary": "没发现问题", "fixed_code": SHELL_PY, "changes": []},
        ensure_ascii=False,
    )
    fixer = _fixer(FixFakeLLM(reply))
    outcome = await fixer.fix(SHELL_PY, language="python", filename="a.py")

    assert any("但模型认为无需修改" in item for item in outcome.warnings)
    assert outcome.note


async def test_fix_notes_when_model_changed_nothing() -> None:
    """模型说有错却没改动任何一行时，也要说清楚。"""
    reply = json.dumps(
        {
            "had_error": True,
            "summary": "已修正",
            "fixed_code": BROKEN_PY,
            "changes": [{"line": 3, "what": "循环边界", "why": "越界", "how": "改", "avoid": "注意"}],
        },
        ensure_ascii=False,
    )
    fixer = _fixer(FixFakeLLM(reply))
    outcome = await fixer.fix(BROKEN_PY, language="python", filename="a.py")

    assert any("完全相同" in item for item in outcome.warnings)


async def test_fix_rejects_empty_code() -> None:
    from app.services.code_fixer import CodeFixerError

    fixer = _fixer(FixFakeLLM())
    with pytest.raises(CodeFixerError):
        await fixer.fix("   \n", language="python", filename="a.py")


async def test_fix_rejects_oversized_code() -> None:
    from app.services.code_checker import MAX_CODE_CHARS
    from app.services.code_fixer import CodeFixerError

    fixer = _fixer(FixFakeLLM())
    with pytest.raises(CodeFixerError):
        await fixer.fix("x = 1\n" * (MAX_CODE_CHARS // 6 + 10), language="python", filename="a.py")


async def test_fix_text_defaults_and_infers() -> None:
    fixer = _fixer(FixFakeLLM())
    outcome = await fixer.fix_text("int main(void) { return 0; }\n", language="c", filename=None)
    assert outcome.filename == "snippet.c"

    inferred = await fixer.fix_text("int main(void) { return 0; }\n", language=None, filename="m.c")
    assert inferred.language == "c"


async def test_fix_prompt_carries_real_local_analysis() -> None:
    """端到端确认：真实代码的本地结论确实进了发给模型的消息。"""
    llm = FixFakeLLM()
    fixer = _fixer(llm)
    await fixer.fix(SHELL_PY, language="python", filename="a.py")

    assert "expected ':'" in llm.system_prompt or "语法" in llm.system_prompt
    assert "```python" in llm.user_prompt


# ===========================================================================
# 6. SQLite 落库与历史（需求 5：方便对比学习）
# ===========================================================================
def test_save_stores_original_and_fixed(sqlite_path: Path) -> None:
    outcome = _run_async(_fixer(FixFakeLLM()).fix(BROKEN_PY, language="python", filename="avg.py"))

    async def scenario(session):
        record = await CodeFixService.save(session, outcome, trace_id="t-fix")
        rows, total = await CodeFixService.list_records(session)
        return record, rows, total

    record, rows, total = _run_db(scenario)

    assert record.id is not None
    assert total == 1
    saved = rows[0]
    # 对比学习的关键：原代码与新代码都必须存下来
    assert saved.original_code == BROKEN_PY
    assert saved.fixed_code == outcome.fixed_code
    assert saved.fixed_code != saved.original_code
    assert saved.had_error is True
    assert saved.change_count == 2
    assert saved.verified is True
    assert saved.diff_text
    assert saved.added_lines > 0
    assert saved.trace_id == "t-fix"

    changes = json.loads(saved.changes_json)
    assert len(changes) == 2
    assert all(item["what"] and item["why"] and item["how"] and item["avoid"] for item in changes)
    assert json.loads(saved.categories_json) == {"logic": 1, "risk": 1}


def test_save_records_syntax_error_before_fix(sqlite_path: Path) -> None:
    outcome = _run_async(
        _fixer(FixFakeLLM()).fix(SHELL_PY, language="python", filename="a.py")
    )

    async def scenario(session):
        return await CodeFixService.save(session, outcome)

    record = _run_db(scenario)
    assert record.syntax_error is not None
    assert "expected ':'" in record.syntax_error


def test_save_marks_unverified_fix(sqlite_path: Path) -> None:
    outcome = _run_async(
        _fixer(FixFakeLLM(BAD_FIX_REPLY)).fix(RISKY_C, language="c", filename="a.c")
    )

    async def scenario(session):
        return await CodeFixService.save(session, outcome)

    record = _run_db(scenario)
    assert record.verified is False
    assert "仍然有语法错误" in (record.verification_note or "")


def test_list_records_filters(sqlite_path: Path) -> None:
    py = _run_async(_fixer(FixFakeLLM()).fix(BROKEN_PY, language="python", filename="avg.py"))
    c = _run_async(_fixer(FixFakeLLM()).fix(RISKY_C, language="c", filename="a.c"))

    async def scenario(session):
        await CodeFixService.save(session, py)
        await CodeFixService.save(session, c)
        only_c, total_c = await CodeFixService.list_records(session, language="c")
        by_name, total_name = await CodeFixService.list_records(session, filename="avg.py")
        page, total_all = await CodeFixService.list_records(session, limit=1)
        return only_c, total_c, by_name, total_name, page, total_all

    only_c, total_c, by_name, total_name, page, total_all = _run_db(scenario)
    assert total_all == 2
    assert total_c == 1 and only_c[0].filename == "a.c"
    assert total_name == 1 and by_name[0].language == "python"
    assert len(page) == 1


def test_get_record(sqlite_path: Path) -> None:
    outcome = _run_async(_fixer(FixFakeLLM()).fix(BROKEN_PY, language="python", filename="avg.py"))

    async def scenario(session):
        saved = await CodeFixService.save(session, outcome)
        return await CodeFixService.get_record(session, saved.id)

    fetched = _run_db(scenario)
    assert fetched is not None and fetched.fixed_code == outcome.fixed_code


# ===========================================================================
# 7. API
# ===========================================================================
@pytest.fixture()
def fix_app() -> Iterator[object]:
    """注入假 LLM 的应用。

    假客户端在 lambda 外面创建一次并共用——写在 lambda 里的话，
    每次解析依赖都会新建一个，断言拿不到之前的调用记录。
    """
    app = create_app()
    settings = get_settings()
    fake = FixFakeLLM()
    app.dependency_overrides[get_code_fixer] = lambda: CodeFixer(settings, fake)
    yield app, fake
    app.dependency_overrides.clear()


def test_api_returns_fixed_code_and_four_questions(sqlite_path: Path, fix_app) -> None:
    app, fake = fix_app
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/fix/code",
            json={"code": BROKEN_PY, "language": "python", "filename": "avg.py"},
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["language"] == "python"
    assert body["language_label"] == "Python"
    assert body["had_error"] is True
    assert "for score in scores" in body["fixed_code"]
    assert len(body["changes"]) == 2

    change = body["changes"][0]
    for field in ("what", "why", "how", "avoid"):
        assert change[field], f"四问字段 {field} 不能为空"
    assert change["category"] == "logic"

    assert body["categories"] == {"logic": 1, "risk": 1}
    assert body["diff"] and body["diff_stats"]["total_changed"] > 0
    assert body["verification"]["verified"] is True
    assert body["verification"]["note"]
    assert body["local_issues"] is not None
    assert body["record_id"] is not None
    assert body["trace_id"]
    assert fake.calls, "AI 应该被调用过"


def test_api_reports_local_syntax_error(sqlite_path: Path, fix_app) -> None:
    app, _ = fix_app
    with TestClient(app) as client:
        body = client.post(
            "/api/v1/fix/code", json={"code": SHELL_PY, "language": "python"}
        ).json()

    before = body["verification"]["syntax_before"]
    assert before is not None
    assert before["tool"] == "ast"
    assert before["line"] == 1


def test_api_c_syntax_error_uses_tree_sitter(sqlite_path: Path, fix_app) -> None:
    app, _ = fix_app
    with TestClient(app) as client:
        body = client.post(
            "/api/v1/fix/code", json={"code": "int main() { return 0 }\n", "language": "C"}
        ).json()

    assert body["language"] == "c"
    assert body["verification"]["syntax_before"]["tool"] == "tree-sitter"


def test_api_infers_language_from_filename(sqlite_path: Path, fix_app) -> None:
    app, _ = fix_app
    with TestClient(app) as client:
        body = client.post(
            "/api/v1/fix/code", json={"code": "int main(void) { return 0; }\n", "filename": "m.c"}
        ).json()
    assert body["language"] == "c"
    assert body["filename"] == "m.c"


def test_api_rejects_unsupported_language(sqlite_path: Path, fix_app) -> None:
    app, _ = fix_app
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/fix/code", json={"code": "package main\n", "language": "go"}
        )
    assert response.status_code == 400
    assert "不支持的语言" in response.json()["detail"]


def test_api_rejects_whitespace_only_code(sqlite_path: Path, fix_app) -> None:
    app, _ = fix_app
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/fix/code", json={"code": "   \n\n", "language": "python"}
        )
    assert response.status_code == 400


def test_api_works_without_llm(sqlite_path: Path) -> None:
    """没配模型时返回 200 + 本地分析结论（而不是 503）。"""
    app = create_app()
    settings = get_settings()
    app.dependency_overrides[get_code_fixer] = lambda: CodeFixer(settings, OfflineLLM())
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/fix/code", json={"code": SHELL_PY, "language": "python"}
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["ai_available"] is False
    assert body["fixed_code"] == SHELL_PY       # 原代码原样返回，不能丢
    assert body["note"]


# ---- 历史接口：需求 5 的"方便对比学习" ----
def test_history_flow(sqlite_path: Path, fix_app) -> None:
    app, _ = fix_app
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/fix/code",
            json={"code": BROKEN_PY, "language": "python", "filename": "avg.py"},
        ).json()
        listed = client.get("/api/v1/fix/history").json()
        detail = client.get(f"/api/v1/fix/history/{created['record_id']}").json()

    assert listed["total"] == 1
    item = listed["items"][0]
    assert item["filename"] == "avg.py"
    assert item["change_count"] == 2
    # 列表不该带代码全文（会太重），详情才带
    assert "original_code" not in item

    # 详情：原代码、新代码、diff、四问说明都齐全，才能左右对比
    assert detail["original_code"] == BROKEN_PY
    assert detail["fixed_code"] == created["fixed_code"]
    assert detail["diff"]
    assert detail["verification_note"]
    assert len(detail["changes"]) == 2
    assert detail["changes"][0]["avoid"]
    assert detail["categories"] == {"logic": 1, "risk": 1}


def test_history_detail_404(sqlite_path: Path, fix_app) -> None:
    app, _ = fix_app
    with TestClient(app) as client:
        response = client.get("/api/v1/fix/history/9999")
    assert response.status_code == 404


def test_history_filters(sqlite_path: Path, fix_app) -> None:
    app, _ = fix_app
    with TestClient(app) as client:
        client.post("/api/v1/fix/code", json={"code": BROKEN_PY, "language": "python", "filename": "a.py"})
        client.post("/api/v1/fix/code", json={"code": RISKY_C, "language": "c", "filename": "b.c"})

        only_c = client.get("/api/v1/fix/history", params={"language": "c"}).json()
        by_name = client.get("/api/v1/fix/history", params={"filename": "a.py"}).json()
        page = client.get("/api/v1/fix/history", params={"limit": 1}).json()

    assert only_c["total"] == 1 and only_c["items"][0]["filename"] == "b.c"
    assert by_name["total"] == 1 and by_name["items"][0]["language"] == "python"
    assert len(page["items"]) == 1
