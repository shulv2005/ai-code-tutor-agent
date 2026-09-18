"""AI 自动检测测试：本地语法检查、风格规则、风险模式、Prompt 构造、AI 合并、API 与落库。

大模型一律用可注入的假客户端替代，因此整条链路离线可测；
本地静态检查这部分则完全不依赖网络，是真跑 ast / tree-sitter。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_code_checker
from app.core.config import get_settings
from app.core.database import dispose_engine, get_session, init_db
from app.core.llm_client import LLMMessage, LLMResponse, LLMUsage
from app.main import create_app
from app.services.code_check_service import CodeCheckService
from app.services.code_checker import (
    DEEP_CHECK_SYSTEM,
    PROMPT_VERSION,
    CodeChecker,
    UnsupportedLanguageError,
    build_messages,
    check_naming,
    check_python_ast_risks,
    check_risk_patterns,
    compute_metrics,
    count_lines_by_kind,
    locate_syntax_error,
    normalize_language,
)
from app.services.repo.parser import parse_source

# ---------------------------------------------------------------------------
# 样例代码
# ---------------------------------------------------------------------------
GOOD_PY = '''"""学生成绩统计。"""


def average(scores):
    """求平均分。"""
    if not scores:
        return 0.0
    total = 0
    for score in scores:
        total += score
    return total / len(scores)


def main():
    """程序入口。"""
    print(average([88, 92, 79]))


if __name__ == "__main__":
    main()
'''

BROKEN_PY = "def broken(:\n    pass\n"

RISKY_PY = '''def collect(items=[]):
    """把元素收集起来。"""
    items.append(1)
    return items


def read(path):
    """读文件。"""
    try:
        handle = open(path, encoding="utf-8")
        return handle.read()
    except:
        return None


def is_empty(text):
    """判断是否为空。"""
    return text == None
'''

RISKY_C = '''#include <stdio.h>
#include <string.h>

int main(void) {
    char buf[10];
    gets(buf);
    strcpy(buf, "hello");
    return 0;
}
'''

RISKY_JAVA = '''public class Demo {
    public boolean same(String a, String b) {
        if (a == "x") {
            return true;
        }
        try {
            Integer.parseInt(a);
        } catch (Exception e) {
        }
        return false;
    }
}
'''

MESSY_PY = "\n".join(
    ["def f(x):", "\tif x:", "        return 1", "    return 0", "   ", "x = 1   "]
)

# AI 的模拟回复：覆盖四类输出
AI_REPLY = json.dumps(
    {
        "score": 68,
        "summary": "能跑通，但边界情况没考虑，风格也需要打磨。",
        "errors": [
            {
                "line": 3,
                "severity": "error",
                "title": "循环边界差一",
                "detail": "range 到 n 会多访问一个元素。",
                "suggestion": "改成 range(n - 1)",
            }
        ],
        "style": [
            {
                "line": None,
                "severity": "info",
                "title": "变量名 total 可以更具体",
                "detail": "看不出是什么的总和。",
                "suggestion": "改成 score_total",
            }
        ],
        "risks": [
            {
                "line": 5,
                "severity": "warning",
                "title": "没有处理空列表",
                "detail": "空列表时除法会出错。",
                "suggestion": "先判断 if not scores: return 0",
            }
        ],
        "advice": ["写完循环先检查起点和终点", "边界情况（空、一个元素）要单独试一试"],
        "highlights": ["函数拆分得清楚"],
    },
    ensure_ascii=False,
)


class CheckFakeLLM:
    """可编排的假 LLM：记录调用、按固定内容回复。"""

    def __init__(self, reply: str = AI_REPLY, *, fail_with: Exception | None = None) -> None:
        self.reply = reply
        self.fail_with = fail_with
        self.calls: list[list[LLMMessage]] = []

    @property
    def model(self) -> str:
        return "fake-checker"

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
            model="fake-checker",
            finish_reason="stop",
            usage=LLMUsage(prompt_tokens=800, completion_tokens=200, total_tokens=1000),
            latency_ms=12.0,
        )

    async def aclose(self) -> None:
        return None


class OfflineLLM(CheckFakeLLM):
    """自称"未配置"的假客户端，用于验证无 AI 时的降级路径。"""

    @property
    def configured(self) -> bool:
        return False


def _checker(llm: CheckFakeLLM) -> CodeChecker:
    return CodeChecker(get_settings(), llm)


def _metrics(code: str, language: str, filename: str):
    return compute_metrics(code, language, parse_source(code.encode(), filename, language))


# ===========================================================================
# 1. 语言归一
# ===========================================================================
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("c", "c"),
        ("C", "c"),
        ("c11", "c"),
        ("java", "java"),
        ("Java", "java"),
        ("python", "python"),
        ("PYTHON3", "python"),
        (" py ", "python"),
    ],
)
def test_normalize_language_accepts_aliases(value: str, expected: str) -> None:
    assert normalize_language(value) == expected


def test_normalize_language_falls_back_to_filename() -> None:
    """语言留空时按文件后缀推断，学生少选一项也不会报错。"""
    assert normalize_language(None, "homework.py") == "python"
    assert normalize_language("", "Main.java") == "java"
    assert normalize_language(None, "sort.c") == "c"


@pytest.mark.parametrize("bad", ["go", "javascript", "ruby", "", None])
def test_normalize_language_rejects_unsupported(bad: str | None) -> None:
    if bad in ("", None):
        with pytest.raises(UnsupportedLanguageError):
            normalize_language(bad)   # 既没给语言也没给文件名
    else:
        with pytest.raises(UnsupportedLanguageError):
            normalize_language(bad)


def test_unsupported_language_message_lists_supported() -> None:
    with pytest.raises(UnsupportedLanguageError) as excinfo:
        normalize_language("go")
    assert "c" in str(excinfo.value) and "java" in str(excinfo.value)


# ===========================================================================
# 2. 语法检查（阶段 1 的核心）
# ===========================================================================
def test_python_syntax_ok_returns_none() -> None:
    assert locate_syntax_error(GOOD_PY, language="python", filename="a.py") is None


def test_python_syntax_error_has_line_and_column() -> None:
    """Python 走 ast，报错位置必须精确到行列——这是给学生看的关键信息。"""
    info = locate_syntax_error(BROKEN_PY, language="python", filename="a.py")

    assert info is not None
    assert info.tool == "ast"
    assert info.line == 1
    assert info.column is not None and info.column > 0
    assert "invalid syntax" in info.raw


def test_python_syntax_error_reports_second_line() -> None:
    code = "x = 1\ny = = 2\n"
    info = locate_syntax_error(code, language="python", filename="a.py")
    assert info is not None and info.line == 2


def test_c_syntax_error_located_by_tree_sitter() -> None:
    """C 走 tree-sitter：容错解析不抛异常，靠 ERROR/MISSING 节点定位。"""
    info = locate_syntax_error("int main() { return 0 }\n", language="c", filename="a.c")

    assert info is not None
    assert info.tool == "tree-sitter"
    assert info.line == 1
    assert "语法错误" in info.message


def test_valid_c_has_no_syntax_error() -> None:
    code = "#include <stdio.h>\n\nint main(void) {\n    return 0;\n}\n"
    assert locate_syntax_error(code, language="c", filename="a.c") is None


def test_java_syntax_error_located() -> None:
    info = locate_syntax_error(
        "public class A {\n    int f() { return 1 }\n}\n", language="java", filename="A.java"
    )
    assert info is not None
    assert info.line == 2


def test_valid_java_has_no_syntax_error() -> None:
    code = "public class A {\n    int f() { return 1; }\n}\n"
    assert locate_syntax_error(code, language="java", filename="A.java") is None


# ===========================================================================
# 3. 统计指标
# ===========================================================================
def test_count_lines_by_kind_python() -> None:
    code = "# 注释\n\nx = 1\ny = 2  # 行内注释\n"
    code_lines, comment_lines, blank_lines = count_lines_by_kind(code, "python")
    assert (code_lines, comment_lines, blank_lines) == (2, 1, 1)


def test_count_lines_by_kind_c_block_comment() -> None:
    code = "/* 多行\n   注释 */\nint main(void) { return 0; }\n"
    code_lines, comment_lines, blank_lines = count_lines_by_kind(code, "c")
    assert (code_lines, comment_lines, blank_lines) == (1, 2, 0)


def test_metrics_report_symbols_and_complexity() -> None:
    metrics = _metrics(GOOD_PY, "python", "a.py")
    names = [item["name"] for item in metrics.symbols]
    assert names == ["average", "main"]
    assert metrics.function_count == 2
    assert metrics.max_complexity >= 2      # average 里有 if + for
    assert metrics.longest_function_name in {"average", "main"}


def test_metrics_detect_mixed_indent() -> None:
    metrics = _metrics(MESSY_PY, "python", "a.py")
    assert metrics.mixed_indent is True
    assert metrics.uses_tabs is True


def test_metrics_comment_ratio() -> None:
    metrics = _metrics(GOOD_PY, "python", "a.py")
    assert 0 < metrics.comment_ratio < 1


# ===========================================================================
# 4. 风格规则与命名
# ===========================================================================
def test_naming_flags_camel_case_function_in_python() -> None:
    code = "def addNumbers(a, b):\n    return a + b\n"
    issues = check_naming(_metrics(code, "python", "a.py"), "python")

    assert any("addNumbers" in item.title for item in issues)
    assert any("add_numbers" in item.suggestion for item in issues)


def test_naming_flags_lowercase_class_in_python() -> None:
    code = "class student:\n    pass\n"
    issues = check_naming(_metrics(code, "python", "a.py"), "python")
    assert any("student" in item.title for item in issues)


def test_naming_accepts_conventional_names() -> None:
    assert check_naming(_metrics(GOOD_PY, "python", "a.py"), "python") == []


def test_naming_java_method_uppercase() -> None:
    code = "public class A {\n    int GetValue() { return 1; }\n}\n"
    issues = check_naming(_metrics(code, "java", "A.java"), "java")
    assert any("GetValue" in item.title for item in issues)


def test_naming_java_class_lowercase() -> None:
    code = "public class demo {\n    int f() { return 1; }\n}\n"
    issues = check_naming(_metrics(code, "java", "demo.java"), "java")
    assert any("demo" in item.title for item in issues)


def test_naming_c_function_uppercase() -> None:
    code = "int Add(int a, int b) { return a + b; }\n"
    issues = check_naming(_metrics(code, "c", "a.c"), "c")
    assert any("Add" in item.title for item in issues)


# ===========================================================================
# 5. 风险模式
# ===========================================================================
def test_python_ast_detects_mutable_default() -> None:
    issues = check_python_ast_risks(RISKY_PY)
    assert any("默认参数" in item.title for item in issues)
    # 行号要指向函数定义那一行（第 1 行）
    assert any(item.line == 1 for item in issues if "默认参数" in item.title)


def test_python_ast_detects_bare_except() -> None:
    issues = check_python_ast_risks(RISKY_PY)
    assert any("裸 except" in item.title for item in issues)


def test_python_ast_detects_none_compare() -> None:
    issues = check_python_ast_risks(RISKY_PY)
    assert any("None" in item.title for item in issues)


def test_python_ast_risk_issues_do_not_crash_on_broken_code() -> None:
    """语法都不对时不要试图做 AST 检查（语法错误由语法检查单独报）。"""
    assert check_python_ast_risks(BROKEN_PY) == []


def test_c_risk_patterns_detected() -> None:
    issues = check_risk_patterns(RISKY_C, "c")
    titles = " ".join(item.title for item in issues)
    assert "gets" in titles
    assert "strcpy" in titles
    assert all(item.severity == "warning" for item in issues)


def test_java_risk_patterns_detected() -> None:
    issues = check_risk_patterns(RISKY_JAVA, "java")
    titles = " ".join(item.title for item in issues)
    assert "==" in titles           # 字符串用 == 比较
    assert "catch" in titles        # 空 catch


def test_risk_patterns_ignore_comments_and_strings() -> None:
    """注释里提到 strcpy 不该被报成风险 —— 这是最容易误报的地方。"""
    code = '// 不要用 strcpy(buf, src)，要用 strncpy\nint main(void) { return 0; }\n'
    issues = check_risk_patterns(code, "c")
    assert issues == []


def test_risk_patterns_report_correct_line() -> None:
    code = "#include <stdio.h>\nint main(void) {\n    char buf[4];\n    gets(buf);\n}\n"
    issues = check_risk_patterns(code, "c")
    assert [item.line for item in issues] == [4]


# ===========================================================================
# 6. Prompt 构造（需求里点名要讲清的部分）
# ===========================================================================
def test_prompt_injects_local_findings() -> None:
    """本地结论必须进提示词：这是"两阶段"能省钱又更深的前提。"""
    findings = "- 语法：**不通过**，第 1 行有语法错误（ast 报告：invalid syntax）"
    messages = build_messages(
        BROKEN_PY, language="python", filename="a.py", local_findings=findings
    )
    system = messages[0].content
    assert "invalid syntax" in system
    assert "第 1 行" in system


def test_prompt_asks_not_to_repeat_local_findings() -> None:
    messages = build_messages(GOOD_PY, language="python", filename="a.py", local_findings="- 无")
    assert "不要重复报告" in messages[0].content


def test_prompt_defines_four_output_buckets() -> None:
    """四类输出的边界必须写进提示词，否则模型会把风格问题报成 error。"""
    system = DEEP_CHECK_SYSTEM
    for key in ("errors", "style", "risks", "advice"):
        assert key in system
    assert "绝不能" in system          # 风格问题不许升级为错误
    assert "评分" in system


def test_prompt_contains_score_anchors() -> None:
    """给评分锚点是为了压住"一律 80 分"的倾向。"""
    assert "90-100" in DEEP_CHECK_SYSTEM
    assert "0-39" in DEEP_CHECK_SYSTEM


def test_prompt_forbids_hallucination() -> None:
    assert "不要猜" in DEEP_CHECK_SYSTEM
    assert "不存在的函数名" in DEEP_CHECK_SYSTEM


def test_prompt_states_audience() -> None:
    assert "大一新生" in DEEP_CHECK_SYSTEM
    assert "通俗" in DEEP_CHECK_SYSTEM


def test_user_message_contains_code_and_language() -> None:
    messages = build_messages(GOOD_PY, language="python", filename="a.py", local_findings="- 无")
    user = messages[1].content
    assert "```python" in user
    assert "def average" in user
    assert "a.py" in user


def test_prompt_version_is_defined() -> None:
    assert PROMPT_VERSION.startswith("code-check/")


# ===========================================================================
# 7. 完整检测（本地 + AI 合并）
# ===========================================================================
async def test_check_merges_local_and_ai_issues() -> None:
    """本地结论与 AI 结论都要在，且各自带 source 标记；本地排在前面。"""
    checker = _checker(CheckFakeLLM())
    outcome = await checker.check(RISKY_PY, language="python", filename="risk.py")

    assert outcome.ai_available is True
    assert outcome.model == "fake-checker"

    assert [item.title for item in outcome.errors] == ["循环边界差一"]
    assert [item.title for item in outcome.style] == ["变量名 total 可以更具体"]

    # 风险桶里两类来源都有：本地规则先合并，AI 的排在其后
    local_risks = [item.title for item in outcome.risks if item.source == "local"]
    ai_risks = [item.title for item in outcome.risks if item.source == "ai"]
    assert len(local_risks) == 3            # 可变默认参数、裸 except、== None
    assert ai_risks == ["没有处理空列表"]
    assert [item.source for item in outcome.risks][: len(local_risks)] == ["local"] * 3

    assert outcome.advice
    # 学习建议里的两条都要在（顺序由模型决定，不写死第一条）
    assert "边界" in " ".join(outcome.advice)
    assert outcome.highlights == ["函数拆分得清楚"]


async def test_check_marks_issue_source() -> None:
    checker = _checker(CheckFakeLLM())
    outcome = await checker.check(RISKY_PY, language="python", filename="risk.py")

    assert all(item.source == "ai" for item in outcome.errors)
    assert any(item.source == "local" for item in outcome.risks)
    assert all(item.source in ("local", "ai") for item in outcome.all_issues)


async def test_check_uses_ai_score() -> None:
    checker = _checker(CheckFakeLLM())
    outcome = await checker.check(GOOD_PY, language="python", filename="a.py")

    assert outcome.score == 68.0
    assert outcome.ai_score == 68.0
    assert outcome.level == "及格"
    assert "AI" in outcome.score_reason


async def test_syntax_error_caps_score() -> None:
    """有语法错误时分数必须被压下来：代码都跑不起来，给 90 分是不负责任的。"""
    reply = json.dumps(
        {"score": 95, "summary": "不错", "errors": [], "style": [], "risks": [], "advice": []},
        ensure_ascii=False,
    )
    checker = _checker(CheckFakeLLM(reply))
    outcome = await checker.check(BROKEN_PY, language="python", filename="a.py")

    assert outcome.local.syntax_ok is False
    assert outcome.score == 45.0
    assert "语法错误" in outcome.score_reason
    assert outcome.ai_score == 95.0        # 原始分保留，便于对比


async def test_check_without_ai_returns_local_results() -> None:
    """没配模型时也要能用：本地结论照样返回，并说明原因。"""
    checker = _checker(OfflineLLM())
    outcome = await checker.check(RISKY_PY, language="python", filename="risk.py")

    assert outcome.ai_available is False
    assert outcome.errors == []            # 本地检查不产生"错误"（那是 AI 的活）
    assert len(outcome.risks) == 3
    # 提示语现在统一指向"网页上填 Key"（以后端 .env 兜底），学生一眼知道去哪配
    assert "请先在网页上输入 API Key" in outcome.note
    assert 0 < outcome.score <= 100
    assert "本地规则评分" in outcome.score_reason


async def test_local_only_score_is_capped_below_excellent() -> None:
    """纯本地评分不给优秀档。

    回归：本地规则看不出逻辑错误，实测一份带真实越界 bug 的代码
    纯本地能拿到 92 分（优秀），这对学生是误导，所以封顶 85 并说明原因。
    """
    checker = _checker(OfflineLLM())
    outcome = await checker.check(GOOD_PY, language="python", filename="a.py")

    assert outcome.score == 85.0
    assert outcome.level == "良好"
    assert "最高只给 85 分" in outcome.score_reason


async def test_check_survives_ai_failure() -> None:
    """AI 调用炸了不能把整个请求带崩：降级为本地结论 + 一条 warning。"""
    checker = _checker(CheckFakeLLM(fail_with=RuntimeError("模型网关 502")))
    outcome = await checker.check(RISKY_PY, language="python", filename="risk.py")

    assert outcome.ai_available is False
    assert len(outcome.risks) == 3
    assert any("502" in item for item in outcome.warnings)
    assert "调用失败" in outcome.note


async def test_check_survives_non_json_reply() -> None:
    checker = _checker(CheckFakeLLM("这段代码看起来不错，没有什么问题。"))
    outcome = await checker.check(RISKY_PY, language="python", filename="risk.py")

    assert outcome.ai_available is False
    assert any("JSON" in item for item in outcome.warnings)


async def test_check_drops_hallucinated_line_numbers() -> None:
    """模型给的行号超出代码范围时必须丢弃，否则会指向不存在的地方。"""
    reply = json.dumps(
        {
            "score": 70,
            "summary": "",
            "errors": [{"line": 9999, "severity": "error", "title": "越界行号"}],
            "style": [{"line": -3, "severity": "info", "title": "负数行号"}],
            "risks": [{"line": "第5行", "severity": "warning", "title": "非法行号"}],
            "advice": [],
        },
        ensure_ascii=False,
    )
    checker = _checker(CheckFakeLLM(reply))
    outcome = await checker.check("x = 1\ny = 2\n", language="python", filename="a.py")

    assert [item.line for item in outcome.errors + outcome.style + outcome.risks] == [None, None, None]


async def test_check_normalizes_bad_severity() -> None:
    reply = json.dumps(
        {
            "score": 70,
            "errors": [{"line": 1, "severity": "严重", "title": "奇怪的程度"}],
            "style": [],
            "risks": [],
            "advice": [],
        },
        ensure_ascii=False,
    )
    checker = _checker(CheckFakeLLM(reply))
    outcome = await checker.check("x = 1\n", language="python", filename="a.py")
    assert outcome.errors[0].severity == "info"


async def test_check_ignores_empty_titles_and_bad_types() -> None:
    reply = json.dumps(
        {
            "score": 70,
            "errors": [{"title": "  "}, "不是字典", {"title": "有效问题"}],
            "style": "不是列表",
            "risks": None,
            "advice": "不是列表",
        },
        ensure_ascii=False,
    )
    checker = _checker(CheckFakeLLM(reply))
    outcome = await checker.check("x = 1\n", language="python", filename="a.py")

    assert [item.title for item in outcome.errors] == ["有效问题"]
    assert outcome.style == [] and outcome.risks == [] and outcome.advice == []


async def test_check_missing_score_falls_back_to_local() -> None:
    reply = json.dumps({"summary": "还行", "errors": [], "style": [], "risks": [], "advice": []})
    checker = _checker(CheckFakeLLM(reply))
    outcome = await checker.check(GOOD_PY, language="python", filename="a.py")

    assert outcome.ai_score is None
    assert "本地规则评分" in outcome.score_reason
    assert any("评分" in item for item in outcome.warnings)


async def test_check_rejects_empty_code() -> None:
    from app.services.code_checker import CodeCheckError

    checker = _checker(CheckFakeLLM())
    with pytest.raises(CodeCheckError):
        await checker.check("   \n  ", language="python", filename="a.py")


async def test_check_rejects_oversized_code() -> None:
    from app.services.code_checker import MAX_CODE_CHARS, CodeCheckError

    checker = _checker(CheckFakeLLM())
    with pytest.raises(CodeCheckError):
        await checker.check("x = 1\n" * (MAX_CODE_CHARS // 6 + 10), language="python", filename="a.py")


async def test_check_text_defaults_filename() -> None:
    checker = _checker(CheckFakeLLM())
    outcome = await checker.check_text("int main(void) { return 0; }\n", language="c", filename=None)
    assert outcome.filename == "snippet.c"
    assert outcome.language == "c"


async def test_check_text_infers_language_from_filename() -> None:
    checker = _checker(CheckFakeLLM())
    outcome = await checker.check_text("int main(void) { return 0; }\n", language=None, filename="m.c")
    assert outcome.language == "c"


async def test_check_prompt_contains_local_findings_for_real_code() -> None:
    """端到端确认：真实代码跑出来的本地结论确实进了发给模型的消息。"""
    llm = CheckFakeLLM()
    checker = _checker(llm)
    await checker.check(BROKEN_PY, language="python", filename="a.py")

    assert "invalid syntax" in llm.system_prompt
    assert "```python" in llm.user_prompt


# ===========================================================================
# 8. SQLite 落库
# ===========================================================================
def _run_db(scenario):
    """一个事件循环里跑完建表 → 场景 → 释放连接池（跨 loop 复用引擎会报错）。"""
    import asyncio

    async def main():
        await init_db()
        try:
            async with get_session() as session:
                return await scenario(session)
        finally:
            await dispose_engine()

    return asyncio.run(main())


async def _outcome(code: str, language: str, filename: str, llm=None):
    return await _checker(llm or CheckFakeLLM()).check(code, language=language, filename=filename)


def test_save_writes_all_metadata(sqlite_path: Path) -> None:
    """需求点：检测结果保存到 SQLite。"""
    outcome = _run_async(_outcome(RISKY_PY, "python", "risk.py"))

    async def scenario(session):
        record = await CodeCheckService.save(session, outcome, code=RISKY_PY, trace_id="t-1")
        rows, total = await CodeCheckService.list_records(session)
        return record, rows, total

    record, rows, total = _run_db(scenario)

    assert record.id is not None
    assert total == 1
    saved = rows[0]
    assert saved.filename == "risk.py"
    assert saved.language == "python"
    assert saved.code == RISKY_PY
    assert saved.score == outcome.score
    assert saved.ai_available is True
    assert saved.model == "fake-checker"
    assert saved.syntax_ok is True
    assert saved.error_count == 1 and saved.risk_count == 4 and saved.style_count == 1
    assert saved.trace_id == "t-1"
    assert saved.created_at is not None

    # JSON 字段要能解回来
    issues = json.loads(saved.issues_json)
    assert {item["source"] for item in issues} == {"local", "ai"}
    advice = json.loads(saved.advice_json)
    assert len(advice) >= 2 and any("边界" in item for item in advice)


def test_save_records_syntax_error(sqlite_path: Path) -> None:
    outcome = _run_async(_outcome(BROKEN_PY, "python", "a.py"))

    async def scenario(session):
        return await CodeCheckService.save(session, outcome, code=BROKEN_PY)

    record = _run_db(scenario)
    assert record.syntax_ok is False
    assert "invalid syntax" in (record.syntax_error or "")
    assert record.score == 45.0


def test_save_marks_local_only_result(sqlite_path: Path) -> None:
    outcome = _run_async(_outcome(RISKY_PY, "python", "risk.py", OfflineLLM()))

    async def scenario(session):
        return await CodeCheckService.save(session, outcome, code=RISKY_PY)

    record = _run_db(scenario)
    assert record.ai_available is False
    assert record.model is None


def test_list_records_filters(sqlite_path: Path) -> None:
    py = _run_async(_outcome(RISKY_PY, "python", "risk.py"))
    c = _run_async(_outcome(RISKY_C, "c", "main.c"))

    async def scenario(session):
        await CodeCheckService.save(session, py, code=RISKY_PY)
        await CodeCheckService.save(session, c, code=RISKY_C)
        only_c, total_c = await CodeCheckService.list_records(session, language="c")
        by_name, total_name = await CodeCheckService.list_records(session, filename="risk.py")
        page, total_all = await CodeCheckService.list_records(session, limit=1)
        return only_c, total_c, by_name, total_name, page, total_all

    only_c, total_c, by_name, total_name, page, total_all = _run_db(scenario)
    assert total_all == 2
    assert total_c == 1 and only_c[0].filename == "main.c"
    assert total_name == 1 and by_name[0].language == "python"
    assert len(page) == 1


def test_get_record(sqlite_path: Path) -> None:
    outcome = _run_async(_outcome(GOOD_PY, "python", "a.py"))

    async def scenario(session):
        saved = await CodeCheckService.save(session, outcome, code=GOOD_PY)
        return await CodeCheckService.get_record(session, saved.id)

    fetched = _run_db(scenario)
    assert fetched is not None and fetched.filename == "a.py"


def _run_async(coro):
    """跑一个协程（用于在同步测试里拿 outcome）。"""
    import asyncio

    return asyncio.run(coro)


# ===========================================================================
# 9. API
# ===========================================================================
@pytest.fixture()
def check_app() -> Iterator[object]:
    """注入假 LLM 的应用。

    假客户端在 lambda 外面创建一次并共用——写在 lambda 里的话，
    每次解析依赖都会新建一个，断言拿不到之前的调用记录。
    """
    app = create_app()
    settings = get_settings()
    fake = CheckFakeLLM()
    app.dependency_overrides[get_code_checker] = lambda: CodeChecker(settings, fake)
    yield app, fake
    app.dependency_overrides.clear()


def test_api_returns_four_buckets_and_score(sqlite_path: Path, check_app) -> None:
    app, fake = check_app
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/check/code",
            json={"code": RISKY_PY, "language": "python", "filename": "risk.py"},
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["language"] == "python"
    assert body["language_label"] == "Python"
    assert body["score"] == 68.0
    assert body["level"] == "及格"
    assert body["syntax_ok"] is True
    assert body["errors"][0]["title"] == "循环边界差一"
    assert body["style"][0]["title"] == "变量名 total 可以更具体"
    assert any(item["title"] == "没有处理空列表" for item in body["risks"])
    assert any(item["source"] == "local" for item in body["risks"])
    assert body["advice"]
    assert body["ai_available"] is True
    assert body["record_id"] is not None
    assert body["trace_id"]
    assert fake.calls, "AI 应该被调用过"


def test_api_returns_syntax_error_details(sqlite_path: Path, check_app) -> None:
    app, _ = check_app
    with TestClient(app) as client:
        body = client.post(
            "/api/v1/check/code",
            json={"code": BROKEN_PY, "language": "python"},
        ).json()

    assert body["syntax_ok"] is False
    assert body["local"]["syntax_error"]["tool"] == "ast"
    assert body["local"]["syntax_error"]["line"] == 1
    assert body["score"] == 45.0


def test_api_c_locates_error_with_tree_sitter(sqlite_path: Path, check_app) -> None:
    app, _ = check_app
    with TestClient(app) as client:
        body = client.post(
            "/api/v1/check/code",
            json={"code": "int main() { return 0 }\n", "language": "C"},
        ).json()

    assert body["language"] == "c"
    assert body["local"]["syntax_error"]["tool"] == "tree-sitter"
    assert body["local"]["syntax_error"]["line"] == 1


def test_api_reports_local_rule_issues(sqlite_path: Path, check_app) -> None:
    app, _ = check_app
    with TestClient(app) as client:
        body = client.post(
            "/api/v1/check/code", json={"code": RISKY_C, "language": "c"}
        ).json()

    local_risks = [item for item in body["risks"] if item["source"] == "local"]
    assert any("gets" in item["title"] for item in local_risks)
    assert all(item["suggestion"] for item in local_risks)


def test_api_metrics_present(sqlite_path: Path, check_app) -> None:
    app, _ = check_app
    with TestClient(app) as client:
        body = client.post(
            "/api/v1/check/code", json={"code": GOOD_PY, "language": "python"}
        ).json()

    metrics = body["local"]["metrics"]
    assert metrics["function_count"] == 2
    assert metrics["comment_lines"] > 0
    assert metrics["symbols"]


def test_api_infers_language_from_filename(sqlite_path: Path, check_app) -> None:
    app, _ = check_app
    with TestClient(app) as client:
        body = client.post(
            "/api/v1/check/code", json={"code": "int main(void) { return 0; }\n", "filename": "m.c"}
        ).json()
    assert body["language"] == "c"
    assert body["filename"] == "m.c"


def test_api_rejects_unsupported_language(sqlite_path: Path, check_app) -> None:
    app, _ = check_app
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/check/code", json={"code": "package main\n", "language": "go"}
        )
    assert response.status_code == 400
    assert "不支持的语言" in response.json()["detail"]


def test_api_rejects_empty_code(sqlite_path: Path, check_app) -> None:
    app, _ = check_app
    with TestClient(app) as client:
        response = client.post("/api/v1/check/code", json={"code": "", "language": "python"})
    assert response.status_code == 422   # Pydantic 的 min_length 拦在前面


def test_api_rejects_whitespace_only_code(sqlite_path: Path, check_app) -> None:
    app, _ = check_app
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/check/code", json={"code": "   \n\n", "language": "python"}
        )
    assert response.status_code == 400


def test_api_persists_record_and_is_queryable(sqlite_path: Path, check_app) -> None:
    app, _ = check_app
    with TestClient(app) as client:
        for code, language in ((RISKY_PY, "python"), (RISKY_C, "c"), (RISKY_JAVA, "java")):
            assert client.post(
                "/api/v1/check/code", json={"code": code, "language": language}
            ).status_code == 200

    async def scenario(session):
        rows, total = await CodeCheckService.list_records(session)
        return rows, total

    # 接口用的是自己的会话，这里重新开一个读同一份 SQLite 文件
    rows, total = _run_db(scenario)
    assert total == 3
    assert {row.language for row in rows} == {"python", "c", "java"}


def test_api_works_without_llm(sqlite_path: Path) -> None:
    """没配模型时接口仍然返回 200 + 本地结论（而不是 503）。"""
    app = create_app()
    settings = get_settings()
    app.dependency_overrides[get_code_checker] = lambda: CodeChecker(settings, OfflineLLM())
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/check/code", json={"code": RISKY_C, "language": "c"}
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["ai_available"] is False
    assert body["note"]
    assert any(item["source"] == "local" for item in body["risks"])


def test_api_survives_llm_failure(sqlite_path: Path) -> None:
    app = create_app()
    settings = get_settings()
    fake = CheckFakeLLM(fail_with=RuntimeError("上游超时"))
    app.dependency_overrides[get_code_checker] = lambda: CodeChecker(settings, fake)
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/check/code", json={"code": GOOD_PY, "language": "python"}
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["ai_available"] is False
    assert body["warnings"]
