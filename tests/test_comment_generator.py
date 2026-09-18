"""代码注释生成测试：Prompt 构造、三层注释要求、本地复检、覆盖统计、SQLite、API。

大模型用可注入的假客户端；本地复检（AST 比对、覆盖统计）是真跑的——
它才是"这份带注释的代码能不能放心用"的判据。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_comment_generator
from app.core.config import get_settings
from app.core.database import dispose_engine, get_session, init_db
from app.core.llm_client import LLMMessage, LLMResponse, LLMUsage
from app.main import create_app
from app.services.comment_generator import (
    COMMENT_SYSTEM,
    PROMPT_VERSION,
    STYLE_GUIDE,
    CommentGenerator,
    build_messages,
    compare_code,
    count_covered_functions,
    count_inline_comments,
    has_file_comment,
)
from app.services.comment_service import CommentService

# ---------------------------------------------------------------------------
# 样例代码
# ---------------------------------------------------------------------------
PLAIN_PY = '''def average(scores):
    total = 0
    for score in scores:
        total += score
    return total / len(scores)


def main():
    print(average([88, 92, 79]))


main()
'''

DOCUMENTED_PY = '''"""学生成绩统计：计算一组成绩的平均分。"""


def average(scores):
    """求平均分。

    Args:
        scores: 分数列表

    Returns:
        平均分
    """
    total = 0
    for score in scores:
        total += score
    return total / len(scores)


def main():
    """程序入口：算一组示例成绩的平均分。"""
    # 打印结果
    print(average([88, 92, 79]))


main()
'''

# 模型"顺手改了逻辑"的版本：只把 total += score 改成 total -= score
CHANGED_PY = DOCUMENTED_PY.replace("total += score", "total -= score")

# 模型"改坏了语法"的版本：三引号没闭合
BROKEN_PY = '''"""学生成绩统计。


def average(scores):
    """求平均分。
    return 0
'''

PLAIN_C = '''#include <stdio.h>

int add(int a, int b) {
    return a + b;
}

int main(void) {
    printf("%d\\n", add(1, 2));
    return 0;
}
'''

DOCUMENTED_C = '''/*
 * 简单的加法演示程序。
 */

#include <stdio.h>

/*
 * 两数相加。
 * 参数：a 第一个加数，b 第二个加数
 * 返回：两数之和
 */
int add(int a, int b) {
    // 直接返回和
    return a + b;
}

/*
 * 程序入口：打印 1 + 2 的结果。
 */
int main(void) {
    printf("%d\\n", add(1, 2));
    return 0;
}
'''

PLAIN_JAVA = '''public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
'''

DOCUMENTED_JAVA = '''/**
 * 一个极简的计算器。
 */
public class Calculator {

    /**
     * 两数相加。
     * @param a 第一个加数
     * @param b 第二个加数
     * @return 两数之和
     */
    public int add(int a, int b) {
        // 直接返回和
        return a + b;
    }
}
'''


def _reply(commented: str, summary: str = "加了文件说明与函数注释") -> str:
    """拼一个模型回复。"""
    return json.dumps(
        {
            "commented_code": commented,
            "summary": summary,
            "coverage": {"file_comment": True, "functions_documented": [], "inline_comments": 1},
        },
        ensure_ascii=False,
    )


class CommentFakeLLM:
    """可编排的假 LLM。"""

    def __init__(self, reply: str = "", *, fail_with: Exception | None = None) -> None:
        self.reply = reply or _reply(DOCUMENTED_PY)
        self.fail_with = fail_with
        self.calls: list[list[LLMMessage]] = []

    @property
    def model(self) -> str:
        return "fake-commenter"

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
            model="fake-commenter",
            finish_reason="stop",
            usage=LLMUsage(prompt_tokens=700, completion_tokens=500, total_tokens=1200),
            latency_ms=9.0,
        )

    async def aclose(self) -> None:
        return None


class OfflineLLM(CommentFakeLLM):
    """自称"未配置"的假客户端。"""

    @property
    def configured(self) -> bool:
        return False


def _generator(llm: CommentFakeLLM) -> CommentGenerator:
    return CommentGenerator(get_settings(), llm)


def _run_async(coro):
    import asyncio

    return asyncio.run(coro)


def _run_db(scenario):
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
def test_prompt_injects_function_list() -> None:
    """函数清单必须进提示词：告诉模型"该给谁写注释"，避免漏掉函数。"""
    structure = "需要写函数级注释的清单：\n    · average（function，第 1-5 行）"
    messages = build_messages(
        PLAIN_PY, language="python", filename="a.py", structure=structure
    )
    assert "average" in messages[0].content
    # 用户消息里也要再提醒一次（实测只放系统提示时模型会漏最后一个函数）
    assert "average" in messages[1].content


def test_prompt_requires_three_levels() -> None:
    system = COMMENT_SYSTEM
    assert "文件级注释" in system
    assert "函数/方法级注释" in system
    assert "关键逻辑行内注释" in system


def test_prompt_requires_only_comments() -> None:
    """最硬的约束：只加注释，不改代码。"""
    assert "只加注释，不要修改代码本身" in COMMENT_SYSTEM
    assert "不要" in COMMENT_SYSTEM and "顺手" in COMMENT_SYSTEM


def test_prompt_forbids_line_by_line_comments() -> None:
    assert "不要逐行加注释" in COMMENT_SYSTEM


def test_prompt_gives_good_and_bad_examples() -> None:
    """给好/坏对照比抽象要求有效。"""
    assert "判断分数是否及格" in COMMENT_SYSTEM      # 好例子
    assert "if 语句，判断 score" in COMMENT_SYSTEM    # 坏例子


def test_prompt_requires_keeping_existing_comments() -> None:
    assert "保留学生原有的注释" in COMMENT_SYSTEM


def test_prompt_forbids_fabricating_params() -> None:
    assert "不要编造" in COMMENT_SYSTEM


def test_style_guide_covers_all_languages() -> None:
    """三种语言都要有具体样例，否则模型给不出对应规范。"""
    assert set(STYLE_GUIDE) == {"python", "c", "java"}
    assert "Args:" in STYLE_GUIDE["python"]
    assert "@param" in STYLE_GUIDE["java"]
    assert "参数：" in STYLE_GUIDE["c"]


def test_prompt_uses_language_specific_style_guide() -> None:
    for language, marker in (("python", "Args:"), ("java", "@param"), ("c", "参数：")):
        messages = build_messages(
            "x", language=language, filename="f", structure="结构"
        )
        assert marker in messages[0].content, f"{language} 的规范没被注入"


def test_prompt_placeholders_are_replaced() -> None:
    messages = build_messages(PLAIN_PY, language="python", filename="a.py", structure="- 两个函数")
    assert "<<STRUCTURE>>" not in messages[0].content
    assert "<<STYLE_GUIDE>>" not in messages[0].content
    assert "- 两个函数" in messages[0].content


def test_prompt_version_defined() -> None:
    assert PROMPT_VERSION.startswith("comment-gen/")


def test_structure_description_lists_signatures() -> None:
    """结构说明里要带签名——参数名是模型写参数说明的唯一依据。"""
    generator = _generator(CommentFakeLLM())
    local = generator.analyze_locally(PLAIN_PY, language="python", filename="a.py")
    text = generator.describe_structure(local, language="python")

    assert "average" in text
    assert "签名" in text
    assert "scores" in text          # 参数名来自真实签名
    assert "main" in text


def test_structure_description_mentions_existing_comments() -> None:
    generator = _generator(CommentFakeLLM())
    local = generator.analyze_locally(DOCUMENTED_PY, language="python", filename="a.py")
    text = generator.describe_structure(local, language="python")
    assert "已经有" in text


def test_structure_description_handles_no_functions() -> None:
    generator = _generator(CommentFakeLLM())
    local = generator.analyze_locally("x = 1\ny = 2\n", language="python", filename="a.py")
    text = generator.describe_structure(local, language="python")
    assert "没有定义函数" in text


# ===========================================================================
# 2. 代码一致性比对（复检的核心）
# ===========================================================================
def test_compare_python_pure_comment_addition_is_unchanged() -> None:
    unchanged, note = compare_code(PLAIN_PY, DOCUMENTED_PY, "python")
    assert unchanged is True
    assert "一致" in note


def test_compare_python_detects_logic_change() -> None:
    """把 += 改成 -= 必须被查出来——这是本功能最危险的失败模式。"""
    unchanged, note = compare_code(DOCUMENTED_PY, CHANGED_PY, "python")
    assert unchanged is False
    assert "逻辑上有差异" in note


def test_compare_python_detects_renamed_variable() -> None:
    unchanged, _ = compare_code("x = 1\nprint(x)\n", "value = 1\nprint(value)\n", "python")
    assert unchanged is False


def test_compare_python_ignores_whitespace_only_changes() -> None:
    """纯缩进/空行风格变化不影响逻辑，不该被判成"改了代码"。"""
    unchanged, _ = compare_code("x = 1\ny = 2\n", "x = 1\n\n\ny = 2\n", "python")
    assert unchanged is True


def test_compare_python_ignores_added_docstrings() -> None:
    """加文档字符串正是本任务的目标，不能被算成改动。"""
    original = "def f():\n    return 1\n"
    documented = 'def f():\n    """什么都不做，只返回 1。"""\n    return 1\n'
    assert compare_code(original, documented, "python")[0] is True


def test_compare_python_returns_none_for_broken_original() -> None:
    """原代码本身有语法错误时无法比对，必须如实返回"无法判定"。"""
    unchanged, note = compare_code("def f(:\n", "def f(:\n", "python")
    assert unchanged is None
    assert "无法自动确认" in note


def test_compare_c_pure_comment_addition() -> None:
    unchanged, note = compare_code(PLAIN_C, DOCUMENTED_C, "c")
    assert unchanged is True
    assert "一致" in note


def test_compare_c_detects_code_change() -> None:
    changed = PLAIN_C.replace("return a + b;", "return a - b;")
    assert compare_code(PLAIN_C, changed, "c")[0] is False


def test_compare_c_ignores_reformatting() -> None:
    """C 里空白不影响语义，重新换行不该被判成改动。"""
    reformatted = "int add(int a,\n        int b) {\n    return a + b;\n}\n"
    assert compare_code("int add(int a, int b) { return a + b; }\n", reformatted, "c")[0] is True


def test_compare_c_keeps_strings_intact() -> None:
    """字符串里的 // 不是注释，去注释时不能把它删掉。"""
    code = 'printf("http://example.com");\n'
    assert compare_code(code, code, "c")[0] is True


def test_compare_java() -> None:
    assert compare_code(PLAIN_JAVA, DOCUMENTED_JAVA, "java")[0] is True
    assert compare_code(PLAIN_JAVA, PLAIN_JAVA.replace("a + b", "a - b"), "java")[0] is False


# ===========================================================================
# 3. 注释覆盖统计
# ===========================================================================
def test_count_covered_functions_python() -> None:
    """覆盖率按**函数名**在生成后的代码里重新定位。

    回归：早先用"原代码的行号"去生成后的代码里找函数，加了文件级注释与
    文档字符串后行号整体下移，结果明明每个函数都有注释却报 0/2。
    """
    functions = [{"name": "average"}, {"name": "main"}]
    covered, missing = count_covered_functions(DOCUMENTED_PY, functions, language="python")
    assert covered == 2
    assert missing == []


def test_count_covered_functions_reports_missing() -> None:
    functions = [{"name": "average"}, {"name": "main"}]
    # 只给 average 加了注释，main 没加
    partial = DOCUMENTED_PY.replace('    """程序入口：算一组示例成绩的平均分。"""\n', "")
    covered, missing = count_covered_functions(partial, functions, language="python")
    assert covered == 1
    assert missing == ["main"]


def test_count_covered_functions_ignores_body_comment_fake_docstring() -> None:
    """函数体第一行恰好是普通注释时，不能算成文档字符串。"""
    code = 'def f():\n    # 这只是一行注释，不是文档字符串\n    return 1\n'
    covered, missing = count_covered_functions(code, [{"name": "f"}], language="python")
    assert covered == 0 and missing == ["f"]


def test_count_covered_functions_c() -> None:
    functions = [{"name": "add"}, {"name": "main"}]
    covered, missing = count_covered_functions(
        DOCUMENTED_C, functions, language="c", filename="a.c"
    )
    assert covered == 2
    assert missing == []


def test_count_covered_functions_c_reports_missing() -> None:
    stripped = DOCUMENTED_C.replace("/*\n * 程序入口：打印 1 + 2 的结果。\n */\n", "")
    covered, missing = count_covered_functions(
        stripped, [{"name": "add"}, {"name": "main"}], language="c", filename="a.c"
    )
    assert covered == 1
    assert missing == ["main"]


def test_count_covered_functions_out_of_range_is_missing() -> None:
    covered, missing = count_covered_functions(
        "x = 1\n", [{"name": "f"}], language="python"
    )
    assert covered == 0 and missing == ["f"]


def test_has_file_comment() -> None:
    assert has_file_comment(DOCUMENTED_PY, "python") is True
    assert has_file_comment(PLAIN_PY, "python") is False
    assert has_file_comment(DOCUMENTED_C, "c") is True
    assert has_file_comment(PLAIN_C, "c") is False
    assert has_file_comment(DOCUMENTED_JAVA, "java") is True


def test_count_inline_comments_counts_trailing_style() -> None:
    """行尾注释（`x = 1  # 说明`）算关键逻辑注释。"""
    code = 'def f(x):\n    y = x + 1  # 加一\n    return y  # 返回\n'
    assert count_inline_comments(code, "python", filename="a.py") == 2


def test_count_inline_comments_counts_whole_line_style() -> None:
    """整行注释（写在代码上方）同样算关键逻辑注释。

    回归：最初只统计行尾注释，实测模型把关键逻辑注释写成独立一行时被算成 0 条，
    明明加了注释却报"没有行内注释"。
    """
    code = (
        'def f(x):\n'
        '    """说明。"""\n'
        "    # 第一步：加一\n"
        "    y = x + 1\n"
        "    # 第二步：返回\n"
        "    return y\n"
    )
    assert count_inline_comments(code, "python", filename="a.py") == 2


def test_count_inline_comments_excludes_file_and_function_level() -> None:
    """只有文件级与函数级注释时，关键逻辑注释应为 0。"""
    code = (
        '"""文件说明。"""\n'
        "\n"
        "\n"
        "def f(x):\n"
        '    """函数说明。"""\n'
        "    return x\n"
    )
    assert count_inline_comments(code, "python", filename="a.py") == 0


def test_count_inline_comments_c_excludes_blocks() -> None:
    code = (
        "/*\n * 文件说明\n */\n"
        "\n"
        "/*\n * 两数相加\n */\n"
        "int add(int a, int b) {\n"
        "    // 直接返回和\n"
        "    return a + b;\n"
        "}\n"
    )
    assert count_inline_comments(code, "c", filename="a.c") == 1


# ===========================================================================
# 4. 复检汇总口径
# ===========================================================================
def test_verify_passes_for_good_generation() -> None:
    generator = _generator(CommentFakeLLM())
    local = generator.analyze_locally(PLAIN_PY, language="python", filename="a.py")
    check = generator.verify(PLAIN_PY, DOCUMENTED_PY, language="python", filename="a.py", local=local)

    assert check.syntax_ok is True
    assert check.code_unchanged is True
    assert check.functions_covered == 2
    assert check.coverage_ratio == 1.0
    assert check.file_comment is True
    assert check.added_comment_lines > 0
    assert check.verified is True
    assert "原有代码未被改动" in check.note


def test_verify_fails_when_code_changed() -> None:
    """改了逻辑必须 verified=False，并且说明里点名。"""
    generator = _generator(CommentFakeLLM())
    local = generator.analyze_locally(DOCUMENTED_PY, language="python", filename="a.py")
    check = generator.verify(
        DOCUMENTED_PY, CHANGED_PY, language="python", filename="a.py", local=local
    )

    assert check.code_unchanged is False
    assert check.verified is False
    assert "改动" in check.note


def test_verify_fails_when_syntax_broken() -> None:
    generator = _generator(CommentFakeLLM())
    local = generator.analyze_locally(PLAIN_PY, language="python", filename="a.py")
    check = generator.verify(PLAIN_PY, BROKEN_PY, language="python", filename="a.py", local=local)

    assert check.syntax_ok is False
    assert check.verified is False
    assert "语法错误" in check.note


def test_verify_fails_when_no_comment_added() -> None:
    """原样返回（一行注释都没加）不该算通过。"""
    generator = _generator(CommentFakeLLM())
    local = generator.analyze_locally(PLAIN_PY, language="python", filename="a.py")
    check = generator.verify(PLAIN_PY, PLAIN_PY, language="python", filename="a.py", local=local)

    assert check.added_comment_lines == 0
    assert check.verified is False
    assert "没有新增注释行" in check.note


def test_verify_reports_missing_function_comment() -> None:
    generator = _generator(CommentFakeLLM())
    local = generator.analyze_locally(PLAIN_PY, language="python", filename="a.py")
    partial = DOCUMENTED_PY.replace('    """程序入口：算一组示例成绩的平均分。"""\n', "")
    check = generator.verify(PLAIN_PY, partial, language="python", filename="a.py", local=local)

    assert check.functions_covered == 1
    assert check.coverage_ratio == 0.5
    assert check.verified is True          # 有注释、代码没改，仍算可用
    assert "没写注释" in check.note         # 但要点出漏了哪个


# ===========================================================================
# 5. 完整流程
# ===========================================================================
async def test_generate_returns_commented_code() -> None:
    outcome = await _generator(CommentFakeLLM()).generate(
        PLAIN_PY, language="python", filename="avg.py"
    )

    assert outcome.ai_available is True
    assert outcome.model == "fake-commenter"
    assert outcome.commented_code == DOCUMENTED_PY
    assert outcome.original_code == PLAIN_PY
    assert outcome.verification.verified is True
    assert outcome.functions_total == 2
    assert outcome.summary


async def test_generate_c_and_java() -> None:
    c = await _generator(CommentFakeLLM(_reply(DOCUMENTED_C))).generate(
        PLAIN_C, language="c", filename="a.c"
    )
    assert c.verification.verified is True
    assert c.verification.code_unchanged is True

    java = await _generator(CommentFakeLLM(_reply(DOCUMENTED_JAVA))).generate(
        PLAIN_JAVA, language="java", filename="A.java"
    )
    assert java.verification.verified is True
    assert java.verification.file_comment is True


async def test_generate_flags_code_change_loudly() -> None:
    """模型顺手改了代码时：verified=False + warnings + note 三处都要说清。"""
    outcome = await _generator(CommentFakeLLM(_reply(CHANGED_PY))).generate(
        DOCUMENTED_PY, language="python", filename="avg.py"
    )

    assert outcome.verification.code_unchanged is False
    assert outcome.verification.verified is False
    assert any("逻辑上有差异" in item for item in outcome.warnings)
    assert "改动了原有逻辑" in outcome.note or "只添加注释" in outcome.note


async def test_generate_rejects_truncated_output() -> None:
    """模型用"…（其余不变）"省略内容时必须拦下，否则等于毁掉学生的文件。"""
    truncated = '"""说明。"""\n\n\ndef average(scores):\n    ...\n'
    outcome = await _generator(CommentFakeLLM(_reply(truncated))).generate(
        PLAIN_PY, language="python", filename="avg.py"
    )

    assert outcome.commented_code == PLAIN_PY         # 回退为原代码
    assert any("疑似省略" in item for item in outcome.warnings)


async def test_generate_falls_back_when_model_returns_nothing() -> None:
    reply = json.dumps({"commented_code": "", "summary": "空"}, ensure_ascii=False)
    outcome = await _generator(CommentFakeLLM(reply)).generate(
        PLAIN_PY, language="python", filename="avg.py"
    )

    assert outcome.commented_code == PLAIN_PY
    assert any("没有返回带注释的代码" in item for item in outcome.warnings)


async def test_generate_without_ai_returns_original() -> None:
    """没配模型时返回原代码 + 结构信息 + 明确提示（不能返回空串）。"""
    outcome = await _generator(OfflineLLM()).generate(
        PLAIN_PY, language="python", filename="avg.py"
    )

    assert outcome.ai_available is False
    assert outcome.commented_code == PLAIN_PY
    # 提示语统一指向"网页上填 Key"
    assert "请先在网页上输入 API Key" in outcome.note
    assert outcome.functions_total == 2
    assert outcome.verification.verified is False      # 没加注释，自然不算通过


async def test_generate_survives_ai_failure() -> None:
    outcome = await _generator(CommentFakeLLM(fail_with=RuntimeError("网关 502"))).generate(
        PLAIN_PY, language="python", filename="avg.py"
    )

    assert outcome.ai_available is False
    assert outcome.commented_code == PLAIN_PY
    assert any("502" in item for item in outcome.warnings)


async def test_generate_survives_non_json_reply() -> None:
    outcome = await _generator(CommentFakeLLM("这是注释版代码……")).generate(
        PLAIN_PY, language="python", filename="avg.py"
    )

    assert outcome.ai_available is False
    assert any("JSON" in item for item in outcome.warnings)


async def test_generate_rejects_empty_and_oversized() -> None:
    from app.services.code_checker import MAX_CODE_CHARS
    from app.services.comment_generator import CommentGeneratorError

    generator = _generator(CommentFakeLLM())
    with pytest.raises(CommentGeneratorError):
        await generator.generate("   \n", language="python", filename="a.py")
    with pytest.raises(CommentGeneratorError):
        await generator.generate("x = 1\n" * (MAX_CODE_CHARS // 6 + 10), language="python", filename="a.py")


async def test_generate_text_infers_language() -> None:
    generator = _generator(CommentFakeLLM(_reply(DOCUMENTED_C)))
    outcome = await generator.generate_text(PLAIN_C, language=None, filename="a.c")
    assert outcome.language == "c"
    assert outcome.filename == "a.c"

    default = await generator.generate_text("int main(void) { return 0; }\n", language="c", filename=None)
    assert default.filename == "snippet.c"


async def test_generate_prompt_carries_real_structure() -> None:
    llm = CommentFakeLLM()
    await _generator(llm).generate(PLAIN_PY, language="python", filename="avg.py")

    assert "average" in llm.system_prompt
    assert "Args:" in llm.system_prompt          # Python 规范样例
    assert "```python" in llm.user_prompt
    assert "只加注释" in llm.user_prompt


# ===========================================================================
# 6. SQLite 落库与历史
# ===========================================================================
def test_save_stores_both_versions(sqlite_path: Path) -> None:
    outcome = _run_async(
        _generator(CommentFakeLLM()).generate(PLAIN_PY, language="python", filename="avg.py")
    )

    async def scenario(session):
        record = await CommentService.save(session, outcome, trace_id="t-comment")
        rows, total = await CommentService.list_records(session)
        return record, rows, total

    record, rows, total = _run_db(scenario)

    assert record.id is not None
    assert total == 1
    saved = rows[0]
    assert saved.original_code == PLAIN_PY
    assert saved.commented_code == DOCUMENTED_PY
    assert saved.functions_total == 2
    assert saved.functions_covered == 2
    assert saved.coverage_ratio == 1.0
    assert saved.verified is True
    assert saved.code_unchanged is True
    assert saved.added_comment_lines > 0
    assert saved.trace_id == "t-comment"
    assert saved.created_at is not None


def test_save_records_unknown_unchanged_as_null(sqlite_path: Path) -> None:
    """无法判定与"确认没改"必须区分开存。"""
    broken = "def f(:\n"
    outcome = _run_async(
        _generator(CommentFakeLLM(_reply('"""说明。"""\n\n\ndef f(:\n'))).generate(
            broken, language="python", filename="a.py"
        )
    )

    async def scenario(session):
        return await CommentService.save(session, outcome)

    record = _run_db(scenario)
    assert record.code_unchanged is None
    assert record.verified is False


def test_list_records_filters(sqlite_path: Path) -> None:
    py = _run_async(
        _generator(CommentFakeLLM()).generate(PLAIN_PY, language="python", filename="avg.py")
    )
    c = _run_async(
        _generator(CommentFakeLLM(_reply(DOCUMENTED_C))).generate(PLAIN_C, language="c", filename="a.c")
    )

    async def scenario(session):
        await CommentService.save(session, py)
        await CommentService.save(session, c)
        only_c, total_c = await CommentService.list_records(session, language="c")
        by_name, total_name = await CommentService.list_records(session, filename="avg.py")
        page, total_all = await CommentService.list_records(session, limit=1)
        return only_c, total_c, by_name, total_name, page, total_all

    only_c, total_c, by_name, total_name, page, total_all = _run_db(scenario)
    assert total_all == 2
    assert total_c == 1 and only_c[0].filename == "a.c"
    assert total_name == 1 and by_name[0].language == "python"
    assert len(page) == 1


def test_get_record(sqlite_path: Path) -> None:
    outcome = _run_async(
        _generator(CommentFakeLLM()).generate(PLAIN_PY, language="python", filename="avg.py")
    )

    async def scenario(session):
        saved = await CommentService.save(session, outcome)
        return await CommentService.get_record(session, saved.id)

    fetched = _run_db(scenario)
    assert fetched is not None and fetched.commented_code == DOCUMENTED_PY


# ===========================================================================
# 7. API
# ===========================================================================
@pytest.fixture()
def comment_app() -> Iterator[object]:
    """注入假 LLM 的应用（假客户端在 lambda 外创建一次并共用）。"""
    app = create_app()
    settings = get_settings()
    fake = CommentFakeLLM()
    app.dependency_overrides[get_comment_generator] = lambda: CommentGenerator(settings, fake)
    yield app, fake
    app.dependency_overrides.clear()


def test_api_generates_commented_code(sqlite_path: Path, comment_app) -> None:
    app, fake = comment_app
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/comment/generate",
            json={"code": PLAIN_PY, "language": "python", "filename": "avg.py"},
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["language"] == "python"
    assert body["language_label"] == "Python"
    assert body["original_code"] == PLAIN_PY
    assert body["commented_code"] == DOCUMENTED_PY
    assert body["summary"]

    verification = body["verification"]
    assert verification["syntax_ok"] is True
    assert verification["code_unchanged"] is True
    assert verification["functions_total"] == 2
    assert verification["functions_covered"] == 2
    assert verification["coverage_ratio"] == 1.0
    assert verification["file_comment"] is True
    assert verification["added_comment_lines"] > 0
    assert verification["verified"] is True
    assert verification["note"]

    assert body["record_id"] is not None
    assert body["trace_id"]
    assert fake.calls


def test_api_returns_original_on_code_change(sqlite_path: Path, comment_app) -> None:
    """模型改了代码时：接口如实标注，warnings 里有明确提示。"""
    app, _ = comment_app
    app.dependency_overrides[get_comment_generator] = lambda: CommentGenerator(
        get_settings(), CommentFakeLLM(_reply(CHANGED_PY))
    )
    try:
        with TestClient(app) as client:
            body = client.post(
                "/api/v1/comment/generate", json={"code": DOCUMENTED_PY, "language": "python"}
            ).json()
    finally:
        app.dependency_overrides.clear()

    assert body["verification"]["code_unchanged"] is False
    assert body["verification"]["verified"] is False
    assert body["warnings"]


def test_api_c_and_java_style(sqlite_path: Path, comment_app) -> None:
    app, _ = comment_app
    for language, code, commented in (
        ("c", PLAIN_C, DOCUMENTED_C),
        ("Java", PLAIN_JAVA, DOCUMENTED_JAVA),
    ):
        app.dependency_overrides[get_comment_generator] = lambda c=commented: CommentGenerator(
            get_settings(), CommentFakeLLM(_reply(c))
        )
        with TestClient(app) as client:
            body = client.post(
                "/api/v1/comment/generate", json={"code": code, "language": language}
            ).json()
        assert body["verification"]["verified"] is True, body
        assert body["verification"]["file_comment"] is True
    app.dependency_overrides.clear()


def test_api_infers_language(sqlite_path: Path, comment_app) -> None:
    app, _ = comment_app
    with TestClient(app) as client:
        body = client.post(
            "/api/v1/comment/generate", json={"code": PLAIN_C, "filename": "m.c"}
        ).json()
    assert body["language"] == "c"
    assert body["filename"] == "m.c"


def test_api_rejects_unsupported_language(sqlite_path: Path, comment_app) -> None:
    app, _ = comment_app
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/comment/generate", json={"code": "package main\n", "language": "go"}
        )
    assert response.status_code == 400
    assert "不支持的语言" in response.json()["detail"]


def test_api_rejects_whitespace_only(sqlite_path: Path, comment_app) -> None:
    app, _ = comment_app
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/comment/generate", json={"code": "  \n\n", "language": "python"}
        )
    assert response.status_code == 400


def test_api_works_without_llm(sqlite_path: Path) -> None:
    """没配模型时返回 200 + 原代码（不能返回空串，也不能 503）。"""
    app = create_app()
    settings = get_settings()
    app.dependency_overrides[get_comment_generator] = lambda: CommentGenerator(settings, OfflineLLM())
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/comment/generate", json={"code": PLAIN_PY, "language": "python"}
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["ai_available"] is False
    assert body["commented_code"] == PLAIN_PY
    assert body["note"]


def test_history_flow(sqlite_path: Path, comment_app) -> None:
    app, _ = comment_app
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/comment/generate",
            json={"code": PLAIN_PY, "language": "python", "filename": "avg.py"},
        ).json()
        listed = client.get("/api/v1/comment/history").json()
        detail = client.get(f"/api/v1/comment/history/{created['record_id']}").json()

    assert listed["total"] == 1
    item = listed["items"][0]
    assert item["filename"] == "avg.py"
    assert item["added_comment_lines"] > 0
    assert item["coverage_ratio"] == 1.0
    # 列表不带代码全文（太重），详情才带
    assert "original_code" not in item

    assert detail["original_code"] == PLAIN_PY
    assert detail["commented_code"] == DOCUMENTED_PY
    assert detail["verification_note"]


def test_history_detail_404(sqlite_path: Path, comment_app) -> None:
    app, _ = comment_app
    with TestClient(app) as client:
        assert client.get("/api/v1/comment/history/9999").status_code == 404


def test_history_filters(sqlite_path: Path, comment_app) -> None:
    app, _ = comment_app
    with TestClient(app) as client:
        client.post(
            "/api/v1/comment/generate",
            json={"code": PLAIN_PY, "language": "python", "filename": "a.py"},
        )
        client.post(
            "/api/v1/comment/generate", json={"code": PLAIN_C, "language": "c", "filename": "b.c"}
        )
        only_c = client.get("/api/v1/comment/history", params={"language": "c"}).json()
        by_name = client.get("/api/v1/comment/history", params={"filename": "a.py"}).json()

    assert only_c["total"] == 1 and only_c["items"][0]["filename"] == "b.c"
    assert by_name["total"] == 1 and by_name["items"][0]["language"] == "python"
