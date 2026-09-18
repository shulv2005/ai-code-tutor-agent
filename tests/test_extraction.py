"""LLM 输出提取测试：这是 Agent 链路最常见的故障点。"""

from __future__ import annotations

from app.agents.extraction import (
    check_generated_test,
    extract_python_code,
    python_test_functions,
    syntax_error,
)

GOOD_TEST = "import pytest\n\n\ndef test_ok():\n    assert True\n"


# ---------------------------------------------------------------------------
# syntax_error
# ---------------------------------------------------------------------------
def test_syntax_error_none_for_valid_code() -> None:
    assert syntax_error(GOOD_TEST) is None


def test_syntax_error_reports_line() -> None:
    error = syntax_error("def broken(:\n    pass\n")
    assert error is not None
    assert "SyntaxError" in error


def test_syntax_error_for_empty() -> None:
    assert syntax_error("   ") == "代码为空"


# ---------------------------------------------------------------------------
# 围栏提取
# ---------------------------------------------------------------------------
def test_extracts_from_python_fence() -> None:
    text = f"这是测试代码：\n```python\n{GOOD_TEST}```\n希望有帮助！"
    code, warnings = extract_python_code(text)
    assert code.strip() == GOOD_TEST.strip()
    assert warnings == []


def test_extracts_from_bare_fence() -> None:
    text = f"```\n{GOOD_TEST}```"
    code, _ = extract_python_code(text)
    assert "def test_ok" in code


def test_extracts_from_py_fence() -> None:
    text = f"```py\n{GOOD_TEST}```"
    code, _ = extract_python_code(text)
    assert "def test_ok" in code


def test_prefers_block_containing_tests() -> None:
    """模型有时会先给一段非测试代码，再给测试代码。"""
    text = (
        "先看辅助代码：\n```python\nx = 1\n```\n"
        f"测试如下：\n```python\n{GOOD_TEST}```"
    )
    code, _ = extract_python_code(text)
    assert "def test_ok" in code


def test_merges_multiple_test_blocks() -> None:
    """模型把 import 与测试拆成两个块时必须合并，否则 import 会丢。"""
    text = (
        "```python\nimport pytest\n```\n"
        "```python\ndef test_a():\n    assert True\n```\n"
        "```python\ndef test_b():\n    assert True\n```\n"
    )
    code, warnings = extract_python_code(text)
    assert "import pytest" in code
    assert "def test_a" in code
    assert "def test_b" in code
    assert any("合并" in item for item in warnings)
    assert syntax_error(code) is None


# ---------------------------------------------------------------------------
# 无围栏 / 脏输出
# ---------------------------------------------------------------------------
def test_extracts_without_fence_and_warns() -> None:
    text = "好的，下面是测试：\nimport pytest\n\n\ndef test_x():\n    assert 1\n\n以上。"
    code, warnings = extract_python_code(text)
    assert "def test_x" in code
    assert syntax_error(code) is None
    assert any("未检测到" in item for item in warnings)


def test_strips_leading_and_trailing_prose() -> None:
    text = (
        "Here is the test file you asked for.\n"
        "It covers normal and edge cases.\n"
        "import pytest\n\n\n"
        "def test_alpha():\n"
        "    assert 1 == 1\n\n"
        "Let me know if you want more tests.\n"
    )
    code, _ = extract_python_code(text)
    assert syntax_error(code) is None
    assert python_test_functions(code) == ["test_alpha"]


def test_handles_fully_indented_output() -> None:
    """模型在列表里给出的代码常被整体缩进，必须能还原。"""
    text = "```python\n    import pytest\n\n    def test_indented():\n        assert 1\n```"
    code, _ = extract_python_code(text)
    assert syntax_error(code) is None, code
    assert python_test_functions(code) == ["test_indented"]


def test_empty_output_is_reported() -> None:
    code, warnings = extract_python_code("")
    assert code == ""
    assert warnings


def test_truncated_output_is_caught_by_validation() -> None:
    """被 max_tokens 截断时代码不完整，必须能被校验拦下（触发重试）。

    注意：提取器会把残缺尾部裁掉以得到可解析片段，因此不能指望
    `syntax_error` 报错——真正的防线是 check_generated_test 发现
    "没有任何 test_ 函数"。
    """
    text = "```python\nimport pytest\n\n\ndef test_partial():\n    assert add("
    code, warnings = extract_python_code(text)

    problems = check_generated_test(code)
    assert problems, "截断的输出必须被校验拦下"
    assert any("test_" in item for item in problems)
    assert any("截断" in item for item in warnings)


def test_merge_keeps_import_block_needed_by_tests() -> None:
    """回归：合并时不能丢掉 import 块，否则生成结果运行时会 NameError。"""
    text = (
        "```python\nimport pytest\nfrom pkg.core import add\n```\n"
        "```python\ndef test_add():\n    assert add(1, 2) == 3\n```\n"
    )
    code, _ = extract_python_code(text)
    assert "import pytest" in code
    assert "from pkg.core import add" in code
    assert check_generated_test(code) == []


def test_json_like_or_nonpython_output_does_not_crash() -> None:
    for text in ["{'code': 'x'}", "no code here at all", "```sql\nSELECT 1;\n```"]:
        code, _ = extract_python_code(text)
        assert isinstance(code, str)


# ---------------------------------------------------------------------------
# 测试函数识别
# ---------------------------------------------------------------------------
def test_python_test_functions_finds_functions_and_methods() -> None:
    code = (
        "def test_a():\n    pass\n\n"
        "async def test_b():\n    pass\n\n"
        "class TestGroup:\n"
        "    def test_c(self):\n"
        "        pass\n\n"
        "def helper():\n"
        "    pass\n"
    )
    assert python_test_functions(code) == ["test_a", "test_b", "test_c"]


def test_python_test_functions_falls_back_on_syntax_error() -> None:
    assert python_test_functions("def test_broken(:\n") == ["test_broken"]


# ---------------------------------------------------------------------------
# 生成结果校验
# ---------------------------------------------------------------------------
def test_check_passes_for_valid_test() -> None:
    assert check_generated_test(GOOD_TEST) == []


def test_check_rejects_syntax_error_first() -> None:
    problems = check_generated_test("def broken(:\n")
    assert len(problems) == 1
    assert "无法解析" in problems[0]


def test_check_rejects_missing_test_functions() -> None:
    problems = check_generated_test("import pytest\n\n\ndef helper():\n    return 1\n")
    assert any("test_" in item for item in problems)


def test_check_rejects_pytest_used_without_import() -> None:
    """用了 pytest.raises 却没 import pytest，运行必然 NameError。"""
    code = "def test_raises():\n    with pytest.raises(ValueError):\n        raise ValueError\n"
    problems = check_generated_test(code)
    assert any("import pytest" in item for item in problems)


def test_check_accepts_pytest_imported_variants() -> None:
    variants = [
        "import pytest\n\n\ndef test_a():\n    with pytest.raises(ValueError):\n        raise ValueError\n",
        "import pytest as pt\n\n\ndef test_a():\n    assert True\n",
    ]
    assert check_generated_test(variants[0]) == []
    assert check_generated_test(variants[1]) == []


def test_check_accepts_test_without_pytest_usage() -> None:
    assert check_generated_test("def test_plain():\n    assert 1 == 1\n") == []
