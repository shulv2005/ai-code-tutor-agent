"""代码解析测试：AST（Python）与 Tree-sitter（多语言）两条链路。"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.services.repo.language import detect_language, supported_languages
from app.services.repo.parser import (
    decode_source,
    iter_source_files,
    parse_file,
    parse_source,
    to_posix_relative,
)
from tests.conftest import BROKEN_PY, SAMPLE_GO, SAMPLE_JS, SAMPLE_PY


def _by_name(parsed, name: str):
    return next(item for item in parsed.symbols if item.qualified_name == name)


# ---------------------------------------------------------------------------
# 语言识别
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("a.py", "python"),
        ("a.pyi", "python"),
        ("b.js", "javascript"),
        ("b.mjs", "javascript"),
        ("c.go", "go"),
        ("readme.md", None),
        ("Makefile", None),
    ],
)
def test_detect_language(filename: str, expected: str | None) -> None:
    assert detect_language(filename) == expected


def test_supported_languages_are_available_offline() -> None:
    """语法包必须自带编译产物，离线也应全部可用。"""
    languages = supported_languages()
    assert {"python", "javascript", "go"} <= set(languages)


# ---------------------------------------------------------------------------
# Python / ast
# ---------------------------------------------------------------------------
def test_python_extracts_functions_classes_and_docstrings() -> None:
    parsed = parse_source(SAMPLE_PY.encode(), "pkg/core.py")
    assert parsed.parse_error is None
    assert parsed.language == "python"
    assert parsed.size_bytes > 0

    add = _by_name(parsed, "add")
    assert add.kind == "function"
    assert add.docstring == "Add two numbers."
    assert add.signature == "def add(a: int, b: int) -> int"
    assert add.complexity == 1
    assert add.is_async is False

    service = _by_name(parsed, "Service")
    assert service.kind == "class"
    assert service.docstring == "A service."

    run = _by_name(parsed, "Service.run")
    assert run.kind == "method"
    assert run.start_line > service.start_line


def test_python_async_and_complexity() -> None:
    parsed = parse_source(SAMPLE_PY.encode(), "pkg/core.py")
    fetch = _by_name(parsed, "fetch")
    assert fetch.is_async is True
    assert fetch.signature.startswith("async def fetch")
    # if + for + while = 3 个判定点，圈复杂度 1 + 3 = 4
    assert fetch.complexity == 4


def test_python_line_range_covers_body() -> None:
    parsed = parse_source(SAMPLE_PY.encode(), "pkg/core.py")
    add = _by_name(parsed, "add")
    lines = SAMPLE_PY.splitlines()
    body = "\n".join(lines[add.start_line - 1 : add.end_line])
    assert "return a + b" in body


def test_python_imports() -> None:
    parsed = parse_source(SAMPLE_PY.encode(), "pkg/core.py")
    modules = {item.module for item in parsed.imports}
    assert "os" in modules
    assert "typing" in modules


def test_python_syntax_error_degrades_without_raising() -> None:
    """仓库里存在语法错误文件是常态，必须降级而不是抛异常。"""
    parsed = parse_source(BROKEN_PY.encode(), "pkg/broken.py")
    assert parsed.symbols == []
    assert parsed.parse_error is not None
    assert "SyntaxError" in parsed.parse_error


def test_python_nested_function_gets_qualified_name() -> None:
    source = (
        "class Outer:\n"
        "    def method(self):\n"
        "        def inner():\n"
        "            return 1\n"
        "        return inner\n"
    )
    parsed = parse_source(source.encode(), "n.py")
    assert _by_name(parsed, "Outer.method").kind == "method"
    assert _by_name(parsed, "Outer.method.inner").qualified_name == "Outer.method.inner"


def test_class_bases_in_signature() -> None:
    parsed = parse_source(b"class A(B, C):\n    pass\n", "n.py")
    assert _by_name(parsed, "A").signature == "class A(B, C)"


# ---------------------------------------------------------------------------
# JavaScript / Go —— tree-sitter
# ---------------------------------------------------------------------------
def test_javascript_symbols() -> None:
    parsed = parse_source(SAMPLE_JS.encode(), "web.js")
    assert parsed.parse_error is None

    greet = _by_name(parsed, "greet")
    assert greet.kind == "function"
    # 注释挂在 export_statement 外层，需要向上穿透才能取到
    assert greet.docstring == "// Greet someone politely."

    double = _by_name(parsed, "double")
    assert double.kind == "function"
    # 匿名箭头函数用变量名补全签名，否则只有 "(x) =>"
    assert double.signature == "double = (x) =>"

    assert _by_name(parsed, "Widget").kind == "class"
    assert _by_name(parsed, "render").kind == "method"


def test_go_symbols_and_imports() -> None:
    parsed = parse_source(SAMPLE_GO.encode(), "main.go")
    assert parsed.parse_error is None

    assert _by_name(parsed, "helper").kind == "function"
    assert _by_name(parsed, "Start").kind == "method"
    assert _by_name(parsed, "Server").kind == "struct"
    assert _by_name(parsed, "Server").docstring == "// Server serves things."

    # 只应有 import_spec，不能出现 "import (" 这种包装节点
    assert [item.module for item in parsed.imports] == ["fmt"]


def test_go_complexity() -> None:
    source = b"package m\n\nfunc f(x int) int {\n\tif x > 0 {\n\t\treturn 1\n\t}\n\treturn 0\n}\n"
    parsed = parse_source(source, "m.go")
    assert _by_name(parsed, "f").complexity == 2


def test_unknown_language_degrades() -> None:
    parsed = parse_source(b"hello", "notes.txt")
    assert parsed.language == "unknown"
    assert parsed.symbols == []
    assert parsed.parse_error is not None


def test_empty_source_is_handled() -> None:
    for path in ("a.py", "a.js", "a.go"):
        parsed = parse_source(b"", path)
        assert parsed.symbols == []
        assert parsed.total_lines == 0


# ---------------------------------------------------------------------------
# 文件遍历与编码
# ---------------------------------------------------------------------------
def test_iter_source_files_skips_excluded_dirs_and_non_source(sample_repo: Path) -> None:
    excluded = {".git", "node_modules", "__pycache__"}
    found = {to_posix_relative(item, sample_repo) for item in iter_source_files(
        sample_repo, excluded_dirs=excluded
    )}
    assert "pkg/core.py" in found
    assert "web.js" in found
    assert "main.go" in found
    # README.md 非源码；node_modules 被整树剪枝
    assert not any(item.endswith(".md") for item in found)
    assert not any("node_modules" in item for item in found)


def test_iter_source_files_respects_max_files(sample_repo: Path) -> None:
    found = list(iter_source_files(sample_repo, excluded_dirs={".git"}, max_files=1))
    assert len(found) == 1


def test_parse_file_respects_size_limit(sample_repo: Path) -> None:
    parsed = parse_file(sample_repo / "pkg" / "core.py", sample_repo, max_file_bytes=10)
    assert parsed.parse_error is not None
    assert "上限" in parsed.parse_error


def test_parse_file_reports_relative_posix_path(sample_repo: Path) -> None:
    parsed = parse_file(sample_repo / "pkg" / "core.py", sample_repo, max_file_bytes=10**6)
    assert parsed.path == "pkg/core.py"
    assert "\\" not in parsed.path


def test_decode_source_falls_back_for_non_utf8() -> None:
    """非 UTF-8 文件不能中断索引。"""
    assert decode_source("中文".encode()) == "中文"
    assert decode_source("中文".encode("gbk"))  # 不抛异常即可
    assert decode_source(b"\xff\xfe\x00bad") is not None
