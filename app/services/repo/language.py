"""语言识别与语法 profile 注册表。

设计要点（均来自对 tree-sitter 0.26 的实测结论）：
1. 语法包用「自带编译产物」的独立 wheel（tree_sitter_python 等），**不用**
   tree-sitter-language-pack —— 后者运行时从 GitHub 下载语法库，离线/内网必失败。
2. 各语言的节点类型名并不通用：Python 根节点是 `module`，JS 是 `program`，
   Go 是 `source_file`；函数节点也分别是 `function_definition` /
   `function_declaration` / `method_declaration`。因此按语言建 profile，
   而不是写一份通用 query（实测跨语言 query 会抛 QueryError）。
3. 语法包按需懒加载：缺包时该语言降级为「只统计行数」，不影响其它语言与应用启动。
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LanguageProfile:
    """某语言的语法节点类型集合。"""

    name: str
    root_types: frozenset[str]
    function_types: frozenset[str] = frozenset()
    method_types: frozenset[str] = frozenset()
    class_types: frozenset[str] = frozenset()
    struct_types: frozenset[str] = frozenset()
    import_types: frozenset[str] = frozenset()
    # 参与圈复杂度计算的判定节点
    decision_types: frozenset[str] = frozenset()
    # 注释节点类型（用于抽取 docstring）
    comment_types: frozenset[str] = frozenset()
    # 声明式赋值箭头函数：const f = () => {}，需要从 variable_declarator 取名
    declarator_types: frozenset[str] = frozenset()


PROFILES: dict[str, LanguageProfile] = {
    "python": LanguageProfile(
        name="python",
        root_types=frozenset({"module"}),
        function_types=frozenset({"function_definition"}),
        class_types=frozenset({"class_definition"}),
        import_types=frozenset({"import_statement", "import_from_statement"}),
        comment_types=frozenset({"comment"}),
    ),
    "javascript": LanguageProfile(
        name="javascript",
        root_types=frozenset({"program"}),
        function_types=frozenset(
            {
                "function_declaration",
                "generator_function_declaration",
                "function_expression",
                "arrow_function",
            }
        ),
        method_types=frozenset({"method_definition"}),
        class_types=frozenset({"class_declaration", "class"}),
        import_types=frozenset({"import_statement"}),
        decision_types=frozenset(
            {
                "if_statement",
                "for_statement",
                "for_in_statement",
                "while_statement",
                "do_statement",
                "switch_case",
                "catch_clause",
                "ternary_expression",
            }
        ),
        comment_types=frozenset({"comment"}),
        declarator_types=frozenset({"variable_declarator", "lexical_declaration"}),
    ),
    "go": LanguageProfile(
        name="go",
        root_types=frozenset({"source_file"}),
        function_types=frozenset({"function_declaration"}),
        method_types=frozenset({"method_declaration"}),
        struct_types=frozenset({"type_spec"}),
        # 只收 import_spec：import_declaration 是外层包装（`import (` / `import`），
        # 一起收会产生 "import (" 这种无意义条目，且与 spec 重复
        import_types=frozenset({"import_spec"}),
        decision_types=frozenset(
            {
                "if_statement",
                "for_statement",
                "expression_case",
                "type_case",
                "communication_case",
            }
        ),
        comment_types=frozenset({"comment"}),
    ),
    "c": LanguageProfile(
        name="c",
        # 实测：C 的根节点是 translation_unit（既不是 module 也不是 program）
        root_types=frozenset({"translation_unit"}),
        function_types=frozenset({"function_definition"}),
        struct_types=frozenset({"struct_specifier", "type_definition"}),
        import_types=frozenset({"preproc_include"}),
        decision_types=frozenset(
            {
                "if_statement",
                "for_statement",
                "while_statement",
                "do_statement",
                "case_statement",
                "conditional_expression",
            }
        ),
        comment_types=frozenset({"comment"}),
    ),
    "java": LanguageProfile(
        name="java",
        # 实测：Java 的根节点是 program（与 JavaScript 同名）
        root_types=frozenset({"program"}),
        method_types=frozenset({"method_declaration", "constructor_declaration"}),
        class_types=frozenset({"class_declaration", "enum_declaration"}),
        struct_types=frozenset({"interface_declaration"}),
        import_types=frozenset({"import_declaration", "package_declaration"}),
        decision_types=frozenset(
            {
                "if_statement",
                "for_statement",
                "enhanced_for_statement",
                "while_statement",
                "do_statement",
                "switch_expression",
                "switch_label",
                "catch_clause",
                "ternary_expression",
            }
        ),
        comment_types=frozenset({"line_comment", "block_comment"}),
    ),
}

# 扩展名 -> 语言标识
SUFFIX_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".go": "go",
    ".c": "c",
    ".h": "c",
    ".java": "java",
}

# 语言标识 -> 面向学生展示的名称（前端分类结果里显示）
LANGUAGE_LABELS: dict[str, str] = {
    "python": "Python",
    "javascript": "JavaScript",
    "go": "Go",
    "c": "C",
    "java": "Java",
    "unknown": "未知语言",
}

# 语言标识 -> 语法包模块名（懒加载）
GRAMMAR_MODULES: dict[str, str] = {
    "python": "tree_sitter_python",
    "javascript": "tree_sitter_javascript",
    "go": "tree_sitter_go",
    "c": "tree_sitter_c",
    "java": "tree_sitter_java",
}


def detect_language(path: str | Path) -> str | None:
    """按扩展名识别语言；未收录返回 None。"""
    suffix = Path(path).suffix.lower()
    return SUFFIX_TO_LANGUAGE.get(suffix)


@cache
def load_grammar(language: str) -> Any | None:
    """加载并缓存 tree-sitter Language 对象；不可用时返回 None。

    返回 None 的两种情况：该语言无对应语法包，或语法包未安装。
    调用方据此降级为「只统计行数」，而不是让整条链路失败。
    """
    module_name = GRAMMAR_MODULES.get(language)
    if module_name is None:
        return None
    try:
        tree_sitter = importlib.import_module("tree_sitter")
        module = importlib.import_module(module_name)
    except ImportError:
        logger.warning("语法包不可用，%s 将降级解析: %s", language, module_name)
        return None
    try:
        # 新式语法包导出 language() -> PyCapsule，需包一层 Language
        return tree_sitter.Language(module.language())
    except Exception:  # noqa: BLE001 - 语法包版本差异兜底
        logger.warning("语法包初始化失败: %s", module_name, exc_info=True)
        return None


def get_profile(language: str) -> LanguageProfile | None:
    """获取语言 profile。"""
    return PROFILES.get(language)


def supported_languages() -> list[str]:
    """返回当前环境真正可解析的语言（语法包已安装且 profile 存在）。"""
    return sorted(name for name in PROFILES if load_grammar(name) is not None)


__all__ = [
    "GRAMMAR_MODULES",
    "PROFILES",
    "SUFFIX_TO_LANGUAGE",
    "LanguageProfile",
    "detect_language",
    "get_profile",
    "load_grammar",
    "supported_languages",
]
