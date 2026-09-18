"""多语言源码解析：基于 tree-sitter。

与 python_parser 的差异：这里没有语义层，全部信息从语法树节点类型推断，
因此每种语言的节点类型集合集中在 language.PROFILES 里维护。

实测要点（tree-sitter 0.26 + 语法 wheel）：
- `Parser(lang)` 直接构造即可，老写法 `parser.set_language()` 已不需要。
- 语法包的 `language()` 返回 PyCapsule，必须用 `tree_sitter.Language(...)` 包装。
- 跨语言共用一份 query 会抛 QueryError，所以这里走「按 profile 遍历节点」而非 query。
"""

from __future__ import annotations

import logging
from typing import Any

from app.services.repo.dto import ParsedFile, ParsedImport, ParsedSymbol
from app.services.repo.language import LanguageProfile, get_profile, load_grammar

logger = logging.getLogger(__name__)

# 匿名函数容器：const f = () => {} / {handler: function () {}}
_ANON_PARENTS: frozenset[str] = frozenset(
    {"variable_declarator", "pair", "assignment_expression", "public_field_definition"}
)
# Go 的 type_spec 可能声明 struct / interface / 普通别名，需要看子节点区分
_GO_TYPE_KINDS: dict[str, str] = {"struct_type": "struct", "interface_type": "interface"}


class GrammarUnavailableError(RuntimeError):
    """语法包缺失，调用方应降级为仅统计行数。"""


def _text(node: Any, source: bytes) -> str:
    """取节点对应源码文本。"""
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _normalize_signature(raw: str) -> str:
    """把多行签名压成单行，并去掉尾部块起始符。"""
    collapsed = " ".join(raw.split())
    return collapsed.rstrip("{:").strip()


def _dig_declarator(node: Any, source: bytes, depth: int = 0) -> str | None:
    """沿 declarator 链向内查找标识符，用于 C/C++ 的函数名提取。

    实测：C 的 function_definition **没有 name 字段**，函数名嵌在
    declarator 链里，形如：
        function_definition
          └─ declarator: function_declarator
               └─ declarator: identifier  <- 真正的函数名
    指针返回值还会多一层 pointer_declarator：
        declarator: pointer_declarator → declarator: function_declarator → identifier
    因此需要递归向内找，而不是只看一层。
    """
    if node is None or depth > 6:
        return None
    if node.type in ("identifier", "field_identifier", "type_identifier"):
        return _text(node, source).strip()
    inner = node.child_by_field_name("declarator")
    if inner is not None:
        found = _dig_declarator(inner, source, depth + 1)
        if found:
            return found
    # pointer_declarator 等包装节点没有 declarator 字段时，退化为看第一个具名子节点
    for child in node.children:
        if child.is_named:
            found = _dig_declarator(child, source, depth + 1)
            if found:
                return found
    return None


def _node_name(node: Any, source: bytes) -> str | None:
    """解析符号名。

    三种情形依次尝试：
    1. 具名节点直接取 name 字段（Python / Java / Go / JS 类与函数）；
    2. C/C++ 的函数名嵌在 declarator 链里，需要递归向内查找；
    3. 匿名函数（箭头函数 / 函数表达式）本身无 name，
       回退到父节点的变量名：`const fetchData = async () => {}`。
    """
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return _text(name_node, source).strip()

    declarator = node.child_by_field_name("declarator")
    if declarator is not None:
        found = _dig_declarator(declarator, source)
        if found:
            return found

    parent = node.parent
    if parent is not None and parent.type in _ANON_PARENTS:
        for field_name in ("name", "left", "key"):
            candidate = parent.child_by_field_name(field_name)
            if candidate is not None:
                return _text(candidate, source).strip()
    return None


def _is_async(node: Any, source: bytes, header: str) -> bool:
    """判断是否异步函数：头部以 async 开头，或存在 async 子标记。"""
    if header.lstrip().startswith("async"):
        return True
    return any(child.type == "async" for child in node.children)


# 导出/装饰包装节点：注释挂在包装节点上而非内部声明上，查注释时需向上穿透。
# 例：`// doc` 换行后接 `export function greet()`，greet 的前一个兄弟是 None，
# 注释其实是 export_statement 的前一个兄弟。
_WRAPPER_TYPES: frozenset[str] = frozenset(
    {
        "export_statement",
        "export_declaration",
        "decorated_definition",
        "ambient_declaration",
        # Go 的 type_spec 被包在 type_declaration 里，注释挂在外层
        "type_declaration",
    }
)


def _leading_comment(node: Any, source: bytes, profile: LanguageProfile) -> str | None:
    """收集紧邻在上方的注释作为 docstring 近似值，提升检索与提示词质量。"""
    if not profile.comment_types:
        return None

    anchor = node
    # 向上穿透导出/装饰包装，直到找到有前兄弟的层级
    while (
        anchor.prev_named_sibling is None
        and anchor.parent is not None
        and anchor.parent.type in _WRAPPER_TYPES
    ):
        anchor = anchor.parent

    parts: list[str] = []
    sibling = anchor.prev_named_sibling
    # 最多回溯 5 行注释，避免把文件头 license 误当作文档
    while sibling is not None and sibling.type in profile.comment_types and len(parts) < 5:
        parts.append(_text(sibling, source).strip())
        sibling = sibling.prev_named_sibling
    if not parts:
        return None
    parts.reverse()
    return "\n".join(parts)


def _complexity(node: Any, source: bytes, profile: LanguageProfile) -> int:
    """圈复杂度近似值：1 + 判定节点数量（不深入嵌套函数）。"""
    score = 1
    stack = list(node.children)
    while stack:
        current = stack.pop()
        # 嵌套函数/方法各自统计
        if current.type in profile.function_types or current.type in profile.method_types:
            continue
        if current.type in profile.decision_types:
            score += 1
        elif current.type == "binary_expression":
            operator = current.child_by_field_name("operator")
            if operator is not None and _text(operator, source) in ("&&", "||", "and", "or"):
                score += 1
        stack.extend(current.children)
    return score


# 类/结构体容器节点：函数若位于其中，应归类为方法
_TYPE_SCOPE_TYPES: frozenset[str] = frozenset(
    {
        "class_body",
        "class_declaration",
        "class",
        "class_definition",
        "field_declaration_list",
        "declaration_list",
    }
)


def _enclosing_type(node: Any) -> str | None:
    """向上查找最近的外层类型声明容器，用于区分方法与普通函数。"""
    parent = node.parent
    while parent is not None:
        if parent.type in _TYPE_SCOPE_TYPES:
            return parent.type
        parent = parent.parent
    return None


def _signature(node: Any, source: bytes, header: str, *, synthetic_name: str | None = None) -> str:
    """签名 = 节点起始到函数体起始之间的文本。

    匿名函数（`const f = () => {}`）自身签名只有 `(x) =>`，脱离上下文不可读，
    因此用变量名补成 `f = (x) =>`，便于 Step 4 生成测试时理解符号含义。
    """
    body = node.child_by_field_name("body")
    if body is not None and body.start_byte > node.start_byte:
        signature = _normalize_signature(
            source[node.start_byte : body.start_byte].decode("utf-8", errors="replace")
        )
    else:
        signature = _normalize_signature(header.split("\n", 1)[0])
    return f"{synthetic_name} = {signature}" if synthetic_name else signature


def _clean_import(module: str) -> str:
    """把导入语句整理成纯模块名，便于前端展示。

    各语言的原始写法差异很大，这里统一成「模块名」：
      #include <stdio.h>        -> stdio.h
      import java.util.List;    -> java.util.List
      package com.example;      -> com.example
    """
    text = module.strip().rstrip(";").strip()
    for prefix in ("#include", "import", "package", "using", "from"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
            break
    return text.strip().strip("\"'<>").strip()


def _collect_imports(root: Any, source: bytes, profile: LanguageProfile) -> list[ParsedImport]:
    """抽取导入语句（非 Python 语言为尽力而为）。"""
    imports: list[ParsedImport] = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type in profile.import_types:
            raw = _text(node, source).strip()
            module = raw.split("\n", 1)[0]
            # JS: import ... from 'mod' -> 取 source 字段
            source_node = node.child_by_field_name("source")
            if source_node is not None:
                module = _text(source_node, source).strip().strip("'\"")
            # Go: import "fmt" -> 取 path 字段
            path_node = node.child_by_field_name("path")
            if path_node is not None:
                module = _text(path_node, source).strip().strip('"')
            imports.append(
                ParsedImport(
                    module=_clean_import(module),
                    names=[],
                    line=node.start_point.row + 1,
                )
            )
        stack.extend(node.children)
    return imports


def parse_with_tree_sitter(source: bytes, path: str, language: str) -> ParsedFile:
    """用 tree-sitter 解析源码。

    Raises:
        GrammarUnavailableError: 语法包缺失或该语言未注册 profile。
    """
    profile = get_profile(language)
    grammar = load_grammar(language)
    if profile is None or grammar is None:
        raise GrammarUnavailableError(f"语言不可解析: {language}")

    import tree_sitter  # 局部导入：缺包时仅该路径失败

    parser = tree_sitter.Parser(grammar)
    tree = parser.parse(source)
    root = tree.root_node
    total_lines = source.count(b"\n") + 1 if source else 0

    symbols: list[ParsedSymbol] = []
    stack: list[Any] = [root]
    while stack:
        node = stack.pop()
        node_type = node.type

        kind: str | None = None
        if node_type in profile.method_types:
            kind = "method"
        elif node_type in profile.class_types and node is not root:
            kind = "class"
        elif node_type in profile.struct_types and node is not root:
            # Go: type_spec 只有在声明 struct/interface 时才算符号
            declared = node.child_by_field_name("type")
            kind = _GO_TYPE_KINDS.get(declared.type) if declared is not None else None
        elif node_type in profile.function_types:
            # 类体内的函数算方法
            kind = "method" if _enclosing_type(node) is not None else "function"

        if kind is not None and node is not root:
            name = _node_name(node, source)
            if name:
                header = _text(node, source)
                # 名字来自父节点 => 匿名函数，签名需要带上变量名
                synthetic = name if node.child_by_field_name("name") is None else None
                symbols.append(
                    ParsedSymbol(
                        name=name,
                        kind=kind,  # type: ignore[arg-type]
                        start_line=node.start_point.row + 1,
                        end_line=node.end_point.row + 1,
                        qualified_name=name,
                        signature=_signature(node, source, header, synthetic_name=synthetic),
                        docstring=_leading_comment(node, source, profile),
                        is_async=_is_async(node, source, header),
                        complexity=_complexity(node, source, profile),
                    )
                )

        stack.extend(node.children)

    # tree-sitter 容错解析：有错误也返回已识别的符号，只是标注降级
    parse_error = None
    if root.has_error:
        parse_error = "SyntaxError: 语法树包含错误节点，符号可能不完整"

    return ParsedFile(
        path=path,
        language=language,
        total_lines=total_lines,
        symbols=symbols,
        imports=_collect_imports(root, source, profile),
        parse_error=parse_error,
    )


__all__ = ["GrammarUnavailableError", "parse_with_tree_sitter"]
