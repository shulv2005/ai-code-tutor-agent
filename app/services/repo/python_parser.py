"""Python 源码解析：基于内置 `ast` 模块。

为什么 Python 单独走 AST 而不是 tree-sitter：
- `ast` 是标准库，零依赖、零额外解析开销，且语义信息（docstring、参数注解、
  async 标记、装饰器）是现成的结构化字段，不需要从语法树里猜。
- Tree-sitter 留给非 Python 语言（见 tree_sitter_parser.py）。

产出经 `ParsedFile` 归一化，两种解析器对上层完全同构。
"""

from __future__ import annotations

import ast
import logging

from app.services.repo.dto import ParsedFile, ParsedImport, ParsedSymbol

logger = logging.getLogger(__name__)

# 参与圈复杂度计算的判定节点
_DECISION_NODES: tuple[type[ast.AST], ...] = (
    ast.If,
    ast.IfExp,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.ExceptHandler,
    ast.Assert,
    ast.With,
    ast.AsyncWith,
    ast.comprehension,
)

# 这些节点开启新的作用域，复杂度统计到此为止（其内部单独统计）
_SCOPE_NODES: tuple[type[ast.AST], ...] = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
)


def _iter_scope_nodes(root: ast.AST):
    """遍历 root 作用域内的节点，但不深入嵌套的函数/类（它们各算各的）。"""
    stack: list[ast.AST] = list(ast.iter_child_nodes(root))
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, _SCOPE_NODES):
            continue
        stack.extend(ast.iter_child_nodes(node))


def _complexity(node: ast.AST) -> int:
    """McCabe 圈复杂度近似值：1 + 判定点数量。

    Step 4 用它优先给高风险函数生成测试。
    """
    score = 1
    for child in _iter_scope_nodes(node):
        if isinstance(child, _DECISION_NODES):
            score += 1
        elif isinstance(child, ast.BoolOp):
            # `a and b and c` 有 n-1 个短路分支
            score += max(len(child.values) - 1, 0)
        elif isinstance(child, ast.Match):
            score += len(child.cases)
    return score


def _format_args(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """还原参数列表文本（含类型注解与默认值）。"""
    try:
        return ast.unparse(node.args)
    except Exception:  # noqa: BLE001 - 极端语法节点兜底
        return "..."


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """构造可读签名，例如 `async def fetch(url: str, retries: int = 3) -> str`。"""
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    returns = ""
    if node.returns is not None:
        try:
            returns = f" -> {ast.unparse(node.returns)}"
        except Exception:  # noqa: BLE001
            returns = ""
    return f"{prefix} {node.name}({_format_args(node)}){returns}"


def _collect_imports(tree: ast.Module) -> list[ParsedImport]:
    """收集导入语句：每个被导入模块/符号单独成条，便于 Step 3 检索。"""
    imports: list[ParsedImport] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(
                    ParsedImport(module=alias.name, names=[alias.asname or alias.name],
                                 line=node.lineno)
                )
        elif isinstance(node, ast.ImportFrom):
            # 相对导入用前导点表示层级：from ..pkg import x -> module="..pkg"
            module = f"{'.' * node.level}{node.module or ''}"
            names = [alias.name for alias in node.names]
            imports.append(ParsedImport(module=module, names=names, line=node.lineno))
    return imports


def _function_symbol(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    prefix: str,
    kind: str,
) -> ParsedSymbol:
    """把一个函数/方法节点转成 ParsedSymbol。"""
    qualified = f"{prefix}{node.name}" if prefix else node.name
    return ParsedSymbol(
        name=node.name,
        kind=kind,  # type: ignore[arg-type]
        start_line=node.lineno,
        end_line=getattr(node, "end_lineno", node.lineno) or node.lineno,
        qualified_name=qualified,
        signature=_signature(node),
        docstring=ast.get_docstring(node),
        is_async=isinstance(node, ast.AsyncFunctionDef),
        complexity=_complexity(node),
    )


def _class_symbol(node: ast.ClassDef, prefix: str) -> ParsedSymbol:
    """把一个类节点转成 ParsedSymbol（签名带基类列表）。"""
    qualified = f"{prefix}{node.name}" if prefix else node.name
    bases: list[str] = []
    for base in node.bases:
        try:
            bases.append(ast.unparse(base))
        except Exception:  # noqa: BLE001
            continue
    signature = f"class {node.name}" + (f"({', '.join(bases)})" if bases else "")
    return ParsedSymbol(
        name=node.name,
        kind="class",
        start_line=node.lineno,
        end_line=getattr(node, "end_lineno", node.lineno) or node.lineno,
        qualified_name=qualified,
        signature=signature,
        docstring=ast.get_docstring(node),
        complexity=1,
    )


def _walk_body(
    body: list[ast.stmt],
    prefix: str,
    *,
    inside_class: bool,
    symbols: list[ParsedSymbol],
) -> None:
    """递归遍历语句体，收集函数/方法与类。"""
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # 类体内的函数算方法，其余（含嵌套函数）算函数
            kind = "method" if inside_class else "function"
            symbols.append(_function_symbol(node, prefix, kind))
            # 嵌套函数也记录，qualified_name 带上外层前缀
            _walk_body(
                node.body,
                prefix=f"{prefix}{node.name}." if prefix else f"{node.name}.",
                inside_class=False,
                symbols=symbols,
            )
        elif isinstance(node, ast.ClassDef):
            symbols.append(_class_symbol(node, prefix))
            _walk_body(
                node.body,
                prefix=f"{prefix}{node.name}." if prefix else f"{node.name}.",
                inside_class=True,
                symbols=symbols,
            )


def parse_python(source: str, path: str) -> ParsedFile:
    """解析 Python 源码。

    语法错误不抛异常，而是写入 parse_error 并返回空符号列表 —— 仓库里存在
    语法错误的文件是常态（模板、py2 残留、故意错误用例），不应中断整个索引。
    """
    total_lines = source.count("\n") + 1 if source else 0
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        logger.debug("Python 语法错误，降级处理: %s (%s)", path, exc)
        return ParsedFile(
            path=path,
            language="python",
            total_lines=total_lines,
            parse_error=f"SyntaxError: {exc.msg} (line {exc.lineno})",
        )
    except ValueError as exc:
        # 源码含 NUL 字节等非法内容
        return ParsedFile(
            path=path,
            language="python",
            total_lines=total_lines,
            parse_error=f"ValueError: {exc}",
        )

    symbols: list[ParsedSymbol] = []
    _walk_body(tree.body, prefix="", inside_class=False, symbols=symbols)
    symbols.sort(key=lambda item: (item.start_line, item.name))

    return ParsedFile(
        path=path,
        language="python",
        total_lines=total_lines,
        symbols=symbols,
        imports=_collect_imports(tree),
    )


__all__ = ["parse_python"]
