"""解析结果 DTO：解析层与持久化层之间的稳定契约。

刻意与 ORM 解耦：解析器只产出这些纯数据对象，由 service 层决定如何落库。
这样 Step 4/6 的 Agent 也能直接复用解析结果，无需触碰数据库。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# 与 CodeSymbol.kind 取值保持一致
SymbolKind = Literal["function", "method", "class", "struct", "interface"]


@dataclass(slots=True)
class ParsedSymbol:
    """文件内的一个可定位符号。"""

    name: str
    kind: SymbolKind
    # 1-based 闭区间行号，供 Step 4/6 精确切片
    start_line: int
    end_line: int
    qualified_name: str = ""
    signature: str = ""
    docstring: str | None = None
    is_async: bool = False
    complexity: int = 1

    def __post_init__(self) -> None:
        if not self.qualified_name:
            self.qualified_name = self.name


@dataclass(slots=True)
class ParsedImport:
    """一条导入语句。"""

    module: str
    names: list[str] = field(default_factory=list)
    line: int = 0


@dataclass(slots=True)
class ParsedFile:
    """单个文件的完整解析结果。"""

    path: str
    language: str
    total_lines: int = 0
    size_bytes: int = 0
    symbols: list[ParsedSymbol] = field(default_factory=list)
    imports: list[ParsedImport] = field(default_factory=list)
    # 非 None 表示解析降级（语法错误 / 缺语法包 / 文件超限），文件仍入库
    parse_error: str | None = None

    @property
    def symbol_count(self) -> int:
        return len(self.symbols)

    def iter_lines(self, source: str) -> dict[str, str]:
        """按符号切片源码：{qualified_name: 代码文本}，供 LLM 提示词使用。"""
        lines = source.splitlines()
        result: dict[str, str] = {}
        for symbol in self.symbols:
            start = max(symbol.start_line - 1, 0)
            end = min(symbol.end_line, len(lines))
            result[symbol.qualified_name] = "\n".join(lines[start:end])
        return result
