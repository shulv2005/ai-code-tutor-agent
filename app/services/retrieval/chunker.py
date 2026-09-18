"""代码分块：把 CodeSymbol + 源码切片组装成可检索的 CodeChunk。

关键设计：
1. **按文件分组读取**：一万个符号可能只分布在几百个文件里，逐个符号读文件会
   产生大量重复 IO，因此先按 CodeFile 分组，每个文件只读一次。
2. **行号切片而非重新解析**：Step 2 已把 start_line/end_line 存好，
   这里直接按行切片，避免二次解析开销。
3. **读取失败不中断**：文件被删/编码异常时降级为「只有签名无正文」，
   该符号仍可被关键词检索命中。
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from app.models.code import CodeFile, CodeSymbol
from app.services.retrieval.dto import CodeChunk

logger = logging.getLogger(__name__)

# 测试代码识别：实测在真实仓库（requests）上，测试符号占索引的 62%，
# 会把 request() 这类实现函数挤到 Top-10 之外，因此必须能区分并降权。
_TEST_DIR_NAMES = frozenset(
    {"tests", "test", "testing", "spec", "specs", "__tests__", "e2e", "integration"}
)
# 注意第二个分支必须以 .* 开头：re.match 从位置 0 起匹配，
# 否则 "widget.test.js" 这种「.test. 出现在中间」的文件名会被漏判。
_TEST_FILE_PATTERN = re.compile(r"^(test_.*|.*_test|conftest)\.\w+$|.*\.(test|spec)\.\w+$")
_TEST_SYMBOL_PATTERN = re.compile(r"^(test_|Test[A-Z_]|should_|it_)")


def is_test_path(path: str) -> bool:
    """按路径判断是否测试文件。"""
    parts = path.replace("\\", "/").split("/")
    if any(part.lower() in _TEST_DIR_NAMES for part in parts[:-1]):
        return True
    return bool(parts and _TEST_FILE_PATTERN.match(parts[-1].lower()))


def is_test_symbol(qualified_name: str) -> bool:
    """按符号名判断是否测试用例（非测试目录里的 TestX 类也要能识别）。"""
    return bool(_TEST_SYMBOL_PATTERN.match(qualified_name.split(".")[-1]))


def is_test_chunk(chunk: CodeChunk) -> bool:
    """判断分块是否属于测试代码。"""
    return is_test_path(chunk.path) or is_test_symbol(chunk.qualified_name)


def _read_lines(path: Path) -> list[str] | None:
    """读取文件所有行；失败返回 None。"""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        logger.debug("读取源码失败: %s", path, exc_info=True)
        return None
    return text.splitlines()


def slice_symbol(lines: Sequence[str], start_line: int, end_line: int) -> str:
    """按 1-based 闭区间行号切出源码（行号越界自动收敛）。"""
    if not lines:
        return ""
    start = max(start_line - 1, 0)
    end = min(end_line, len(lines))
    if start >= end:
        return ""
    return "\n".join(lines[start:end])


def build_chunks(
    pairs: Sequence[tuple[CodeSymbol, CodeFile]],
    repo_root: Path,
    *,
    max_chunk_chars: int,
) -> list[CodeChunk]:
    """把 (符号, 文件) 列表转成 CodeChunk 列表。

    Args:
        pairs: 符号与其所属文件。
        repo_root: 仓库本地根目录，用于定位源码。
        max_chunk_chars: 单块正文上限，避免超大函数拖慢嵌入。
    """
    by_path: dict[str, list[tuple[CodeSymbol, CodeFile]]] = defaultdict(list)
    for symbol, code_file in pairs:
        by_path[code_file.path].append((symbol, code_file))

    chunks: list[CodeChunk] = []
    for path, items in by_path.items():
        lines = _read_lines(repo_root / path)
        for symbol, code_file in items:
            code = slice_symbol(lines, symbol.start_line, symbol.end_line) if lines else ""
            if len(code) > max_chunk_chars:
                code = code[:max_chunk_chars]
            chunks.append(
                CodeChunk(
                    symbol_id=symbol.id,
                    repository_id=symbol.repository_id,
                    path=path,
                    language=code_file.language,
                    qualified_name=symbol.qualified_name,
                    kind=symbol.kind,
                    signature=symbol.signature,
                    docstring=symbol.docstring,
                    start_line=symbol.start_line,
                    end_line=symbol.end_line,
                    code=code,
                )
            )

    # 按 symbol_id 排序，保证索引构建顺序确定（便于测试与增量比对）
    chunks.sort(key=lambda item: item.symbol_id)
    return chunks


__all__ = [
    "build_chunks",
    "is_test_chunk",
    "is_test_path",
    "is_test_symbol",
    "slice_symbol",
]
