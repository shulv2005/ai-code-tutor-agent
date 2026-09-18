"""解析层统一门面：对外只暴露「给文件路径/字节，拿 ParsedFile」。

选择策略：
- `.py` -> 内置 ast（语义信息最全）
- 其它已注册语言 -> tree-sitter（多语言扩展）
- 语法包缺失或语言未收录 -> 降级为「只统计行数」，并写入 parse_error

降级而非报错是有意设计：仓库里必然混有非源码文件与巨型生成文件，
单个文件解析失败不应中断整仓库索引。
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

from app.services.repo.dto import ParsedFile
from app.services.repo.language import detect_language, get_profile, load_grammar
from app.services.repo.python_parser import parse_python
from app.services.repo.tree_sitter_parser import GrammarUnavailableError, parse_with_tree_sitter

logger = logging.getLogger(__name__)


def decode_source(raw: bytes) -> str:
    """把源码字节解码成文本，尽量不抛异常。

    latin-1 兜底是刻意的：它对任意字节序列都不会失败，避免个别非 UTF-8
    文件（GBK 注释、二进制误入）中断整个索引。
    """
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def parse_source(source: bytes, path: str, language: str | None = None) -> ParsedFile:
    """解析单个文件的源码字节。"""
    resolved = language or detect_language(path)
    size_bytes = len(source)
    if resolved is None:
        return ParsedFile(
            path=path,
            language="unknown",
            total_lines=source.count(b"\n") + 1 if source else 0,
            size_bytes=size_bytes,
            parse_error="语言未收录，跳过符号解析",
        )

    if resolved == "python":
        parsed = parse_python(decode_source(source), path)
        parsed.size_bytes = size_bytes
        return parsed

    # 非 Python：需要有 profile 且语法包可用
    if get_profile(resolved) is None or load_grammar(resolved) is None:
        return ParsedFile(
            path=path,
            language=resolved,
            total_lines=source.count(b"\n") + 1 if source else 0,
            size_bytes=size_bytes,
            parse_error=f"语法包不可用，{resolved} 降级为仅统计行数",
        )

    try:
        parsed = parse_with_tree_sitter(source, path, resolved)
    except GrammarUnavailableError as exc:
        return ParsedFile(
            path=path,
            language=resolved,
            total_lines=source.count(b"\n") + 1 if source else 0,
            size_bytes=size_bytes,
            parse_error=str(exc),
        )
    except Exception as exc:  # noqa: BLE001 - 语法包版本差异兜底，不让单文件炸掉索引
        logger.warning("解析失败: %s", path, exc_info=True)
        return ParsedFile(
            path=path,
            language=resolved,
            total_lines=source.count(b"\n") + 1 if source else 0,
            size_bytes=size_bytes,
            parse_error=f"{type(exc).__name__}: {exc}",
        )

    parsed.size_bytes = size_bytes
    return parsed


def to_posix_relative(path: Path, root: Path) -> str:
    """转成仓库内相对路径，统一 '/' 分隔（跨平台一致，供 Step 3 检索与 Step 6 patch）。"""
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        relative = path
    return relative.as_posix()


def iter_source_files(
    root: Path,
    *,
    excluded_dirs: frozenset[str] | set[str],
    max_files: int | None = None,
) -> Iterator[Path]:
    """遍历仓库中的源码文件，跳过排除目录。

    用 os.walk 风格的剪枝而不是 rglob：命中 node_modules 这类目录时
    直接整棵子树跳过，缺了剪枝在真实仓库上会慢几个数量级。
    """
    count = 0
    stack: list[Path] = [root]
    while stack:
        current = stack.pop()
        try:
            entries = sorted(current.iterdir(), key=lambda item: item.name)
        except (OSError, PermissionError):
            logger.debug("目录不可读，跳过: %s", current)
            continue
        for entry in entries:
            if entry.is_symlink():
                # 不跟随符号链接，避免目录环与越界读取
                continue
            if entry.is_dir():
                if entry.name in excluded_dirs:
                    continue
                stack.append(entry)
            elif entry.is_file():
                if detect_language(entry.name) is None:
                    continue
                if max_files is not None and count >= max_files:
                    return
                count += 1
                yield entry


def parse_file(path: Path, root: Path, *, max_file_bytes: int) -> ParsedFile:
    """读取并解析磁盘上的单个文件。"""
    relative = to_posix_relative(path, root)
    language = detect_language(path) or "unknown"
    try:
        size = path.stat().st_size
    except OSError as exc:
        return ParsedFile(path=relative, language=language, parse_error=f"stat 失败: {exc}")

    if size > max_file_bytes:
        return ParsedFile(
            path=relative,
            language=language,
            total_lines=0,
            size_bytes=size,
            parse_error=f"文件超过解析上限 {max_file_bytes} 字节（实际 {size}）",
        )

    try:
        raw = path.read_bytes()
    except OSError as exc:
        return ParsedFile(path=relative, language=language, parse_error=f"读取失败: {exc}")

    return parse_source(raw, relative, language)


__all__ = [
    "decode_source",
    "iter_source_files",
    "parse_file",
    "parse_source",
    "to_posix_relative",
]
