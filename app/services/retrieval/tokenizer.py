"""代码感知分词器：BM25 关键词召回的第一道质量关口。

BM25 是纯词袋模型，分词质量直接决定关键词召回效果。代码检索有三个特殊需求：
1. **标识符拆词**：`getUserName` / `get_user_name` 都应能命中 `user`、`name`。
2. **中文支持**：Issue 用中文描述、代码用英文标识符，必须能双向对上。
3. **符号保留**：`os.path`、`aiohttp`、`C++` 这类整体不能被切散。

因此这里不用单一的空白切分，而是「驼峰/下划线拆解 + jieba 中文切分 + 原文整体保留」。
"""

from __future__ import annotations

import functools
import logging
import re
from collections.abc import Iterable

logger = logging.getLogger(__name__)

# 驼峰边界：getUserName -> get User Name；HTTPServer -> HTTP Server
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
# 非字母数字（含下划线、点、连字符）作为分隔符
_NON_ALNUM = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff]+")
# 纯 ASCII 标识符
_ASCII_WORD = re.compile(r"^[0-9A-Za-z_]+$")
_HAS_CJK = re.compile(r"[\u4e00-\u9fff]")

try:  # jieba 是可选依赖：缺失时中文退化为单字切分，仍可用
    import jieba

    jieba.setLogLevel(logging.WARNING)
    _JIEBA_AVAILABLE = True
except ImportError:  # pragma: no cover - 依赖已声明，仅防御
    _JIEBA_AVAILABLE = False


def _split_identifier(token: str) -> list[str]:
    """把标识符拆成有意义的子词：getUserName -> [getusername, get, user, name]。"""
    parts: list[str] = []
    for segment in _NON_ALNUM.split(token):
        if not segment:
            continue
        # 先按下划线/连字符切，再按驼峰切
        for piece in _CAMEL_BOUNDARY.split(segment):
            piece = piece.strip()
            if piece:
                parts.append(piece.lower())
    return parts


@functools.lru_cache(maxsize=50_000)
def _tokenize_cached(text: str) -> tuple[str, ...]:
    """分词结果缓存：同一 chunk 会被反复 tokenize（建索引与增量更新）。"""
    return tuple(_tokenize_uncached(text))


def _tokenize_uncached(text: str) -> list[str]:
    tokens: list[str] = []

    for raw in text.split():
        if not raw:
            continue

        if _HAS_CJK.search(raw):
            # 中文：先用 jieba 切词，再对其中夹带的英文标识符做拆解
            if _JIEBA_AVAILABLE:
                pieces: Iterable[str] = jieba.lcut(raw)
            else:
                pieces = list(raw)
            for piece in pieces:
                piece = piece.strip()
                if not piece:
                    continue
                if _HAS_CJK.search(piece):
                    tokens.append(piece.lower())
                else:
                    tokens.extend(_split_identifier(piece))
            continue

        if _ASCII_WORD.match(raw):
            tokens.extend(_split_identifier(raw))
        else:
            # 含标点的复合词（os.path、aiohttp.client）：
            # 既保留整体，也拆出子词，兼顾精确与模糊匹配
            lowered = raw.lower()
            tokens.append(lowered)
            tokens.extend(_split_identifier(raw))

    # 去掉长度 1 的纯符号噪声，但保留单个字母数字（如变量 x、类型 T）
    return [token for token in tokens if token and (len(token) > 1 or token.isalnum())]


def tokenize_code(text: str) -> list[str]:
    """对代码或自然语言文本分词，供 BM25 使用。"""
    if not text:
        return []
    return list(_tokenize_cached(text))


def tokenize_query(query: str) -> list[str]:
    """查询分词。与文档分词共用同一套规则，保证词表对齐。"""
    return tokenize_code(query)


def clear_cache() -> None:
    """清空分词缓存（测试或多语言切换时使用）。"""
    _tokenize_cached.cache_clear()


__all__ = ["clear_cache", "tokenize_code", "tokenize_query"]
