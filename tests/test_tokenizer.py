"""分词器测试：BM25 关键词召回的质量取决于此。"""

from __future__ import annotations

import pytest

from app.services.retrieval.tokenizer import tokenize_code, tokenize_query


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # 驼峰与下划线必须收敛到同一组词，否则 Issue 里的写法与代码对不上
        ("getUserName", ["get", "user", "name"]),
        ("get_user_name", ["get", "user", "name"]),
        ("HTTPServer", ["http", "server"]),
        ("parseXMLFile", ["parse", "xml", "file"]),
        ("aiohttp", ["aiohttp"]),
    ],
)
def test_identifier_splitting(text: str, expected: list[str]) -> None:
    assert tokenize_code(text) == expected


def test_camel_and_snake_are_equivalent() -> None:
    assert tokenize_code("fetchUserData") == tokenize_code("fetch_user_data")


def test_dotted_path_keeps_whole_and_parts() -> None:
    tokens = tokenize_code("os.path.join")
    assert "os.path.join" in tokens  # 整体保留，支持精确匹配
    assert "join" in tokens  # 拆出子词，支持模糊匹配


def test_chinese_is_tokenized() -> None:
    tokens = tokenize_code("解析配置文件")
    assert tokens
    assert all(token.strip() for token in tokens)
    # 不应退化成一整串
    assert len(tokens) >= 2


def test_mixed_chinese_and_identifier() -> None:
    """中文句子里夹带英文标识符是 Issue 的常见形态。"""
    tokens = tokenize_code("调用 fetchData 获取用户信息")
    assert "fetch" in tokens
    assert "data" in tokens


def test_query_and_document_share_vocabulary() -> None:
    """查询与文档必须用同一套分词规则，否则词表对不齐。"""
    assert tokenize_query("getUserName") == tokenize_code("getUserName")


def test_empty_and_whitespace() -> None:
    assert tokenize_code("") == []
    assert tokenize_code("   \n\t ") == []


def test_short_alnum_tokens_survive() -> None:
    """单字母变量名/类型参数不应被当作噪声丢掉。"""
    tokens = tokenize_code("T x")
    assert "t" in tokens
    assert "x" in tokens


def test_punctuation_only_input_yields_nothing_meaningful() -> None:
    assert all(len(token) > 0 for token in tokenize_code("()[]{}"))
