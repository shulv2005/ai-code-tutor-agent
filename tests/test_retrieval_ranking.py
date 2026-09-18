"""检索排序质量测试：测试代码降权。

背景（实测数据）：在真实仓库 requests 上，测试符号占索引的 62%（463/743），
会把 request() 这类实现函数挤出 Top-10。因此检索默认把测试符号排到实现代码之后。

这里构造一个「实现 + 大量测试」的仓库来锁定该行为，避免以后回归。
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import git
import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import create_app
from app.services.repo.git_service import RepoRef
from app.services.retrieval.chunker import is_test_chunk, is_test_path, is_test_symbol

IMPL = '''
def process_payment(order_id, amount):
    """Charge the customer for an order."""
    if amount <= 0:
        raise ValueError("amount must be positive")
    return {"order_id": order_id, "charged": amount}
'''

# 大量测试符号，模拟真实仓库中测试代码占比很高的情况
TESTS = "\n\n".join(
    f'def test_process_payment_case_{index}(order_id, amount):\n'
    f'    """Charge the customer for an order, case {index}."""\n'
    f"    assert process_payment(order_id, amount)\n"
    for index in range(12)
)


@pytest.fixture()
def mixed_repo(tmp_path: Path) -> Path:
    """实现文件 + 测试目录的仓库，用于验证测试降权。"""
    root = tmp_path / "mixed_src"
    (root / "tests").mkdir(parents=True)
    (root / "payments.py").write_text(IMPL, encoding="utf-8")
    (root / "tests" / "test_payments.py").write_text(TESTS, encoding="utf-8")

    repo = git.Repo.init(root, initial_branch="main")
    repo.index.add(["payments.py", "tests/test_payments.py"])
    actor = git.Actor("Tester", "tester@example.com")
    repo.index.commit("init", author=actor, committer=actor)
    return root


@pytest.fixture()
def mixed_clone(monkeypatch: pytest.MonkeyPatch, mixed_repo: Path) -> None:
    """把 URL 解析指向 mixed_repo。"""

    def fake_parse(url: str, allowed_hosts: object) -> RepoRef:
        return RepoRef(
            host="github.com",
            owner="local",
            name="mixed",
            clone_url=str(mixed_repo),
            sanitized_url="https://github.com/local/mixed.git",
        )

    monkeypatch.setattr("app.services.repo.service.parse_repo_url", fake_parse)


# ---------------------------------------------------------------------------
# 测试代码识别（纯函数）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "path",
    [
        "tests/test_foo.py",
        "test/foo.py",
        "src/__tests__/foo.js",
        "pkg/testing/helpers.go",
        "conftest.py",
        "foo_test.py",
        "widget.test.js",
        "a/spec/b.py",
    ],
)
def test_is_test_path_true(path: str) -> None:
    assert is_test_path(path) is True


@pytest.mark.parametrize(
    "path",
    ["payments.py", "src/requests/api.py", "contest.py", "src/latest.py", "testing_utils.py"],
)
def test_is_test_path_false(path: str) -> None:
    assert is_test_path(path) is False


@pytest.mark.parametrize("name", ["test_foo", "TestClient", "TestA", "should_work", "it_does"])
def test_is_test_symbol_true(name: str) -> None:
    assert is_test_symbol(name) is True


@pytest.mark.parametrize("name", ["process_payment", "latest", "contest", "Testament"])
def test_is_test_symbol_false(name: str) -> None:
    assert is_test_symbol(name) is False


def test_is_test_chunk_combines_both_signals() -> None:
    from app.services.retrieval.dto import CodeChunk

    def chunk(path: str, name: str) -> CodeChunk:
        return CodeChunk(
            symbol_id=1, repository_id=1, path=path, language="python",
            qualified_name=name, kind="function", signature="", docstring=None,
            start_line=1, end_line=2, code="",
        )

    # 非测试目录但符号名是测试样式
    assert is_test_chunk(chunk("src/a.py", "test_something")) is True
    # 测试目录即使符号名正常也算测试代码
    assert is_test_chunk(chunk("tests/helper.py", "build_fixture")) is True
    assert is_test_chunk(chunk("src/a.py", "process_payment")) is False


# ---------------------------------------------------------------------------
# 端到端排序行为
# ---------------------------------------------------------------------------
@contextmanager
def _indexed_client() -> Iterator[tuple[TestClient, int]]:
    """启动应用并完成建库 + 索引。

    必须用 `with TestClient(...)`：FastAPI 的 lifespan（内含 init_db 建表）
    只在上下文管理器内执行，直接 `TestClient(app)` 会因表不存在而报
    "no such table: repositories"。
    """
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/v1/repositories", json={"url": "https://github.com/local/mixed.git"}
        )
        assert response.status_code == 201, response.text
        repo_id = response.json()["repository"]["id"]
        built = client.post(f"/api/v1/repositories/{repo_id}/index")
        assert built.status_code == 200, built.text
        yield client, repo_id


def test_implementation_ranks_above_tests_by_default(
    sqlite_path: Path, repo_workspace: Path, mixed_clone: None, retrieval_env: Path
) -> None:
    """查询命中测试用例与实现函数时，实现函数必须排前面。"""
    with _indexed_client() as (client, repo_id):
        response = client.post(
            "/api/v1/search",
            json={"repository_id": repo_id, "query": "Charge the customer for an order", "top_k": 5},
        )
        assert response.status_code == 200, response.text
        hits = response.json()["hits"]

    assert hits, "不应返回空结果"
    assert hits[0]["qualified_name"] == "process_payment"
    assert hits[0]["path"] == "payments.py"
    # 测试用例不应出现在实现代码之前
    assert not is_test_path(hits[0]["path"])


def test_include_tests_restores_test_hits(
    sqlite_path: Path, repo_workspace: Path, mixed_clone: None, retrieval_env: Path
) -> None:
    """显式要求包含测试时（Step 4 检索现有测试作参考），测试结果应回到原始融合顺序。

    用「字面上最匹配某个测试用例」的查询做对照：默认模式下实现代码仍被前置，
    关闭降权后测试用例按原始相关性排到首位。
    """
    query = "test_process_payment_case_7"

    with _indexed_client() as (client, repo_id):
        default = client.post(
            "/api/v1/search", json={"repository_id": repo_id, "query": query, "top_k": 5}
        ).json()["hits"]
        with_tests = client.post(
            "/api/v1/search",
            json={"repository_id": repo_id, "query": query, "top_k": 5, "include_tests": True},
        ).json()["hits"]

    assert default and with_tests
    # 默认：实现代码被前置，尽管查询字面最贴近某个测试用例
    assert default[0]["qualified_name"] == "process_payment"
    assert not is_test_path(default[0]["path"])
    # 关闭降权：测试用例按原始相关性回到首位
    assert is_test_path(with_tests[0]["path"])
    # 两种模式都能召回该测试用例，只是位置不同
    assert any(hit["qualified_name"] == "test_process_payment_case_7" for hit in default)
    assert with_tests[0]["qualified_name"] == "test_process_payment_case_7"


def test_non_test_hits_form_a_prefix_by_default(
    sqlite_path: Path, repo_workspace: Path, mixed_clone: None, retrieval_env: Path
) -> None:
    """默认排序中，非测试结果必须是前缀（测试只在后面兜底）。

    这是"降权而非过滤"的可观测不变式：结果不足时测试才补位。
    """
    with _indexed_client() as (client, repo_id):
        hits = client.post(
            "/api/v1/search",
            json={"repository_id": repo_id, "query": "Charge the customer", "top_k": 5},
        ).json()["hits"]

    flags = [is_test_path(hit["path"]) for hit in hits]
    assert flags == sorted(flags), f"非测试结果未全部前置: {flags}"
    assert flags[0] is False


def test_tests_are_backfilled_when_nothing_else_matches(
    sqlite_path: Path, repo_workspace: Path, mixed_clone: None, retrieval_env: Path
) -> None:
    """降权而非硬过滤：当只有测试代码能匹配时，仍应返回结果。"""
    with _indexed_client() as (client, repo_id):
        response = client.post(
            "/api/v1/search",
            json={"repository_id": repo_id, "query": "test_process_payment_case_7", "top_k": 3},
        )

    hits = response.json()["hits"]
    assert hits, "降权不应导致结果为空"
    assert any(is_test_path(hit["path"]) for hit in hits)


def test_test_symbol_ids_are_persisted_in_index_meta(
    sqlite_path: Path, repo_workspace: Path, mixed_clone: None, retrieval_env: Path
) -> None:
    """测试符号集合必须随索引落盘，否则重启后降权失效。"""
    import json

    with _indexed_client() as (_, repo_id):
        pass

    meta_path = retrieval_env / f"repo_{repo_id}" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    assert "test_symbol_ids" in meta
    assert len(meta["test_symbol_ids"]) == 12  # 12 个 test_ 函数
    assert len(meta["symbol_ids"]) == 13  # 12 测试 + 1 实现


def test_prefer_non_test_can_be_disabled_by_config(
    sqlite_path: Path, repo_workspace: Path, mixed_clone: None, retrieval_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """配置可关闭降权（满足部分场景下就是要先看测试代码）。"""
    from app.api.deps import reset_retrieval_service

    monkeypatch.setenv("RETRIEVAL__PREFER_NON_TEST", "false")
    get_settings.cache_clear()
    reset_retrieval_service()

    try:
        with _indexed_client() as (client, repo_id):
            hits = client.post(
                "/api/v1/search",
                json={"repository_id": repo_id, "query": "Charge the customer", "top_k": 3},
            ).json()["hits"]
    finally:
        get_settings.cache_clear()
        reset_retrieval_service()

    # 关闭降权后，测试用例可以出现在前面
    assert any(is_test_path(hit["path"]) for hit in hits)
