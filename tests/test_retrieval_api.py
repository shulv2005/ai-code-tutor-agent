"""混合检索 API 全链路测试：建索引 -> 检索 -> 陈旧检测。

使用 hashing 嵌入后端（见 retrieval_env 夹具），因此测试完全离线、
确定性可重复。语义质量由 test_semantic_retrieval.py 单独覆盖。
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app

REPO_URL = "https://github.com/local/sample.git"
EXPECTED_SYMBOLS = 12


def _setup_repo(client: TestClient) -> int:
    """注册并索引样例仓库，返回 repository_id。"""
    response = client.post("/api/v1/repositories", json={"url": REPO_URL})
    assert response.status_code == 201, response.text
    return response.json()["repository"]["id"]


def _build_index(client: TestClient, repo_id: int, **params: object) -> dict:
    response = client.post(f"/api/v1/repositories/{repo_id}/index", params=params)
    assert response.status_code == 200, response.text
    return response.json()


def _search(client: TestClient, repo_id: int, query: str, **extra: object) -> dict:
    response = client.post(
        "/api/v1/search", json={"repository_id": repo_id, "query": query, **extra}
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_build_index_end_to_end(
    sqlite_path: Path, repo_workspace: Path, offline_clone: None, retrieval_env: Path
) -> None:
    app = create_app()
    with TestClient(app) as client:
        repo_id = _setup_repo(client)
        body = _build_index(client, repo_id)

    assert body["repository_id"] == repo_id
    assert body["chunk_count"] == EXPECTED_SYMBOLS
    assert body["vector_count"] == EXPECTED_SYMBOLS
    assert body["dimension"] == 256
    assert body["embedder"] == "hashing"
    assert body["skipped"] is False
    assert body["trace_id"]

    # 索引文件应落盘
    directory = retrieval_env / f"repo_{repo_id}"
    assert (directory / "vectors.faiss").exists()
    assert (directory / "bm25.json").exists()
    assert (directory / "meta.json").exists()


def test_rebuild_is_skipped_when_index_is_current(
    sqlite_path: Path, repo_workspace: Path, offline_clone: None, retrieval_env: Path
) -> None:
    app = create_app()
    with TestClient(app) as client:
        repo_id = _setup_repo(client)
        first = _build_index(client, repo_id)
        second = _build_index(client, repo_id)
        forced = _build_index(client, repo_id, force=True)

    assert first["skipped"] is False
    assert second["skipped"] is True  # head_commit 未变，复用
    assert forced["skipped"] is False  # force 强制重建


def test_search_missing_index_returns_409(
    sqlite_path: Path, repo_workspace: Path, offline_clone: None, retrieval_env: Path
) -> None:
    app = create_app()
    with TestClient(app) as client:
        repo_id = _setup_repo(client)
        response = client.post(
            "/api/v1/search", json={"repository_id": repo_id, "query": "add"}
        )

    assert response.status_code == 409
    assert "索引" in response.json()["detail"]


def test_keyword_search_hits_expected_symbols(
    sqlite_path: Path, repo_workspace: Path, offline_clone: None, retrieval_env: Path
) -> None:
    app = create_app()
    with TestClient(app) as client:
        repo_id = _setup_repo(client)
        _build_index(client, repo_id)

        by_docstring = _search(client, repo_id, "Add two numbers")
        by_signature = _search(client, repo_id, "fetch url retries")

    top_docstring = by_docstring["hits"][0]
    assert top_docstring["qualified_name"] == "add"
    assert top_docstring["path"] == "pkg/core.py"
    assert top_docstring["language"] == "python"

    top_signature = by_signature["hits"][0]
    assert top_signature["qualified_name"] == "fetch"
    assert top_signature["signature"].startswith("async def fetch")


def test_search_hits_carry_usable_code_context(
    sqlite_path: Path, repo_workspace: Path, offline_clone: None, retrieval_env: Path
) -> None:
    """命中必须自带可直接喂给 Agent 的源码片段与行号。"""
    app = create_app()
    with TestClient(app) as client:
        repo_id = _setup_repo(client)
        _build_index(client, repo_id)
        body = _search(client, repo_id, "Add two numbers")

    hit = body["hits"][0]
    assert hit["code"], "命中必须带源码片段"
    assert "def add" in hit["code"]
    assert hit["start_line"] < hit["end_line"]
    # 分路明细可用于解释排序原因
    assert hit["bm25_rank"] is not None
    assert body["timings_ms"]  # 各阶段耗时应被记录
    assert body["trace_id"]


def test_search_across_languages(
    sqlite_path: Path, repo_workspace: Path, offline_clone: None, retrieval_env: Path
) -> None:
    app = create_app()
    with TestClient(app) as client:
        repo_id = _setup_repo(client)
        _build_index(client, repo_id)
        js = _search(client, repo_id, "greet")
        go = _search(client, repo_id, "helper")

    assert js["hits"][0]["language"] == "javascript"
    assert js["hits"][0]["qualified_name"] == "greet"
    assert go["hits"][0]["language"] == "go"
    assert go["hits"][0]["qualified_name"] == "helper"


def test_search_language_filter(
    sqlite_path: Path, repo_workspace: Path, offline_clone: None, retrieval_env: Path
) -> None:
    app = create_app()
    with TestClient(app) as client:
        repo_id = _setup_repo(client)
        _build_index(client, repo_id)
        filtered = _search(client, repo_id, "add", language="go")

    assert all(hit["language"] == "go" for hit in filtered["hits"])


def test_search_top_k_is_respected(
    sqlite_path: Path, repo_workspace: Path, offline_clone: None, retrieval_env: Path
) -> None:
    app = create_app()
    with TestClient(app) as client:
        repo_id = _setup_repo(client)
        _build_index(client, repo_id)
        body = _search(client, repo_id, "def", top_k=3)

    assert len(body["hits"]) <= 3
    assert body["total"] == len(body["hits"])


def test_search_scores_are_descending(
    sqlite_path: Path, repo_workspace: Path, offline_clone: None, retrieval_env: Path
) -> None:
    app = create_app()
    with TestClient(app) as client:
        repo_id = _setup_repo(client)
        _build_index(client, repo_id)
        hits = _search(client, repo_id, "config", top_k=5)["hits"]

    scores = [hit["score"] for hit in hits]
    assert scores == sorted(scores, reverse=True)


def test_search_unknown_repository_returns_404(
    sqlite_path: Path, repo_workspace: Path, offline_clone: None, retrieval_env: Path
) -> None:
    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/search", json={"repository_id": 999999, "query": "add"}
        )
    assert response.status_code == 404


def test_search_validation_errors(
    sqlite_path: Path, repo_workspace: Path, offline_clone: None, retrieval_env: Path
) -> None:
    app = create_app()
    with TestClient(app) as client:
        repo_id = _setup_repo(client)
        _build_index(client, repo_id)
        empty_query = client.post(
            "/api/v1/search", json={"repository_id": repo_id, "query": ""}
        )
        bad_k = client.post(
            "/api/v1/search", json={"repository_id": repo_id, "query": "a", "top_k": 0}
        )

    assert empty_query.status_code == 422
    assert bad_k.status_code == 422


def test_index_unknown_repository_returns_404(
    sqlite_path: Path, repo_workspace: Path, offline_clone: None, retrieval_env: Path
) -> None:
    app = create_app()
    with TestClient(app) as client:
        response = client.post("/api/v1/repositories/999999/index")
    assert response.status_code == 404


def test_index_without_symbols_returns_409(
    sqlite_path: Path, repo_workspace: Path, offline_clone: None, retrieval_env: Path
) -> None:
    """仓库存在但没有任何符号时，应明确报错而不是建出空索引。"""
    import asyncio

    from sqlalchemy import delete

    from app.core.database import get_session
    from app.models.code import CodeSymbol

    app = create_app()
    with TestClient(app) as client:
        repo_id = _setup_repo(client)

        async def wipe() -> None:
            async with get_session() as session:
                await session.execute(
                    delete(CodeSymbol).where(CodeSymbol.repository_id == repo_id)
                )
                await session.commit()

        asyncio.run(wipe())
        response = client.post(f"/api/v1/repositories/{repo_id}/index")

    assert response.status_code == 409
    assert "符号" in response.json()["detail"]


def test_stale_index_is_flagged(
    sqlite_path: Path, repo_workspace: Path, offline_clone: None, retrieval_env: Path
) -> None:
    """代码更新后索引应被标记为陈旧，并给出重建提示。"""
    import git

    app = create_app()
    with TestClient(app) as client:
        repo_id = _setup_repo(client)
        _build_index(client, repo_id)
        assert _search(client, repo_id, "add")["stale"] is False

        # 模拟上游有新提交
        repository = client.get(f"/api/v1/repositories/{repo_id}").json()
        local = Path(repository["local_path"])
        repo = git.Repo(local)
        (local / "pkg" / "new.py").write_text("def brand_new():\n    pass\n", encoding="utf-8")
        repo.index.add(["pkg/new.py"])
        actor = git.Actor("T", "t@e.com")
        commit = repo.index.commit("new commit", author=actor, committer=actor)
        repo.close()

        # 直接更新数据库中的 head_commit（模拟重新克隆后未重建索引）
        async def bump() -> None:
            from app.core.database import get_session
            from app.models.repository import Repository

            async with get_session() as session:
                row = await session.get(Repository, repo_id)
                assert row is not None
                row.head_commit = commit.hexsha
                await session.commit()

        import asyncio

        asyncio.run(bump())

        body = _search(client, repo_id, "add")

    assert body["stale"] is True
    assert body["notes"]


def test_openapi_exposes_search_routes(
    sqlite_path: Path, repo_workspace: Path, offline_clone: None, retrieval_env: Path
) -> None:
    app = create_app()
    with TestClient(app) as client:
        paths = client.get("/openapi.json").json()["paths"]

    assert "/api/v1/search" in paths
    assert "/api/v1/repositories/{repository_id}/index" in paths
