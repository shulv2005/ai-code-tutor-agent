"""仓库解析 API 全链路测试：POST 注册索引 -> GET 结构 -> 重新索引。

通过 offline_clone 夹具把 URL 指向本地样例仓库，因此除 URL 校验外，
克隆、解析、落库走的都是真实代码。
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app

REPO_URL = "https://github.com/local/sample.git"

# 样例仓库：pkg/{__init__,core,broken}.py + web.js + main.go
EXPECTED_FILES = 5
EXPECTED_SYMBOLS = 12
EXPECTED_FAILED = 1  # broken.py 语法错误


def _index(client: TestClient, url: str = REPO_URL, **extra: object) -> dict:
    response = client.post("/api/v1/repositories", json={"url": url, **extra})
    assert response.status_code == 201, response.text
    return response.json()


def test_index_repository_end_to_end(
    sqlite_path: Path, repo_workspace: Path, offline_clone: None
) -> None:
    app = create_app()
    with TestClient(app) as client:
        body = _index(client)

    repository = body["repository"]
    stats = body["stats"]

    assert repository["status"] == "ready"
    assert repository["owner"] == "local"
    assert repository["host"] == "github.com"
    assert repository["default_branch"] == "main"
    assert repository["head_commit"]
    assert repository["error_message"] is None

    assert stats["file_count"] == EXPECTED_FILES
    assert stats["symbol_count"] == EXPECTED_SYMBOLS
    assert stats["failed_files"] == EXPECTED_FAILED
    assert stats["skipped"] is False
    # 索引过程应写入 trace_id
    assert body["trace_id"]


def test_structure_endpoint_returns_symbols(
    sqlite_path: Path, repo_workspace: Path, offline_clone: None
) -> None:
    app = create_app()
    with TestClient(app) as client:
        repo_id = _index(client)["repository"]["id"]
        response = client.get(f"/api/v1/repositories/{repo_id}/structure")

    assert response.status_code == 200
    body = response.json()
    assert body["total_files"] == EXPECTED_FILES
    assert body["languages"]["python"] == 3
    assert body["languages"]["javascript"] == 1
    assert body["languages"]["go"] == 1

    by_path = {item["path"]: item for item in body["files"]}
    assert "pkg/core.py" in by_path

    core = by_path["pkg/core.py"]
    names = {symbol["qualified_name"] for symbol in core["symbols"]}
    assert {"add", "fetch", "Service", "Service.run"} <= names

    add = next(s for s in core["symbols"] if s["qualified_name"] == "add")
    assert add["kind"] == "function"
    assert add["signature"] == "def add(a: int, b: int) -> int"
    assert add["docstring"] == "Add two numbers."
    assert add["start_line"] < add["end_line"]

    # 语法错误文件仍入库，并记录原因
    assert by_path["pkg/broken.py"]["parse_error"] is not None
    assert by_path["pkg/broken.py"]["symbol_count"] == 0


def test_structure_filters_by_language(
    sqlite_path: Path, repo_workspace: Path, offline_clone: None
) -> None:
    app = create_app()
    with TestClient(app) as client:
        repo_id = _index(client)["repository"]["id"]
        response = client.get(
            f"/api/v1/repositories/{repo_id}/structure", params={"language": "go"}
        )

    body = response.json()
    assert [item["path"] for item in body["files"]] == ["main.go"]
    # 语言分布统计不受分页/过滤影响
    assert body["languages"]["python"] == 3


def test_second_index_is_skipped_unless_forced(
    sqlite_path: Path, repo_workspace: Path, offline_clone: None
) -> None:
    app = create_app()
    with TestClient(app) as client:
        first = _index(client)
        second = _index(client)
        forced = _index(client, force=True)

    # 幂等：同一仓库不重复克隆，返回缓存结果
    assert second["repository"]["id"] == first["repository"]["id"]
    assert second["stats"]["skipped"] is True

    assert forced["repository"]["id"] == first["repository"]["id"]
    assert forced["stats"]["skipped"] is False
    assert forced["stats"]["symbol_count"] == EXPECTED_SYMBOLS


def test_list_and_detail_endpoints(
    sqlite_path: Path, repo_workspace: Path, offline_clone: None
) -> None:
    app = create_app()
    with TestClient(app) as client:
        repo_id = _index(client)["repository"]["id"]

        listing = client.get("/api/v1/repositories")
        detail = client.get(f"/api/v1/repositories/{repo_id}")
        missing = client.get("/api/v1/repositories/999999")

    assert listing.status_code == 200
    assert listing.json()["total"] == 1
    assert listing.json()["items"][0]["id"] == repo_id

    assert detail.status_code == 200
    assert detail.json()["id"] == repo_id

    assert missing.status_code == 404


def test_reindex_endpoint(
    sqlite_path: Path, repo_workspace: Path, offline_clone: None
) -> None:
    app = create_app()
    with TestClient(app) as client:
        repo_id = _index(client)["repository"]["id"]
        response = client.post(f"/api/v1/repositories/{repo_id}/reindex")

    assert response.status_code == 200
    body = response.json()
    assert body["stats"]["file_count"] == EXPECTED_FILES
    assert body["stats"]["symbol_count"] == EXPECTED_SYMBOLS


def test_reindex_without_local_copy_returns_409(
    sqlite_path: Path, repo_workspace: Path, offline_clone: None
) -> None:
    from app.services.repo.git_service import remove_tree

    app = create_app()
    with TestClient(app) as client:
        body = _index(client)
        repo_id = body["repository"]["id"]
        remove_tree(Path(body["repository"]["local_path"]))

        response = client.post(f"/api/v1/repositories/{repo_id}/reindex")

    assert response.status_code == 409


def test_invalid_url_returns_400(sqlite_path: Path, repo_workspace: Path) -> None:
    """白名单外主机必须在克隆前就被拒绝。"""
    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/repositories", json={"url": "https://evil.example.com/a/b.git"}
        )
        local_file = client.post(
            "/api/v1/repositories", json={"url": "file:///etc/passwd"}
        )
        empty = client.post("/api/v1/repositories", json={"url": ""})

    assert response.status_code == 400
    assert "白名单" in response.json()["detail"]
    assert local_file.status_code == 400
    assert empty.status_code == 422  # pydantic 校验


def test_languages_endpoint(sqlite_path: Path) -> None:
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/api/v1/repositories/languages")

    assert response.status_code == 200
    assert {"python", "javascript", "go"} <= set(response.json()["languages"])


def test_openapi_exposes_repository_routes(sqlite_path: Path) -> None:
    app = create_app()
    with TestClient(app) as client:
        paths = client.get("/openapi.json").json()["paths"]

    assert "/api/v1/repositories" in paths
    assert "/api/v1/repositories/{repository_id}/structure" in paths
