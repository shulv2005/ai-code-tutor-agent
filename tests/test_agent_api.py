"""测试生成 API 测试：三种输入方式、错误码映射与沙箱落盘。

通过 `app.dependency_overrides` 注入假 LLM 客户端，
因此整条 HTTP 链路（含上下文构建、检索、落盘）都可离线验证。
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.agents.test_agent import TestAgent
from app.api.deps import get_test_agent
from app.core.config import get_settings
from app.core.llm_client import set_llm_client
from app.main import create_app
from tests.conftest import FakeLLMClient

REPO_URL = "https://github.com/local/sample.git"
SNIPPET = 'def add(a: int, b: int) -> int:\n    """Add two numbers."""\n    return a + b\n'


@pytest.fixture()
def agent_app(fake_llm: FakeLLMClient) -> Iterator[object]:
    """注入了假 LLM 的应用实例。"""
    app = create_app()
    app.dependency_overrides[get_test_agent] = lambda: TestAgent(get_settings(), fake_llm)
    yield app
    app.dependency_overrides.clear()


def _generate(client: TestClient, payload: dict) -> dict:
    response = client.post("/api/v1/agent/generate_test", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


# ---------------------------------------------------------------------------
# 输入方式 1：直接给代码片段
# ---------------------------------------------------------------------------
def test_generate_from_code_snippet(sqlite_path: Path, llm_env: Path, agent_app: object) -> None:
    with TestClient(agent_app) as client:  # type: ignore[arg-type]
        body = _generate(
            client,
            {"code": SNIPPET, "file_path": "pkg/core.py", "symbol_name": "add"},
        )

    assert body["target"]["path"] == "pkg/core.py"
    assert body["target"]["qualified_name"] == "add"
    assert body["target"]["kind"] == "snippet"
    assert "def test_add_positive_numbers" in body["test_code"]
    assert body["test_functions"] == ["test_add_positive_numbers", "test_add_edge_cases"]
    assert body["model"] == "fake-model"
    assert body["attempts"] == 1
    assert body["usage"]["total_tokens"] == 150
    assert body["trace_id"]
    assert body["prompt_version"]


def test_snippet_is_saved_into_sandbox(
    sqlite_path: Path, llm_env: Path, agent_app: object
) -> None:
    with TestClient(agent_app) as client:  # type: ignore[arg-type]
        body = _generate(client, {"code": SNIPPET, "symbol_name": "add"})

    assert body["saved_path"] is not None
    saved = Path(body["saved_path"])
    assert saved.exists()
    assert saved.name == "test_generated.py"
    assert saved.parent.name == "tests"
    assert saved.parent.parent.name == "sandbox_repo"
    # 沙箱目录必须落在配置的工作区内
    assert llm_env in saved.parents
    assert body["sandbox_dir"] == str(saved.parent.parent)


def test_save_to_sandbox_can_be_disabled(
    sqlite_path: Path, llm_env: Path, agent_app: object
) -> None:
    with TestClient(agent_app) as client:  # type: ignore[arg-type]
        body = _generate(client, {"code": SNIPPET, "save_to_sandbox": False})

    assert body["saved_path"] is None
    assert body["sandbox_dir"] is None


# ---------------------------------------------------------------------------
# 输入方式 2/3：仓库符号 / 自然语言检索
# ---------------------------------------------------------------------------
def test_generate_from_repository_symbol(
    sqlite_path: Path, repo_workspace: Path, offline_clone: None, llm_env: Path, agent_app: object
) -> None:
    with TestClient(agent_app) as client:  # type: ignore[arg-type]
        created = client.post("/api/v1/repositories", json={"url": REPO_URL})
        repo_id = created.json()["repository"]["id"]

        structure = client.get(
            f"/api/v1/repositories/{repo_id}/structure", params={"limit": 50}
        ).json()
        symbol = next(
            item
            for item in structure["files"]
            if item["path"] == "pkg/core.py"
            for item in item["symbols"]
            if item["qualified_name"] == "add"
        )

        body = _generate(
            client,
            {"repository_id": repo_id, "symbol_id": symbol["id"]},
        )

    assert body["target"]["qualified_name"] == "add"
    assert body["target"]["path"] == "pkg/core.py"
    assert body["target"]["kind"] == "function"
    assert body["target"]["signature"] == "def add(a: int, b: int) -> int"
    assert body["target"]["repository"] == "local/sample"
    # 同文件相关符号应被带上（fetch / Service 等）
    assert body["target"]["related_count"] > 0


def test_generate_from_natural_language_query(
    sqlite_path: Path,
    repo_workspace: Path,
    offline_clone: None,
    retrieval_env: Path,
    llm_env: Path,
    agent_app: object,
) -> None:
    """Step 3 检索 -> Step 4 生成 的串联。"""
    with TestClient(agent_app) as client:  # type: ignore[arg-type]
        created = client.post("/api/v1/repositories", json={"url": REPO_URL})
        repo_id = created.json()["repository"]["id"]
        assert client.post(f"/api/v1/repositories/{repo_id}/index").status_code == 200

        body = _generate(
            client,
            {"repository_id": repo_id, "query": "Add two numbers"},
        )

    assert body["target"]["qualified_name"] == "add"
    assert "def test_add" in body["test_code"]


def test_query_mode_without_index_returns_409(
    sqlite_path: Path,
    repo_workspace: Path,
    offline_clone: None,
    retrieval_env: Path,
    llm_env: Path,
    agent_app: object,
) -> None:
    """显式使用隔离的空索引目录：本用例的前提就是"索引不存在"。"""
    with TestClient(agent_app) as client:  # type: ignore[arg-type]
        created = client.post("/api/v1/repositories", json={"url": REPO_URL})
        repo_id = created.json()["repository"]["id"]
        response = client.post(
            "/api/v1/agent/generate_test",
            json={"repository_id": repo_id, "query": "add"},
        )

    assert response.status_code == 409


# ---------------------------------------------------------------------------
# 错误处理
# ---------------------------------------------------------------------------
def test_missing_input_returns_422(sqlite_path: Path, llm_env: Path, agent_app: object) -> None:
    with TestClient(agent_app) as client:  # type: ignore[arg-type]
        response = client.post("/api/v1/agent/generate_test", json={})
        empty_code = client.post("/api/v1/agent/generate_test", json={"code": "   "})

    assert response.status_code == 422
    assert empty_code.status_code == 422


def test_unknown_repository_returns_404(
    sqlite_path: Path, llm_env: Path, agent_app: object
) -> None:
    with TestClient(agent_app) as client:  # type: ignore[arg-type]
        response = client.post(
            "/api/v1/agent/generate_test", json={"repository_id": 999999, "query": "x"}
        )
    assert response.status_code == 404


def test_unknown_symbol_returns_404(
    sqlite_path: Path, repo_workspace: Path, offline_clone: None, llm_env: Path, agent_app: object
) -> None:
    with TestClient(agent_app) as client:  # type: ignore[arg-type]
        created = client.post("/api/v1/repositories", json={"url": REPO_URL})
        repo_id = created.json()["repository"]["id"]
        response = client.post(
            "/api/v1/agent/generate_test",
            json={"repository_id": repo_id, "symbol_id": 999999},
        )
    assert response.status_code == 404


def test_unconfigured_llm_returns_503(
    sqlite_path: Path, llm_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """未配置 API Key 时用 503（服务端未就绪），与参数错误区分开。"""
    monkeypatch.setenv("LLM__API_KEY", "")
    get_settings.cache_clear()
    set_llm_client(None)

    app = create_app()
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/agent/generate_test", json={"code": SNIPPET}
            )
    finally:
        get_settings.cache_clear()

    assert response.status_code == 503
    assert "LLM__API_KEY" in response.json()["detail"]


def test_generation_failure_returns_422(
    sqlite_path: Path, llm_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """模型始终给不出可用代码时，应返回 422 并说明原因。"""
    broken = FakeLLMClient("```python\nno tests\n```")
    app = create_app()
    app.dependency_overrides[get_test_agent] = lambda: TestAgent(get_settings(), broken)
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/agent/generate_test", json={"code": SNIPPET, "max_attempts": 1}
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422
    assert "未生成可用的测试代码" in response.json()["detail"]
    assert broken.call_count == 1


# ---------------------------------------------------------------------------
# 状态接口
# ---------------------------------------------------------------------------
def test_llm_status_endpoint(sqlite_path: Path, llm_env: Path, agent_app: object) -> None:
    with TestClient(agent_app) as client:  # type: ignore[arg-type]
        response = client.get("/api/v1/agent/status")

    assert response.status_code == 200
    body = response.json()
    assert body["configured"] is True  # 假客户端自称已配置
    assert body["model"] == "fake-model"
    assert body["prompt_version"]


def test_openapi_exposes_agent_routes(
    sqlite_path: Path, llm_env: Path, agent_app: object
) -> None:
    with TestClient(agent_app) as client:  # type: ignore[arg-type]
        paths = client.get("/openapi.json").json()["paths"]

    assert "/api/v1/agent/generate_test" in paths
    assert "/api/v1/agent/status" in paths
