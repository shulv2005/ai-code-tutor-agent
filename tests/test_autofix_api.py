"""auto_fix API 测试：仓库 URL + Issue 描述 → 完整自动修复链路。"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import git
import pytest
from fastapi.testclient import TestClient

from app.agents.fix_agent import FixAgent
from app.agents.test_agent import TestAgent
from app.api.deps import get_fix_agent, get_test_agent
from app.core.config import get_settings
from app.main import create_app
from app.services.repo.git_service import RepoRef
from tests.conftest import FakeLLMClient

BUGGY = "def add(a, b):\n    return a - b\n"
FIXED = "def add(a, b):\n    return a + b\n"
REPO_URL = "https://github.com/local/buggy.git"

GENERATED_TEST = '''```python
from calc import add


def test_add_sums_two_numbers():
    assert add(1, 2) == 3
```
'''

FIX_PATCH = """\
## 分析
实现用了减号，测试期望求和。

## 归类
source_bug

## 补丁
```diff
--- a/calc.py
+++ b/calc.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a - b
+    return a + b
```
"""


@pytest.fixture()
def buggy_source(tmp_path: Path) -> Path:
    """含真实 bug 的源仓库（会被"克隆"进沙箱工作区）。"""
    root = tmp_path / "buggy_src"
    root.mkdir()
    (root / "calc.py").write_text(BUGGY, encoding="utf-8")
    repo = git.Repo.init(root, initial_branch="main")
    repo.index.add(["calc.py"])
    actor = git.Actor("T", "t@e.com")
    repo.index.commit("init", author=actor, committer=actor)
    repo.close()
    return root


@pytest.fixture()
def autofix_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """只配置目录与后端，**不替换** URL 解析（保留真实的 URL 校验）。"""
    workspace = tmp_path / "workspaces"
    monkeypatch.setenv("DOCKER__WORKSPACE_DIR", str(workspace))
    monkeypatch.setenv("DOCKER__BACKEND", "local")
    monkeypatch.setenv("DOCKER__INSTALL_DEPENDENCIES", "false")
    monkeypatch.setenv("REPOSITORY__WORKSPACE_DIR", str(tmp_path / "repos"))
    monkeypatch.setenv("RETRIEVAL__INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("RETRIEVAL__EMBEDDER", "hashing")
    monkeypatch.setenv("RETRIEVAL__EMBEDDING_DIM", "256")
    get_settings.cache_clear()
    workspace.mkdir(parents=True, exist_ok=True)
    yield workspace
    get_settings.cache_clear()


@pytest.fixture()
def autofix_env(
    autofix_dirs: Path, monkeypatch: pytest.MonkeyPatch, buggy_source: Path
) -> Iterator[Path]:
    """在 autofix_dirs 基础上把 URL 解析指向本地仓库（离线跑通克隆）。"""

    def fake_parse(url: str, allowed_hosts: object) -> RepoRef:
        return RepoRef(
            host="github.com",
            owner="local",
            name="buggy",
            clone_url=str(buggy_source),
            sanitized_url=REPO_URL,
        )

    monkeypatch.setattr("app.services.repo.service.parse_repo_url", fake_parse)
    yield autofix_dirs


@pytest.fixture()
def autofix_app(fake_agents: FakeLLMClient) -> Iterator[object]:
    """注入了假 LLM 的应用；Test Agent 与 Fix Agent 共用同一个假客户端。"""
    app = create_app()
    settings = get_settings()
    app.dependency_overrides[get_test_agent] = lambda: TestAgent(settings, fake_agents)
    app.dependency_overrides[get_fix_agent] = lambda: FixAgent(settings, fake_agents)
    yield app
    app.dependency_overrides.clear()


@pytest.fixture()
def fake_agents() -> FakeLLMClient:
    """按调用顺序返回：先测试代码，再修复补丁。"""
    return FakeLLMClient([GENERATED_TEST, FIX_PATCH])


# ---------------------------------------------------------------------------
# 主路径
# ---------------------------------------------------------------------------
def test_auto_fix_converges_with_url_and_issue(
    sqlite_path: Path, autofix_env: Path, autofix_app: object
) -> None:
    """仓库 URL + Issue 描述 → 克隆 → 索引 → 定位 → 生成测试 → 修复 → 通过。"""
    with TestClient(autofix_app) as client:  # type: ignore[arg-type]
        response = client.post(
            "/api/v1/agent/auto_fix",
            json={"repository_url": REPO_URL, "issue": "add 两个数结果是错的", "max_attempts": 3},
        )

    assert response.status_code == 200, response.text
    body = response.json()

    assert body["status"] == "passed"
    assert body["success"] is True
    assert body["attempts"] == 2

    # 定位到的目标
    assert body["target"] is not None
    assert body["target"]["qualified_name"] == "add"
    assert body["target"]["path"] == "calc.py"

    # 最终测试结果
    assert body["failed"] == 0
    assert body["passed"] >= 1
    assert body["coverage_percent"] is not None

    # 修复产出的 diff
    assert "calc.py" in body["changed_files"]
    assert "+    return a + b" in body["final_diff"]
    assert body["worktree"]

    # 每轮的归类与补丁信息都应可追溯
    assert body["iterations"][0]["category"] == "source_bug"
    assert body["iterations"][0]["patch_applied"] is True
    assert body["iterations"][0]["patch_files"] == ["calc.py"]
    assert body["trace_id"]


def test_auto_fix_via_repository_id_and_symbol(
    sqlite_path: Path, autofix_env: Path, autofix_app: object
) -> None:
    """也可用已注册仓库 + 指定符号，跳过 Issue 检索。"""
    with TestClient(autofix_app) as client:  # type: ignore[arg-type]
        created = client.post("/api/v1/repositories", json={"url": REPO_URL})
        assert created.status_code == 201, created.text
        repo_id = created.json()["repository"]["id"]

        structure = client.get(f"/api/v1/repositories/{repo_id}/structure").json()
        symbol_id = structure["files"][0]["symbols"][0]["id"]

        response = client.post(
            "/api/v1/agent/auto_fix",
            json={"repository_id": repo_id, "symbol_id": symbol_id, "max_attempts": 2},
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "passed"
    assert body["target"]["qualified_name"] == "add"


def test_original_clone_is_untouched(
    sqlite_path: Path, autofix_env: Path, autofix_app: object
) -> None:
    """安全属性：修复后原始克隆仍应是 buggy 版本。"""
    with TestClient(autofix_app) as client:  # type: ignore[arg-type]
        created = client.post("/api/v1/repositories", json={"url": REPO_URL})
        repo_id = created.json()["repository"]["id"]
        clone_path = Path(created.json()["repository"]["local_path"])

        response = client.post(
            "/api/v1/agent/auto_fix",
            json={"repository_id": repo_id, "issue": "add 算错了", "max_attempts": 3},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "passed"
    # 克隆仓库必须保持 buggy 状态
    assert (clone_path / "calc.py").read_text(encoding="utf-8") == BUGGY


def test_worktree_retains_fix_and_test(
    sqlite_path: Path, autofix_env: Path, autofix_app: object
) -> None:
    with TestClient(autofix_app) as client:  # type: ignore[arg-type]
        response = client.post(
            "/api/v1/agent/auto_fix",
            json={"repository_url": REPO_URL, "issue": "add 算错了"},
        )

    body = response.json()
    worktree = Path(body["worktree"])
    assert (worktree / "calc.py").read_text(encoding="utf-8") == FIXED
    assert (worktree / "agent_tests" / "test_generated.py").exists()
    assert body["generated_test"]


# ---------------------------------------------------------------------------
# 终止与错误
# ---------------------------------------------------------------------------
def test_max_attempts_reported(
    sqlite_path: Path, autofix_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """补丁改了失败特征但没修好，跑满轮次后应如实报告未成功。"""
    wrong_patch = FIX_PATCH.replace("+    return a + b", "+    return a * b")
    fake = FakeLLMClient([GENERATED_TEST, wrong_patch])

    app = create_app()
    settings = get_settings()
    app.dependency_overrides[get_test_agent] = lambda: TestAgent(settings, fake)
    app.dependency_overrides[get_fix_agent] = lambda: FixAgent(settings, fake)
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/agent/auto_fix",
                json={"repository_url": REPO_URL, "issue": "add 算错了", "max_attempts": 2},
            )
    finally:
        app.dependency_overrides.clear()

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "max_attempts"
    assert body["success"] is False
    assert body["attempts"] == 2


def test_environment_failure_reported(
    sqlite_path: Path, autofix_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_reply = "## 分析\n缺依赖\n\n## 归类\nenvironment\n\n## 补丁\n无需补丁\n"
    fake = FakeLLMClient([GENERATED_TEST, env_reply])

    app = create_app()
    settings = get_settings()
    app.dependency_overrides[get_test_agent] = lambda: TestAgent(settings, fake)
    app.dependency_overrides[get_fix_agent] = lambda: FixAgent(settings, fake)
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/agent/auto_fix",
                json={"repository_url": REPO_URL, "issue": "add 算错了"},
            )
    finally:
        app.dependency_overrides.clear()

    body = response.json()
    assert body["status"] == "not_patchable"
    assert body["success"] is False
    assert "environment" in body["message"]


def test_missing_input_returns_422(
    sqlite_path: Path, autofix_env: Path, autofix_app: object
) -> None:
    with TestClient(autofix_app) as client:  # type: ignore[arg-type]
        no_repo = client.post("/api/v1/agent/auto_fix", json={"issue": "x"})
        no_issue = client.post(
            "/api/v1/agent/auto_fix", json={"repository_url": REPO_URL}
        )

    assert no_repo.status_code == 422
    assert no_issue.status_code == 422


def test_unknown_repository_returns_404(
    sqlite_path: Path, autofix_env: Path, autofix_app: object
) -> None:
    with TestClient(autofix_app) as client:  # type: ignore[arg-type]
        response = client.post(
            "/api/v1/agent/auto_fix",
            json={"repository_id": 999999, "issue": "x"},
        )
    assert response.status_code == 404


def test_invalid_repo_url_returns_400(
    sqlite_path: Path, autofix_dirs: Path, autofix_app: object
) -> None:
    """用 autofix_dirs（不替换 URL 解析），确保走的是真实校验逻辑。"""
    with TestClient(autofix_app) as client:  # type: ignore[arg-type]
        response = client.post(
            "/api/v1/agent/auto_fix",
            json={"repository_url": "file:///etc/passwd", "issue": "x"},
        )
    assert response.status_code == 400
    assert "协议" in response.json()["detail"]


def test_unconfigured_llm_returns_503(
    sqlite_path: Path, autofix_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM__API_KEY", "")
    get_settings.cache_clear()
    from app.core.llm_client import set_llm_client

    set_llm_client(None)

    app = create_app()
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/agent/auto_fix",
                json={"repository_url": REPO_URL, "issue": "add 算错了"},
            )
    finally:
        get_settings.cache_clear()

    assert response.status_code == 503
    assert "LLM__API_KEY" in response.json()["detail"]


def test_openapi_exposes_auto_fix(
    sqlite_path: Path, autofix_env: Path, autofix_app: object
) -> None:
    with TestClient(autofix_app) as client:  # type: ignore[arg-type]
        paths = client.get("/openapi.json").json()["paths"]
    assert "/api/v1/agent/auto_fix" in paths
