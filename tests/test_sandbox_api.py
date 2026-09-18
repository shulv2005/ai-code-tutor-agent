"""执行沙箱 API 测试：真实执行、覆盖率返回、路径防护与状态接口。

本机没有 Docker，因此走 local 降级后端；但**路径防护、错误码、结果结构
与真实执行链路都被完整覆盖**，Docker 与 local 共用同一套 API 契约。
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.agents.test_agent import TestAgent
from app.api.deps import get_test_agent
from app.core.config import get_settings
from app.main import create_app
from tests.conftest import FakeLLMClient

SELF_CONTAINED_TEST = '''```python
import pytest


def add(a: int, b: int) -> int:
    """Stand-in implementation."""
    if not isinstance(a, int) or not isinstance(b, int):
        raise TypeError("int required")
    return a + b


def test_add_positive():
    assert add(1, 2) == 3


@pytest.mark.parametrize("a, b, expected", [(0, 0, 0), (-1, 1, 0)])
def test_add_edge_cases(a, b, expected):
    assert add(a, b) == expected


def test_add_rejects_non_int():
    with pytest.raises(TypeError):
        add("1", 2)
```
'''


@pytest.fixture()
def sandbox_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """把沙箱工作区指向临时目录，并强制 local 后端（本机无 Docker）。"""
    workspace = tmp_path / "workspaces"
    monkeypatch.setenv("DOCKER__WORKSPACE_DIR", str(workspace))
    monkeypatch.setenv("DOCKER__BACKEND", "local")
    monkeypatch.setenv("DOCKER__INSTALL_DEPENDENCIES", "false")
    get_settings.cache_clear()
    workspace.mkdir(parents=True, exist_ok=True)
    yield workspace
    get_settings.cache_clear()


def _make_job(workspace: Path, name: str = "job1") -> Path:
    """在沙箱工作区内造一个可执行的测试目录。"""
    job = workspace / name / "sandbox_repo"
    tests = job / "tests"
    tests.mkdir(parents=True)
    (job / "calc.py").write_text(
        "def add(a, b):\n    return a + b\n", encoding="utf-8"
    )
    (tests / "test_calc.py").write_text(
        "from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )
    return job


# ---------------------------------------------------------------------------
# 真实执行
# ---------------------------------------------------------------------------
def test_run_returns_result_and_coverage(
    sqlite_path: Path, sandbox_workspace: Path
) -> None:
    job = _make_job(sandbox_workspace)

    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/sandbox/run",
            json={"workspace": "job1/sandbox_repo", "test_target": "tests"},
        )

    assert response.status_code == 200, response.text
    body = response.json()

    assert body["backend"] == "local"
    assert body["isolated"] is False
    assert body["exit_code"] == 0
    assert body["succeeded"] is True
    assert body["passed"] == 1
    assert body["failed"] == 0
    assert body["timed_out"] is False
    assert body["stdout"]
    assert body["duration_ms"] > 0
    assert body["trace_id"]

    # 覆盖率报告
    assert body["coverage"] is not None
    assert body["coverage"]["num_statements"] > 0
    assert 0 <= body["coverage"]["percent_covered"] <= 100

    # 必须显式提示"无隔离"
    assert any("无隔离" in note for note in body["notes"])
    assert job.exists()


def test_run_reports_test_failures(sqlite_path: Path, sandbox_workspace: Path) -> None:
    job = _make_job(sandbox_workspace, "job2")
    (job / "tests" / "test_calc.py").write_text(
        "from calc import add\n\n\ndef test_bad():\n    assert add(1, 2) == 99\n",
        encoding="utf-8",
    )

    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/sandbox/run", json={"workspace": "job2/sandbox_repo"}
        )

    body = response.json()
    assert response.status_code == 200
    assert body["succeeded"] is False
    assert body["exit_code"] != 0
    assert body["failed"] == 1


def test_coverage_can_be_disabled(sqlite_path: Path, sandbox_workspace: Path) -> None:
    _make_job(sandbox_workspace, "job3")

    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/sandbox/run",
            json={"workspace": "job3/sandbox_repo", "coverage_enabled": False},
        )

    body = response.json()
    assert response.status_code == 200
    assert body["coverage"] is None


def test_absolute_workspace_path_is_accepted(
    sqlite_path: Path, sandbox_workspace: Path
) -> None:
    job = _make_job(sandbox_workspace, "job4")

    app = create_app()
    with TestClient(app) as client:
        response = client.post("/api/v1/sandbox/run", json={"workspace": str(job)})

    assert response.status_code == 200


# ---------------------------------------------------------------------------
# 路径防护
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "candidate",
    ["../", "../../", "job1/../..", "../../../../etc", "nonexistent-dir"],
)
def test_path_traversal_is_rejected(
    sqlite_path: Path, sandbox_workspace: Path, candidate: str
) -> None:
    """沙箱接口接收"代码路径"，不校验就等于开放宿主机任意目录读取。"""
    _make_job(sandbox_workspace, "job5")

    app = create_app()
    with TestClient(app) as client:
        response = client.post("/api/v1/sandbox/run", json={"workspace": candidate})

    assert response.status_code == 400
    assert "越界" in response.json()["detail"] or "不存在" in response.json()["detail"]


def test_absolute_path_outside_workspace_is_rejected(
    sqlite_path: Path, sandbox_workspace: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()

    app = create_app()
    with TestClient(app) as client:
        response = client.post("/api/v1/sandbox/run", json={"workspace": str(outside)})

    assert response.status_code == 400
    assert "越界" in response.json()["detail"]


def test_custom_command_is_rejected_on_local_backend(
    sqlite_path: Path, sandbox_workspace: Path
) -> None:
    """本地后端无隔离，接受任意命令等于暴露宿主机 shell。"""
    _make_job(sandbox_workspace, "job6")

    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/sandbox/run",
            json={"workspace": "job6/sandbox_repo", "command": ["rm", "-rf", "/"]},
        )

    assert response.status_code == 400
    assert "自定义命令" in response.json()["detail"]


def test_missing_workspace_field_is_422(sqlite_path: Path, sandbox_workspace: Path) -> None:
    app = create_app()
    with TestClient(app) as client:
        response = client.post("/api/v1/sandbox/run", json={})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 状态接口
# ---------------------------------------------------------------------------
def test_status_reports_no_isolation_without_docker(
    sqlite_path: Path, sandbox_workspace: Path
) -> None:
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/api/v1/sandbox/status")

    assert response.status_code == 200
    body = response.json()
    assert body["docker_available"] is False
    assert body["backend"] == "local"
    assert body["isolated"] is False
    # 必须如实反映安全配置，不能因为跑在 local 上就假装有隔离
    assert body["security"]["cap_drop_all"] is True
    assert body["security"]["docker_socket_mounted"] is False
    assert body["security"]["privileged"] is False
    assert body["limits"]["pids_limit"] > 0


# ---------------------------------------------------------------------------
# 端到端：Step 4 生成 → Step 5 执行
# ---------------------------------------------------------------------------
def test_generated_test_runs_in_sandbox(
    sqlite_path: Path, sandbox_workspace: Path
) -> None:
    """完整闭环：Agent 生成测试写入沙箱 → 沙箱执行 → 返回结果与覆盖率。

    这是 Step 4 与 Step 5 的接口契约验证：生成产物的落盘位置
    必须能直接被沙箱接口使用。
    """
    fake = FakeLLMClient(SELF_CONTAINED_TEST)
    app = create_app()
    app.dependency_overrides[get_test_agent] = lambda: TestAgent(get_settings(), fake)
    try:
        with TestClient(app) as client:
            generated = client.post(
                "/api/v1/agent/generate_test",
                json={"code": "def add(a, b):\n    return a + b\n", "symbol_name": "add"},
            )
            assert generated.status_code == 200, generated.text
            gen_body = generated.json()

            # 沙箱接口直接吃 generate_test 返回的 sandbox_dir
            sandbox_dir = gen_body["sandbox_dir"]
            assert sandbox_dir

            relative = Path(sandbox_dir).relative_to(sandbox_workspace).as_posix()
            executed = client.post(
                "/api/v1/sandbox/run", json={"workspace": relative, "test_target": "tests"}
            )
    finally:
        app.dependency_overrides.clear()

    assert executed.status_code == 200, executed.text
    body = executed.json()

    assert body["succeeded"] is True, body["stdout"] + body["stderr"]
    # 1 + 2(parametrize) + 1 = 4 个用例
    assert body["passed"] == 4
    assert body["failed"] == 0
    assert body["coverage"] is not None
    assert body["coverage"]["percent_covered"] > 0


def test_openapi_exposes_sandbox_routes(
    sqlite_path: Path, sandbox_workspace: Path
) -> None:
    app = create_app()
    with TestClient(app) as client:
        paths = client.get("/openapi.json").json()["paths"]

    assert "/api/v1/sandbox/run" in paths
    assert "/api/v1/sandbox/status" in paths
