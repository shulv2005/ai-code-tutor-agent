"""端到端链路的 pytest 封装。

复用 `tests/e2e_test.py` 里的 run_pipeline()，用夹具构造的客户端跑同一条链路，
使"全链路整合"纳入常规测试套件，而不只是一个人工脚本。

e2e_test.py 本身被 collect_ignore 排除（它需要网络与 LLM），
这里导入它并注入离线可跑的依赖。
"""

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
from app.core.llm_client import LLMMessage, LLMResponse, LLMUsage
from app.main import create_app
from app.services.repo.git_service import RepoRef
from tests.conftest import FakeLLMClient
from tests.e2e_test import run_pipeline

BUGGY = 'def add(a, b):\n    """Add two numbers."""\n    return a - b\n'
FIXED = 'def add(a, b):\n    """Add two numbers."""\n    return a + b\n'
REPO_URL = "https://github.com/e2e/demo.git"

TEST_CODE = (
    "```python\nfrom calc import add\n\n\n"
    "def test_add_sums_two_numbers():\n    assert add(1, 2) == 3\n\n\n"
    "def test_add_with_negatives():\n    assert add(-1, -2) == -3\n```\n"
)

PATCH = (
    "## 分析\n`add` 用的是减号，测试期望求和。\n\n## 归类\nsource_bug\n\n## 补丁\n"
    "```diff\n--- a/calc.py\n+++ b/calc.py\n@@ -1,3 +1,3 @@\n def add(a, b):\n"
    '     """Add two numbers."""\n'
    "-    return a - b\n+    return a + b\n```\n"
)


class RoutingFakeLLM(FakeLLMClient):
    """按提示词内容路由：生成测试 vs 生成补丁。

    顺序式假客户端在多阶段链路里不可靠——前一步消耗掉一个响应后，
    后一步会拿到错误的回复，产生与产品无关的假失败。
    """

    def __init__(self) -> None:
        super().__init__([TEST_CODE])
        self.test_calls = 0
        self.fix_calls = 0

    async def chat(self, messages: list[LLMMessage], **_: object) -> LLMResponse:
        system = next((item.content for item in messages if item.role == "system"), "")
        self.calls.append(list(messages))
        if "归类" in system and "unified diff" in system:
            self.fix_calls += 1
            content = PATCH
        else:
            self.test_calls += 1
            content = TEST_CODE
        return LLMResponse(
            content=content,
            model="fake-model",
            finish_reason="stop",
            usage=LLMUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            latency_ms=1.0,
        )


@pytest.fixture()
def demo_repo(tmp_path: Path) -> Path:
    """含真实 bug 的演示仓库。"""
    root = tmp_path / "demo_src"
    root.mkdir()
    (root / "calc.py").write_text(BUGGY, encoding="utf-8")
    (root / "README.md").write_text("# demo\n", encoding="utf-8")
    repo = git.Repo.init(root, initial_branch="main")
    repo.index.add(["calc.py", "README.md"])
    actor = git.Actor("E2E", "e2e@example.com")
    repo.index.commit("init", author=actor, committer=actor)
    repo.close()
    return root


@pytest.fixture()
def e2e_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    demo_repo: Path,
    sqlite_path: Path,
) -> Iterator[tuple[TestClient, FakeLLMClient]]:
    """离线可跑的端到端客户端：本地仓库 + 假 LLM，其余全真实。

    必须依赖 sqlite_path 夹具：否则会用默认的 data/app.db，
    而它一旦残留了同 URL 的历史仓库记录，register_and_index 会命中缓存直接返回，
    reindex 又会因 local_path 已失效而返回 409——失败原因与本次改动完全无关。
    """
    monkeypatch.setenv("DOCKER__WORKSPACE_DIR", str(tmp_path / "workspaces"))
    monkeypatch.setenv("DOCKER__BACKEND", "local")
    monkeypatch.setenv("DOCKER__INSTALL_DEPENDENCIES", "false")
    monkeypatch.setenv("REPOSITORY__WORKSPACE_DIR", str(tmp_path / "repos"))
    monkeypatch.setenv("RETRIEVAL__INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("RETRIEVAL__EMBEDDER", "hashing")
    monkeypatch.setenv("RETRIEVAL__EMBEDDING_DIM", "256")
    get_settings.cache_clear()

    def fake_parse(url: str, allowed_hosts: object) -> RepoRef:
        return RepoRef(
            host="github.com", owner="e2e", name="demo",
            clone_url=str(demo_repo), sanitized_url=REPO_URL,
        )

    monkeypatch.setattr("app.services.repo.service.parse_repo_url", fake_parse)

    fake = RoutingFakeLLM()
    settings = get_settings()
    app = create_app()
    app.dependency_overrides[get_test_agent] = lambda: TestAgent(settings, fake)
    app.dependency_overrides[get_fix_agent] = lambda: FixAgent(settings, fake)

    with TestClient(app) as client:
        yield client, fake
    app.dependency_overrides.clear()
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# 全链路
# ---------------------------------------------------------------------------
def test_full_pipeline_succeeds(e2e_client: tuple[TestClient, FakeLLMClient]) -> None:
    """克隆 → 解析 → 检索 → 生成 → 执行 → 修复 → 重跑 → PR 草稿 全链路通过。"""
    client, _ = e2e_client
    report = run_pipeline(
        client,
        repo_url=REPO_URL,
        issue="add 两个数相加结果是错的",
        log=lambda _message: None,  # 测试里静默
    )

    assert report.success, report.to_dict()
    failed = [item.name for item in report.steps if not item.ok]
    assert not failed, f"失败步骤: {failed}"
    assert len(report.steps) == 11


def test_pipeline_reaches_pr_draft(e2e_client: tuple[TestClient, FakeLLMClient]) -> None:
    client, _ = e2e_client
    report = run_pipeline(
        client, repo_url=REPO_URL, issue="add 两个数相加结果是错的",
        log=lambda _message: None,
    )

    fix_step = next(item for item in report.steps if "PR 草稿" in item.name or "auto_fix" in item.name)
    assert fix_step.payload.get("fix"), "未取得 auto_fix 结果"

    # 步骤 8 的 payload 里带 PR 草稿
    draft_step = next(item for item in report.steps if item.name == "生成 PR 草稿")
    draft = draft_step.payload["draft"]
    assert draft["verified"] is True
    assert draft["is_draft"] is True
    assert "calc.py" in draft["files_changed"]
    assert draft["title"].startswith("fix:")
    assert draft["branch_name"].startswith("fix/agent-")


def test_pipeline_report_is_json_serializable(
    e2e_client: tuple[TestClient, FakeLLMClient]
) -> None:
    """报告结构必须可序列化（供 CI 与前端消费）。"""
    import json

    client, _ = e2e_client
    report = run_pipeline(
        client, repo_url=REPO_URL, issue="add 两个数相加结果是错的",
        log=lambda _message: None,
    )
    payload = json.dumps(report.to_dict(), ensure_ascii=False)
    assert "steps" in payload
    assert len(json.loads(payload)["steps"]) == len(report.steps)


def test_llm_was_used_for_both_generation_and_fix(
    e2e_client: tuple[TestClient, FakeLLMClient]
) -> None:
    """确认链路真的走完了"生成测试"和"生成补丁"两条 LLM 路径。"""
    client, fake = e2e_client
    run_pipeline(
        client, repo_url=REPO_URL, issue="add 两个数相加结果是错的",
        log=lambda _message: None,
    )
    assert fake.test_calls >= 1, "未调用测试生成"
    assert fake.fix_calls >= 1, "未调用修复 Agent"


def test_pipeline_leaves_source_clone_untouched(
    e2e_client: tuple[TestClient, FakeLLMClient], demo_repo: Path
) -> None:
    """整合后的安全属性依然成立：原始仓库不被修改。"""
    client, _ = e2e_client
    run_pipeline(
        client, repo_url=REPO_URL, issue="add 两个数相加结果是错的",
        log=lambda _message: None,
    )
    assert (demo_repo / "calc.py").read_text(encoding="utf-8") == BUGGY


# ---------------------------------------------------------------------------
# 步骤 A：各接口可独立调用
# ---------------------------------------------------------------------------
def test_each_route_is_independently_callable(
    e2e_client: tuple[TestClient, FakeLLMClient]
) -> None:
    """逐个调用核心接口，确认"统一调用"没有隐含的顺序依赖。"""
    client, _ = e2e_client

    assert client.get("/api/v1/health").status_code == 200
    assert client.get("/api/v1/health/ready").status_code == 200
    assert client.get("/api/v1/agent/status").status_code == 200
    assert client.get("/api/v1/sandbox/status").status_code == 200
    assert client.get("/api/v1/repositories/languages").status_code == 200

    created = client.post("/api/v1/repositories", json={"url": REPO_URL})
    assert created.status_code == 201
    repo_id = created.json()["repository"]["id"]

    assert client.get(f"/api/v1/repositories/{repo_id}").status_code == 200
    assert client.get(f"/api/v1/repositories/{repo_id}/structure").status_code == 200
    assert client.get("/api/v1/repositories").status_code == 200
    assert client.post(f"/api/v1/repositories/{repo_id}/index").status_code == 200
    assert client.post(f"/api/v1/repositories/{repo_id}/reindex").status_code == 200

    searched = client.post(
        "/api/v1/search", json={"repository_id": repo_id, "query": "add", "top_k": 3}
    )
    assert searched.status_code == 200
    assert searched.json()["hits"]

    generated = client.post(
        "/api/v1/agent/generate_test",
        json={"repository_id": repo_id, "query": "add"},
    )
    assert generated.status_code == 200
    # 生成的沙箱目录必须自带仓库副本，才能被 sandbox/run 直接执行
    sandbox_dir = Path(generated.json()["sandbox_dir"])
    assert (sandbox_dir / "calc.py").exists(), "沙箱目录缺少仓库副本，测试将无法导入模块"

    ran = client.post(
        "/api/v1/sandbox/run", json={"workspace": str(sandbox_dir), "test_target": "tests"}
    )
    assert ran.status_code == 200
    body = ran.json()
    # 测试应真正执行（而不是 collection error）
    assert body["exit_code"] != 2
    assert body["passed"] + body["failed"] + body["errors"] > 0
