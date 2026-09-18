"""端到端集成测试：串联「克隆 → 解析 → 检索 → 生成测试 → 沙箱执行 → 修复 → 重跑 → PR 草稿」。

两种运行方式：

1. 命令行脚本（默认离线演示模式，无需网络与 API Key）：
       python tests/e2e_test.py --offline
       python tests/e2e_test.py --repo https://gitee.com/mirrors/requests.git \\
                               --issue "取消订单时没有校验状态"
   真实模式需要：可访问的仓库、已配置的 LLM（LLM__API_KEY）、可用的 Docker
   （否则沙箱走本地降级后端，会明确提示无隔离）。

2. pytest（见 tests/test_e2e_pipeline.py）：调用本模块的 run_pipeline()，
   用夹具构造的客户端跑同一条链路，纳入常规测试套件。

设计说明：本脚本**分两段**验证——
   A 段逐个调用各模块接口，证明"每个路由都能独立工作"；
   B 段调用 auto_fix 串起全链路，证明"它们组合起来也能工作"。
两段都通过，才说明整合没有问题。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 直接以脚本方式运行（python tests/e2e_test.py）时，sys.path[0] 是 tests/ 而不是
# 项目根目录，必须先把它补上，后续 `import app` 才能成功。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# 结果结构
# ---------------------------------------------------------------------------
@dataclass
class StepResult:
    """单个步骤的结果。"""

    name: str
    ok: bool
    detail: str = ""
    duration_ms: float = 0.0
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineReport:
    """整条链路的报告。"""

    steps: list[StepResult] = field(default_factory=list)
    success: bool = False
    summary: str = ""

    def add(self, step: StepResult) -> StepResult:
        self.steps.append(step)
        return step

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "summary": self.summary,
            "steps": [
                {
                    "name": item.name,
                    "ok": item.ok,
                    "detail": item.detail,
                    "duration_ms": round(item.duration_ms, 1),
                }
                for item in self.steps
            ],
        }


Logger = Callable[[str], None]


def _default_log(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------------------
# 流程实现
# ---------------------------------------------------------------------------
def run_pipeline(
    client: Any,
    *,
    repo_url: str,
    issue: str,
    max_attempts: int = 3,
    log: Logger = _default_log,
    query: str | None = None,
) -> PipelineReport:
    """执行完整链路。

    Args:
        client: 已启动的 TestClient（调用方负责生命周期与依赖注入）。
        repo_url: 仓库地址。
        issue: Issue 描述。
        max_attempts: 自动修复的最大轮次。
        log: 日志输出函数。
        query: 单独验证检索用的查询（缺省用 issue）。
    """
    report = PipelineReport()
    query = query or issue

    def step(name: str) -> tuple[float, Callable[..., StepResult]]:
        """开一个计时步骤。"""
        started = time.perf_counter()

        def finish(ok: bool, detail: str = "", **payload: Any) -> StepResult:
            result = StepResult(
                name=name,
                ok=ok,
                detail=detail,
                duration_ms=(time.perf_counter() - started) * 1000,
                payload=payload,
            )
            report.add(result)
            mark = "OK  " if ok else "FAIL"
            log(f"  [{mark}] {name}（{result.duration_ms:.0f}ms）")
            if detail:
                for line in detail.splitlines():
                    log(f"         {line}")
            return result

        return started, finish

    log("=" * 78)
    log(f"端到端链路  仓库={repo_url}")
    log(f"            Issue={issue}")
    log("=" * 78)

    # --- 0. 服务与依赖状态 ---------------------------------------------
    log("\n[0] 服务与依赖状态")
    _, done = step("健康检查")
    response = client.get("/api/v1/health")
    if response.status_code != 200:
        done(False, f"HTTP {response.status_code}")
        report.summary = "服务不可用"
        return report
    done(True, f"service={response.json()['service']} trace_id={response.json()['trace_id'][:12]}...")

    _, done = step("LLM 配置")
    llm = client.get("/api/v1/agent/status").json()
    done(
        True,
        f"configured={llm['configured']} model={llm['model']}",
    )
    if not llm["configured"]:
        log("         [!] LLM 未配置，生成/修复步骤将不可用（真实模式需设置 LLM__API_KEY）")

    _, done = step("沙箱可用性")
    sandbox = client.get("/api/v1/sandbox/status").json()
    done(
        True,
        f"backend={sandbox['backend']} isolated={sandbox['isolated']} "
        f"docker_available={sandbox['docker_available']}",
    )
    if not sandbox["isolated"]:
        log("         [!] 当前沙箱无隔离能力（Docker 不可用），仅适合执行自己生成的代码")

    # --- 1. 克隆 + 解析 --------------------------------------------------
    log("\n[1] 克隆与解析（POST /api/v1/repositories）")
    _, done = step("注册并索引仓库")
    response = client.post("/api/v1/repositories", json={"url": repo_url})
    if response.status_code != 201:
        done(False, f"HTTP {response.status_code}: {response.text[:200]}")
        report.summary = "克隆/解析失败"
        return report
    body = response.json()
    repo = body["repository"]
    stats = body["stats"]
    repo_id = repo["id"]
    done(
        True,
        f"仓库ID={repo_id} 状态={repo['status']} 分支={repo['default_branch']}\n"
        f"文件={stats['file_count']} 符号={stats['symbol_count']} "
        f"解析失败={stats['failed_files']} 体积={repo['size_bytes'] / 1024 / 1024:.1f}MB",
        repository_id=repo_id,
    )
    if repo["status"] != "ready":
        report.summary = f"仓库状态异常: {repo['status']}"
        return report

    # --- 2. 代码结构 -----------------------------------------------------
    log("\n[2] 代码结构（GET /api/v1/repositories/{id}/structure）")
    _, done = step("读取代码结构")
    structure = client.get(
        f"/api/v1/repositories/{repo_id}/structure", params={"limit": 5}
    ).json()
    done(
        True,
        f"文件总数={structure['total_files']} 符号总数={structure['total_symbols']}\n"
        f"语言分布={structure['languages']}",
    )

    # --- 3. 检索索引 -----------------------------------------------------
    log("\n[3] 构建检索索引（POST /api/v1/repositories/{id}/index）")
    _, done = step("构建 BM25 + FAISS 索引")
    response = client.post(f"/api/v1/repositories/{repo_id}/index")
    if response.status_code != 200:
        done(False, f"HTTP {response.status_code}: {response.text[:200]}")
        report.summary = "检索索引构建失败"
        return report
    index_info = response.json()
    done(
        True,
        f"分块={index_info['chunk_count']} 维度={index_info['dimension']} "
        f"嵌入后端={index_info['embedder']} 耗时={index_info['duration_ms']}ms",
    )

    # --- 4. 混合检索 -----------------------------------------------------
    log("\n[4] 混合检索定位代码（POST /api/v1/search）")
    _, done = step(f"检索「{query}」")
    response = client.post(
        "/api/v1/search", json={"repository_id": repo_id, "query": query, "top_k": 3}
    )
    if response.status_code != 200:
        done(False, f"HTTP {response.status_code}: {response.text[:200]}")
        report.summary = "检索失败"
        return report
    search = response.json()
    hits = search["hits"]
    if not hits:
        done(False, "未检索到任何代码符号")
        report.summary = "检索无命中"
        return report
    lines = [
        f"#{rank} {h['qualified_name']}  {h['path']}  L{h['start_line']}-{h['end_line']}  "
        f"bm25#{h['bm25_rank']} vec#{h['vector_rank']}"
        for rank, h in enumerate(hits, start=1)
    ]
    # 定位到的目标符号：供后续步骤在无 LLM 时也能继续验证
    if len(query.strip()) <= 40 and query.strip().isidentifier():
        target_symbol_id = hits[0]["symbol_id"]
    else:
        target_symbol_id = None
    done(True, "\n".join(lines), symbol_id=target_symbol_id, search=search)

    # --- 5. 测试生成 -----------------------------------------------------
    log("\n[5] 生成测试（POST /api/v1/agent/generate_test）")
    _, done = step("生成 pytest 测试")
    if not llm["configured"]:
        # 注意：step() 必须先于这个分支调用，否则会复用上一步的 done 闭包，
        # 导致步骤被重复计数且标签错位。
        done(False, "LLM 未配置，跳过")
        log("\n  链路在“测试生成”处中断：需要先配置 LLM__API_KEY 才能继续。")
        log("  已完成的验证：克隆、解析、索引构建、混合检索（这四步不依赖 LLM）。")
        report.summary = "LLM 未配置，链路未跑完"
        return report

    payload: dict[str, Any] = {"repository_id": repo_id, "query": query, "max_attempts": 2}
    response = client.post("/api/v1/agent/generate_test", json=payload)
    if response.status_code != 200:
        done(False, f"HTTP {response.status_code}: {response.text[:300]}")
        report.summary = "测试生成失败"
        return report
    generated = response.json()
    done(
        True,
        f"run_id={generated['run_id']} 模型={generated['model']} "
        f"用例={generated['test_functions']}\n落盘={generated['saved_path']}",
        generated=generated,
    )

    # --- 6. 沙箱执行 -----------------------------------------------------
    log("\n[6] 沙箱执行（POST /api/v1/sandbox/run）")
    _, done = step("执行生成的测试")
    sandbox_dir = generated.get("sandbox_dir")
    if not sandbox_dir:
        done(False, "未返回 sandbox_dir（可能未落盘）")
    else:
        response = client.post(
            "/api/v1/sandbox/run", json={"workspace": sandbox_dir, "test_target": "tests"}
        )
        if response.status_code != 200:
            done(False, f"HTTP {response.status_code}: {response.text[:300]}")
        else:
            run = response.json()
            executed = run["passed"] + run["failed"] + run["errors"]
            # 判定标准是"测试真的跑起来了"，而不是"测试通过"——
            # 被测代码本来就带 bug，用例失败才是预期结果。
            # 真正的失败信号是：collection error（exit=2）或一个用例都没收集到。
            collection_error = run["exit_code"] == 2
            ok = not collection_error and executed > 0
            detail = (
                f"backend={run['backend']} isolated={run['isolated']} "
                f"exit={run['exit_code']} 通过={run['passed']} 失败={run['failed']} "
                f"错误={run['errors']}\n"
                f"覆盖率={run['coverage']['percent_covered'] if run['coverage'] else 'N/A'}%"
            )
            if collection_error:
                detail += "\n[!] pytest 收集阶段就报错（exit=2），通常是测试无法导入被测模块"
            elif executed > 0 and run["failed"]:
                detail += "\n[i] 用例失败符合预期：被测代码本身带 bug，下一步交给修复循环"
            done(ok, detail, run=run)

    # --- 7. 自动修复全链路 -----------------------------------------------
    log("\n[7] 自动修复全链路（POST /api/v1/agent/auto_fix）")
    _, done = step("克隆→检索→生成→执行→修复→重跑→PR 草稿")
    response = client.post(
        "/api/v1/agent/auto_fix",
        json={"repository_id": repo_id, "issue": issue, "max_attempts": max_attempts},
    )
    if response.status_code != 200:
        done(False, f"HTTP {response.status_code}: {response.text[:400]}")
        report.summary = "自动修复接口调用失败"
        return report

    fix = response.json()
    lines = [
        f"状态={fix['status']} success={fix['success']} 轮次={fix['attempts']}",
        f"目标={fix['target']['qualified_name'] if fix['target'] else 'N/A'}",
        f"最终用例：通过={fix['passed']} 失败={fix['failed']} 错误={fix['errors']} "
        f"覆盖率={fix['coverage_percent']}%",
    ]
    for item in fix["iterations"]:
        lines.append(
            f"第{item['index']}轮 exit={item['exit_code']} "
            f"通过={item['passed']} 失败={item['failed']} "
            f"归类={item['category']} 补丁={'已应用' if item['patch_applied'] else '未应用'}"
        )
    done(fix["success"], "\n".join(lines), fix=fix)

    # --- 8. PR 草稿 ------------------------------------------------------
    log("\n[8] PR 草稿（包含在 auto_fix 响应中）")
    _, done = step("生成 PR 草稿")
    draft = fix.get("pr_draft")
    if not draft:
        done(False, "响应中缺少 pr_draft 字段")
    else:
        done(
            True,
            f"标题：{draft['title']}\n"
            f"分支：{draft['branch_name']}（base={draft['base_branch']}）\n"
            f"改动：{draft['files_changed']} +{draft['additions']}/-{draft['deletions']}\n"
            f"verified={draft['verified']} is_draft={draft['is_draft']}",
            draft=draft,
        )
        if draft["warnings"]:
            for warning in draft["warnings"]:
                log(f"         [!] {warning}")

    # --- 汇总 ------------------------------------------------------------
    passed_steps = sum(1 for item in report.steps if item.ok)
    failed_steps = [item for item in report.steps if not item.ok]
    report.success = not failed_steps and bool(fix.get("success"))

    log("\n" + "=" * 78)
    log(f"汇总：{passed_steps}/{len(report.steps)} 步通过")
    if failed_steps:
        for item in failed_steps:
            log(f"  失败：{item.name} — {item.detail[:120]}")
    log(f"自动修复是否收敛：{fix.get('success')}")
    if fix.get("success"):
        log("全链路通过：克隆 → 解析 → 检索 → 生成 → 执行 → 修复 → 重跑 → PR 草稿")
    else:
        log(f"链路未收敛（status={fix.get('status')}）：{fix.get('message')}")
    log("=" * 78)

    report.summary = (
        "全链路通过" if report.success else f"未完全通过（{len(failed_steps)} 步失败）"
    )
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_offline_client(workspace_root: Path) -> tuple[Any, str, str]:
    """离线演示模式：构造一个带真实 bug 的本地仓库 + 假 LLM。

    返回 (client, repo_url, issue)。除了 LLM 与"远端"仓库是本地的，
    其余全部走真实代码：真实 git 克隆、真实解析、真实检索、真实 pytest 执行、
    真实补丁应用。
    """
    import os
    import tempfile

    work = Path(tempfile.mkdtemp(prefix="e2e-"))
    buggy = work / "buggy_src"
    buggy.mkdir(parents=True)
    (buggy / "calc.py").write_text(
        'def add(a, b):\n    """Add two numbers."""\n    return a - b\n',
        encoding="utf-8",
    )
    (buggy / "README.md").write_text("# demo\n", encoding="utf-8")

    import git

    repo = git.Repo.init(buggy, initial_branch="main")
    repo.index.add(["calc.py", "README.md"])
    actor = git.Actor("E2E", "e2e@example.com")
    repo.index.commit("init", author=actor, committer=actor)
    repo.close()

    os.environ.update(
        {
            "DATABASE__SQLITE_PATH": str(work / "e2e.db"),
            "REPOSITORY__WORKSPACE_DIR": str(work / "repos"),
            "RETRIEVAL__INDEX_DIR": str(work / "index"),
            "RETRIEVAL__EMBEDDER": "hashing",
            "RETRIEVAL__EMBEDDING_DIM": "256",
            "DOCKER__WORKSPACE_DIR": str(work / "workspaces"),
            "DOCKER__BACKEND": "local",
            "DOCKER__INSTALL_DEPENDENCIES": "false",
            "APP__LOG_LEVEL": "WARNING",
        }
    )

    from app.core.config import get_settings

    get_settings.cache_clear()

    # 把 URL 解析指向本地仓库
    import app.services.repo.service as repo_service
    from app.services.repo.git_service import RepoRef

    url = "https://github.com/e2e/demo.git"

    def fake_parse(u: str, allowed_hosts: object) -> RepoRef:
        return RepoRef(
            host="github.com", owner="e2e", name="demo",
            clone_url=str(buggy), sanitized_url=url,
        )

    repo_service.parse_repo_url = fake_parse  # type: ignore[assignment]

    # 假 LLM：按提示词**内容**路由，而不是按调用顺序。
    # 顺序式假客户端在多阶段链路里很脆弱——第 5 步消耗掉一个响应后，
    # 第 7 步就会拿到"补丁"当"测试代码"，产生与产品无关的假失败。
    from app.core.llm_client import LLMMessage, LLMResponse, LLMUsage, set_llm_client
    from tests.conftest import FakeLLMClient

    test_code = (
        "```python\nfrom calc import add\n\n\n"
        "def test_add_sums_two_numbers():\n    assert add(1, 2) == 3\n\n\n"
        "def test_add_with_negatives():\n    assert add(-1, -2) == -3\n```\n"
    )
    patch = (
        "## 分析\n`add` 用的是减号，测试期望求和，属于源码实现错误。\n\n"
        "## 归类\nsource_bug\n\n## 补丁\n```diff\n--- a/calc.py\n+++ b/calc.py\n"
        "@@ -1,3 +1,3 @@\n def add(a, b):\n"
        '     """Add two numbers."""\n'
        "-    return a - b\n+    return a + b\n```\n"
    )

    class RoutingFakeLLM(FakeLLMClient):
        """根据系统提示词判断当前是生成测试还是生成补丁。"""

        def __init__(self, test_reply: str, patch_reply: str) -> None:
            super().__init__([test_reply])
            self._test_reply = test_reply
            self._patch_reply = patch_reply
            self.fix_calls = 0
            self.test_calls = 0

        async def chat(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
            system = next((m.content for m in messages if m.role == "system"), "")
            is_fix = "归类" in system and "unified diff" in system
            self.calls.append(list(messages))
            if is_fix:
                self.fix_calls += 1
                content = self._patch_reply
            else:
                self.test_calls += 1
                content = self._test_reply
            return LLMResponse(
                content=content,
                model="fake-model",
                finish_reason="stop",
                usage=LLMUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
                latency_ms=1.0,
            )

    fake = RoutingFakeLLM(test_code, patch)
    set_llm_client(fake)

    from fastapi.testclient import TestClient

    from app.agents.fix_agent import FixAgent
    from app.agents.test_agent import TestAgent
    from app.api.deps import get_fix_agent, get_test_agent
    from app.main import create_app

    settings = get_settings()
    app = create_app()
    app.dependency_overrides[get_test_agent] = lambda: TestAgent(settings, fake)
    app.dependency_overrides[get_fix_agent] = lambda: FixAgent(settings, fake)

    client = TestClient(app)
    client.__enter__()  # 手动进入上下文，触发 lifespan 建表
    return client, url, "add 两个数相加结果是错的"


def _build_real_client() -> tuple[Any, None, None]:
    """真实模式：直接用配置好的应用（需要 .env 里的真实配置）。"""
    from fastapi.testclient import TestClient

    from app.main import create_app

    client = TestClient(create_app())
    client.__enter__()
    return client, None, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="端到端集成测试")
    parser.add_argument("--repo", help="仓库 URL（真实模式必填）")
    parser.add_argument("--issue", help="Issue 描述（真实模式必填）")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="离线演示模式：用本地带 bug 的仓库 + 假 LLM，验证链路本身",
    )
    parser.add_argument("--query", help="单独验证检索用的查询（缺省用 issue）")
    parser.add_argument("--max-attempts", type=int, default=3, help="最大修复轮次")
    parser.add_argument("--json", action="store_true", help="额外输出机器可读的 JSON 报告")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="同时输出应用自身的日志（默认只输出步骤日志，避免刷屏）",
    )
    args = parser.parse_args(argv)

    # 默认静音应用日志：Trace/httpx 的 INFO 会把步骤输出冲散，
    # 而本脚本要呈现的正是逐步的执行过程。
    if not args.verbose:
        import logging

        logging.getLogger().setLevel(logging.WARNING)
        for name in ("httpx", "httpcore", "app.trace", "app.services"):
            logging.getLogger(name).setLevel(logging.WARNING)

    if args.offline:
        client, url, issue = _build_offline_client(Path.cwd())
        print("[模式] 离线演示（本地仓库 + 假 LLM，其余全真实）\n")
    else:
        if not args.repo or not args.issue:
            parser.error("真实模式需要 --repo 与 --issue（或使用 --offline）")
        client, _, _ = _build_real_client()
        url, issue = args.repo, args.issue
        print("[模式] 真实仓库\n")

    try:
        report = run_pipeline(
            client,
            repo_url=url,
            issue=issue,
            query=args.query,
            max_attempts=args.max_attempts,
        )
    finally:
        try:
            client.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass

    if args.json:
        print("\n--- JSON 报告 ---")
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))

    return 0 if report.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
