"""API 契约测试：确保所有路由能被统一调用、命名一致、响应契约完整。

这是"整合"的静态保障层——不需要起服务，直接对 OpenAPI schema 做校验，
能在重构时第一时间发现路由缺失、字段改名、响应契约漂移。
"""

from __future__ import annotations

from collections import defaultdict

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

# 期望的完整路由集：任何一条消失都会让链路断掉
EXPECTED_ROUTES = {
    "/api/v1/health": {"get"},
    "/api/v1/health/ready": {"get"},
    "/api/v1/repositories/languages": {"get"},
    "/api/v1/repositories": {"get", "post"},
    "/api/v1/repositories/{repository_id}": {"get"},
    "/api/v1/repositories/{repository_id}/structure": {"get"},
    "/api/v1/repositories/{repository_id}/reindex": {"post"},
    "/api/v1/repositories/{repository_id}/index": {"post"},
    "/api/v1/search": {"post"},
    "/api/v1/agent/status": {"get"},
    "/api/v1/agent/generate_test": {"post"},
    "/api/v1/agent/auto_fix": {"post"},
    "/api/v1/sandbox/status": {"get"},
    "/api/v1/sandbox/run": {"post"},
    # 学生代码导师场景（本地化 AI 代码导师系统）
    "/api/v1/tutor/analyze": {"post"},
    "/api/v1/tutor/check": {"post"},
    "/api/v1/tutor/comment": {"post"},
    "/api/v1/tutor/fix": {"post"},
    "/api/v1/tutor/history": {"get"},
    "/api/v1/tutor/history/{record_id}": {"get", "delete"},
    "/api/v1/tutor/status": {"get"},
    # 本地项目库（按语言自动分类学生本地代码）
    "/api/v1/library/scan": {"get"},
    # file 一条路径三个方法：读 / 在线编辑保存（替换共用）/ 删除（默认进回收站）
    "/api/v1/library/file": {"get", "put", "delete"},
    "/api/v1/library/status": {"get"},
    # 本地代码文件自动分类（扫描并归档到 c/ java/ python/）
    "/api/v1/files/scan": {"post"},
    "/api/v1/files/list": {"get"},
    # 前端拖拽入库：把文件存进项目库，再交给 /files/scan 归档
    "/api/v1/files/upload": {"post"},
    # AI 自动检测（本地静态检查 + AI 深度检测）
    "/api/v1/check/code": {"post"},
    # 代码改错（本地分析 → AI 修正 → 本地复检）
    "/api/v1/fix/code": {"post"},
    "/api/v1/fix/history": {"get"},
    "/api/v1/fix/history/{record_id}": {"get"},
    # 代码注释生成（三层注释 + 代码未被改动的复检）
    "/api/v1/comment/generate": {"post"},
    "/api/v1/comment/history": {"get"},
    "/api/v1/comment/history/{record_id}": {"get"},
    # 多模型与网页填 Key：Key 只存内存，不落库、不进日志
    "/api/v1/models/list": {"get"},
    "/api/v1/auth/set_key": {"post"},
    "/api/v1/auth/clear_key": {"post"},
    "/api/v1/auth/status": {"get"},
}

HTTP_METHODS = {"get", "post", "put", "patch", "delete"}


@pytest.fixture(scope="module")
def spec() -> dict:
    return create_app().openapi()


def _operations(spec: dict) -> list[tuple[str, str, dict]]:
    return [
        (method.upper(), path, op)
        for path, operations in spec["paths"].items()
        for method, op in operations.items()
        if method.lower() in HTTP_METHODS
    ]


# ---------------------------------------------------------------------------
# 路由完整性
# ---------------------------------------------------------------------------
def test_all_expected_routes_exist(spec: dict) -> None:
    actual = set(spec["paths"])
    missing = set(EXPECTED_ROUTES) - actual
    assert not missing, f"缺失路由: {sorted(missing)}"


@pytest.mark.parametrize(("path", "methods"), sorted(EXPECTED_ROUTES.items()))
def test_route_exposes_expected_methods(spec: dict, path: str, methods: set[str]) -> None:
    operations = spec["paths"][path]
    actual = {method for method in operations if method.lower() in HTTP_METHODS}
    assert methods <= actual, f"{path} 缺少方法: {sorted(methods - actual)}"


def test_no_unexpected_route_was_added(spec: dict) -> None:
    """新增路由必须同步更新本测试，避免契约被无声改变。"""
    extra = set(spec["paths"]) - set(EXPECTED_ROUTES) - {"/"}
    assert not extra, f"出现未登记的路由，请同步更新 EXPECTED_ROUTES: {sorted(extra)}"


# ---------------------------------------------------------------------------
# 文档一致性
# ---------------------------------------------------------------------------
def test_every_operation_has_summary_and_tags(spec: dict) -> None:
    problems = [
        f"{method} {path}"
        for method, path, op in _operations(spec)
        if not op.get("summary") or not op.get("tags")
    ]
    assert not problems, f"缺少 summary 或 tags: {problems}"


def test_tags_are_from_known_set(spec: dict) -> None:
    """标签统一，前端才能按模块分组展示。"""
    known = {
        "health",
        "repositories",
        "retrieval",
        "agent",
        "sandbox",
        "tutor",
        "library",
        "files",
        "check",
        "fix",
        "comment",
        # 多模型与网页填 Key：auth=会话管理，models=模型清单
        "auth",
        "models",
        "meta",
    }
    used = {tag for _, _, op in _operations(spec) for tag in op.get("tags", [])}
    assert used <= known, f"出现未知标签: {sorted(used - known)}"


def test_post_endpoints_declare_input(spec: dict) -> None:
    """POST 必须要么有 requestBody，要么有参数（如 /index?force=）。"""
    problems = [
        f"{method} {path}"
        for method, path, op in _operations(spec)
        if method == "POST" and "requestBody" not in op and not op.get("parameters")
    ]
    assert not problems, f"POST 既无请求体也无参数: {problems}"


def test_no_untagged_operations(spec: dict) -> None:
    untagged = [f"{m} {p}" for m, p, op in _operations(spec) if not op.get("tags")]
    assert not untagged


# ---------------------------------------------------------------------------
# 响应契约
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("schema_name", "field"),
    [
        ("SearchResponse", "hits"),
        ("SearchResponse", "trace_id"),
        ("TestGenerationResponse", "saved_path"),
        ("TestGenerationResponse", "sandbox_dir"),
        ("TestGenerationResponse", "trace_id"),
        ("AutoFixResponse", "final_diff"),
        ("AutoFixResponse", "iterations"),
        ("AutoFixResponse", "pr_draft"),
        ("AutoFixResponse", "issue_comment"),
        ("AutoFixResponse", "trace_id"),
        ("SandboxRunResponse", "coverage"),
        ("SandboxRunResponse", "isolated"),
        ("IndexBuildResponse", "embedder"),
        ("PrDraftRead", "branch_name"),
        ("PrDraftRead", "verified"),
        # 学生代码导师场景：前端直接渲染这些字段，改名即白屏
        ("AnalyzeResponse", "language"),
        ("AnalyzeResponse", "language_label"),
        ("AnalyzeResponse", "symbols"),
        ("CheckResponse", "score"),
        ("CheckResponse", "level"),
        ("CheckResponse", "issues"),
        ("CommentResponse", "commented_code"),
        ("FixResponse", "fixed_code"),
        ("FixResponse", "changes"),
        ("CodeChange", "reason"),
        ("TutorStatusResponse", "ai_available"),
        ("TutorStatusResponse", "supported_languages"),
        ("TutorHistoryResponse", "items"),
        # 文件自动分类：前端与验收脚本都直接读这些字段
        ("FileScanResponse", "counts"),
        ("FileScanResponse", "moved"),
        ("FileScanResponse", "files"),
        ("FileScanResponse", "inserted"),
        ("ClassifiedFileResult", "target_path"),
        ("ClassifiedFileResult", "action"),
        ("FileListResponse", "by_language"),
        ("FileListResponse", "items"),
        ("ClassifiedFileRead", "created_at"),
        ("ClassifiedFileRead", "exists"),
        # 拖拽入库的返回字段（前端据它提示"已存在/已改名"）
        ("FileUploadResponse", "rel_path"),
        ("FileUploadResponse", "renamed"),
        ("FileUploadResponse", "language_label"),
        ("FileUploadResponse", "note"),
        # AI 自动检测：需求里点名的四类输出 + 评分
        ("CodeCheckResponse", "score"),
        ("CodeCheckResponse", "level"),
        ("CodeCheckResponse", "errors"),
        ("CodeCheckResponse", "style"),
        ("CodeCheckResponse", "risks"),
        ("CodeCheckResponse", "advice"),
        ("CodeCheckResponse", "syntax_ok"),
        ("CodeCheckResponse", "ai_available"),
        ("CodeCheckResponse", "record_id"),
        ("CodeIssueRead", "source"),
        ("CodeIssueRead", "suggestion"),
        ("LocalCheckRead", "syntax_error"),
        ("LocalCheckRead", "metrics"),
        ("SyntaxErrorRead", "tool"),
        ("CodeMetricsRead", "max_complexity"),
        # 代码注释生成：三层注释 + 代码未被改动的复检结论
        ("CommentGenerateResponse", "commented_code"),
        ("CommentGenerateResponse", "original_code"),
        ("CommentGenerateResponse", "summary"),
        ("CommentGenerateResponse", "verification"),
        ("CommentVerificationRead", "code_unchanged"),
        ("CommentVerificationRead", "coverage_ratio"),
        ("CommentVerificationRead", "functions_covered"),
        ("CommentVerificationRead", "file_comment"),
        ("CommentVerificationRead", "verified"),
        ("CommentRecordDetail", "commented_code"),
        ("CommentRecordDetail", "original_code"),
        # 代码改错：修正后的代码 + 四问式修改说明 + 复检结论
        ("CodeFixResponse", "had_error"),
        ("CodeFixResponse", "fixed_code"),
        ("CodeFixResponse", "changes"),
        ("CodeFixResponse", "diff"),
        ("CodeFixResponse", "verification"),
        ("CodeFixResponse", "categories"),
        ("CodeChangeRead", "what"),
        ("CodeChangeRead", "why"),
        ("CodeChangeRead", "how"),
        ("CodeChangeRead", "avoid"),
        ("VerificationRead", "verified"),
        ("VerificationRead", "note"),
        ("CodeFixRecordDetail", "original_code"),
        ("CodeFixRecordDetail", "fixed_code"),
    ],
)
def test_response_schema_exposes_field(spec: dict, schema_name: str, field: str) -> None:
    """链路端到端依赖这些字段做数据传递，改名即断链。"""
    schemas = spec["components"]["schemas"]
    assert schema_name in schemas, f"缺少 schema: {schema_name}"
    assert field in schemas[schema_name]["properties"], f"{schema_name} 缺少字段 {field}"


def test_auto_fix_response_status_enum_matches_planner(spec: dict) -> None:
    """响应里的 status 取值必须与 Planner 的终止原因集合一致。"""
    from app.agents.planner import LoopStatus

    schema = spec["components"]["schemas"]["AutoFixResponse"]
    declared = set(schema["properties"]["status"]["enum"])

    expected = set(LoopStatus.__args__)  # type: ignore[attr-defined]
    assert declared == expected, (
        f"AutoFixResponse.status 与 LoopStatus 不一致："
        f"仅声明={sorted(declared - expected)} 仅实现={sorted(expected - declared)}"
    )


def test_all_response_models_are_declared(spec: dict) -> None:
    """所有响应模型都应出现在 components 里（否则文档不完整）。"""
    schemas = spec["components"]["schemas"]
    referenced = {
        ref.split("/")[-1]
        for _, _, op in _operations(spec)
        for ref in str(op.get("responses", {})).split("'")
        if ref.startswith("#/components/schemas/")
    }
    missing = {name for name in referenced if name not in schemas}
    assert not missing


# ---------------------------------------------------------------------------
# 运行时可达性
# ---------------------------------------------------------------------------
def test_all_routes_are_reachable_without_server_error(sqlite_path) -> None:
    """不带参数调用所有 GET 路由，不应出现 5xx（404/409/422 都是合理响应）。"""
    app = create_app()
    with TestClient(app) as client:
        problems = []
        for path in spec_paths_without_params():
            response = client.get(path)
            if response.status_code >= 500:
                problems.append(f"GET {path} -> {response.status_code}")
    assert not problems, f"GET 路由返回 5xx: {problems}"


def spec_paths_without_params() -> list[str]:
    """只取不需要路径参数的 GET 路由。"""
    app = create_app()
    paths = []
    for path, operations in app.openapi()["paths"].items():
        if "get" in operations and "{" not in path:
            paths.append(path)
    return sorted(paths)


def test_openapi_json_is_served(sqlite_path) -> None:
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/openapi.json")
    assert response.status_code == 200
    assert response.json()["info"]["title"]


def test_route_inventory_summary(spec: dict) -> None:
    """给出一份可读的路由清单，便于人工核对整合结果。"""
    by_tag: dict[str, list[str]] = defaultdict(list)
    for method, path, op in _operations(spec):
        tag = (op.get("tags") or ["(untagged)"])[0]
        by_tag[tag].append(f"{method:<5} {path}")

    total = sum(len(items) for items in by_tag.values())
    assert total == len(_operations(spec))

    # /api/v1 下的模块标签必须齐全。
    # meta（根路径的服务元信息 JSON）只在前端目录缺失时才注册，
    # 有 frontend/ 时根路径是 307 跳转且不进 schema，所以不纳入这个固定集合。
    api_tags = {tag for tag in by_tag if tag != "meta"}
    assert api_tags == {
        "health",
        "repositories",
        "retrieval",
        "agent",
        "sandbox",
        "tutor",
        "library",
        "files",
        "check",
        "fix",
        "comment",
        "auth",
        "models",
    }, f"模块标签与预期不符: {sorted(api_tags)}"
