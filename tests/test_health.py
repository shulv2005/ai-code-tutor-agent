"""FastAPI 入口与健康检查接口测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


def test_root_endpoint_redirects_to_frontend(sqlite_path: Path) -> None:
    """根路径现在直接跳到学生前端页面，省得同学手输地址。

    前端目录存在时做 307 跳转；只有后端裸跑（没有 frontend/）时才回落到 JSON。
    这里两种部署形态都要覆盖，测试才不会被目录是否存在左右。
    """
    from app.core.config import PROJECT_ROOT

    app = create_app()
    with TestClient(app) as client:
        if (PROJECT_ROOT / "frontend").is_dir():
            response = client.get("/", follow_redirects=False)
            assert response.status_code in {302, 307}
            assert response.headers["location"] == "/ui/index.html"

            # 跟随跳转后应当真的拿到 HTML 页面
            page = client.get("/")
            assert page.status_code == 200
            assert "text/html" in page.headers["content-type"]
            assert "<html" in page.text.lower()
        else:
            response = client.get("/")
            assert response.status_code == 200
            assert response.json()["service"] == "opensource-collab-agent"


def test_frontend_static_assets_are_served(sqlite_path: Path) -> None:
    """前端四件套必须都能通过 /ui 拿到，否则页面会白屏。"""
    from app.core.config import PROJECT_ROOT

    if not (PROJECT_ROOT / "frontend").is_dir():
        pytest.skip("没有 frontend 目录，跳过前端静态资源检查")

    app = create_app()
    with TestClient(app) as client:
        for path in (
            "/ui/index.html",
            "/ui/css/style.css",
            "/ui/js/app.js",
            "/ui/vendor/highlight.min.js",
            "/ui/vendor/github.min.css",
        ):
            response = client.get(path)
            assert response.status_code == 200, f"{path} 拿不到，前端会缺资源"


def test_favicon_returns_no_content(sqlite_path: Path) -> None:
    """浏览器会自动请求图标，返回 204 避免刷 404 日志。"""
    from app.core.config import PROJECT_ROOT

    if not (PROJECT_ROOT / "frontend").is_dir():
        pytest.skip("没有 frontend 目录，跳过图标检查")

    app = create_app()
    with TestClient(app) as client:
        assert client.get("/favicon.ico").status_code == 204


def test_health_endpoint_returns_trace_header(sqlite_path: Path) -> None:
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/api/v1/health")

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "ok"
    assert body["trace_id"]
    assert response.headers["X-Trace-Id"] == body["trace_id"]


def test_incoming_trace_header_is_reused(sqlite_path: Path) -> None:
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/api/v1/health", headers={"X-Trace-Id": "trace-abc"})

    assert response.headers["X-Trace-Id"] == "trace-abc"
    assert response.json()["trace_id"] == "trace-abc"


def test_readiness_endpoint_checks_database(sqlite_path: Path) -> None:
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/api/v1/health/ready")

    assert response.status_code == 200
    assert response.json()["database"] == "ok"


def test_openapi_schema_is_available(sqlite_path: Path) -> None:
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200
    assert "/api/v1/health" in response.json()["paths"]

