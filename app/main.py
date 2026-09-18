"""FastAPI 应用入口：把全部功能模块组装成一个可运行的服务。

===========================================================================
这个服务整合了哪些东西
===========================================================================
一、面向学生的「AI 代码导师」模块（都不需要 Docker）
    /api/v1/tutor      检测 / 生成注释 / 自动改错 + 学习历史（前端三个按钮用的轻量版）
    /api/v1/library    本地项目库：扫描 / 浏览 / 在线编辑 / 替换 / 删除（删除进 .trash）
    /api/v1/files      文件自动分类：把 .c/.h/.java/.py 归档到 c/ java/ python/，
                       并接收前端拖拽上传（POST /upload）
    /api/v1/check      AI 自动检测：本地语法检查（AST / Tree-sitter）+ AI 深度检测
    /api/v1/fix        AI 代码改错：本地分析 → AI 修正 → 本地复检（四问式讲解）
    /api/v1/comment    AI 注释生成：文件级 / 函数级 / 行内三层注释 + 代码未改动复检

    /api/v1/models     可选模型清单（网页「模型设置」下拉框的数据源，**不含任何 Key**）
    /api/v1/auth       网页填的 API Key 的临时保管：set_key / clear_key / status
                       —— Key 只存进程内存，30 分钟不用自动清除，不落库、不进日志

    上面这些 AI 接口都支持"动态模型/Key"：请求里带 model_id / api_key，
    或带 X-Session-Id 请求头（Key 存在内存会话里时用这个），
    就会用**用户自己选的模型 + 他自己的 Key**；都不带则回落到 .env 配置。

二、面向开源贡献的完整链路（Step 1-8）
    /api/v1/repositories  克隆 + 解析代码结构
    /api/v1/search        BM25 + FAISS + RRF 混合检索
    /api/v1/agent         生成 pytest 测试 / 一键自动修复
    /api/v1/sandbox       沙箱执行测试并采集覆盖率

三、网页界面（frontend/ 目录，挂载在 /ui）
    /ui/index.html      学生端：拖入/粘贴代码、跑 AI 功能、看历史、拖分隔条调布局

===========================================================================
create_app 工厂的组装顺序（改这里时请保持顺序）
===========================================================================
1. 读取配置       —— 一切路径、开关都来自 Settings，不在这里写死
2. Trace 中间件   —— 给每个请求分配 trace_id，日志与响应头都能对上
3. CORS           —— 开发期放开本地端口，便于前后端分离调试
4. 版本化路由     —— 只 include 一次 api_router，各模块挂载见 app/api/v1/router.py
5. 前端静态页面   —— /ui 挂 frontend/，根路径 307 跳到学生端
6. 兜底异常处理   —— 未捕获异常也返回 trace_id，便于定位

启动时会打印一份**路由清单**（按模块标签分组）：
这是"整合是否完整"最直接的证据——少挂一个模块，日志里一眼就能看出来。
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import api_router
from app.core.config import PROJECT_ROOT, Settings, get_settings
from app.core.database import dispose_engine, init_db
from app.core.llm_client import close_llm_clients
from app.core.logging import configure_logging
from app.core.trace import TraceIdMiddleware, current_trace_id

logger = logging.getLogger(__name__)

# 前端页面清单：启动横幅与整合自检都用它，避免在多处各写一份地址。
# 顺序即展示顺序：学生端在前（日常用），教师端在后（讲评用）。
FRONTEND_PAGES: tuple[tuple[str, str], ...] = (
    ("index.html", "学生端：检测 / 生成注释 / 自动改错 / 本地项目库"),
)


def log_route_inventory(app: FastAPI) -> dict[str, int]:
    """启动时打印一份按模块标签分组的路由清单。

    参数:
        app: FastAPI 应用实例（此时路由已全部 include 完毕）。

    返回:
        `{模块标签: 该模块的路由条数}`，测试可以直接断言"模块都在"。

    关键逻辑:
        整合类的工作最容易出的问题是"某个模块忘了挂"——它不会报错，
        只是访问时 404。把清单打出来，启动日志本身就是一份自查表。

        **为什么用 app.openapi() 而不是遍历 app.routes**：
        较新版本的 Starlette 把 `include_router(...)` 记录成**一个**
        `_IncludedRouter` 条目，遍历 `app.routes` 根本拿不到各条接口
        （实测：37 个接口只能看到 8 个顶层条目，其中没有一条 `/api/v1/*`）。
        OpenAPI schema 会把所有已注册接口展开，是唯一可靠的来源；
        它同时也排除了 `/`、`/favicon.ico` 这些不打算进文档的路由，
        正好符合"只统计业务模块"的需要。
    """
    prefix = get_settings().app.api_v1_prefix
    by_tag: dict[str, list[str]] = defaultdict(list)

    for path, operations in app.openapi()["paths"].items():
        if not path.startswith(prefix):
            continue
        for method, operation in operations.items():
            if method.upper() not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
                continue
            tags = operation.get("tags") or []
            by_tag[tags[0] if tags else "(未分类)"].append(f"{method.upper()} {path}")

    inventory = {tag: len(items) for tag, items in sorted(by_tag.items())}
    logger.info(
        "已注册 API 路由 %s 条，模块：%s",
        sum(inventory.values()),
        "，".join(f"{tag}({count})" for tag, count in inventory.items()),
    )
    for tag, items in sorted(by_tag.items()):
        logger.debug("模块 %s：%s", tag, " | ".join(sorted(items)))
    return inventory


def frontend_page_urls(app_url: str) -> list[str]:
    """拼出前端各页面的完整访问地址。

    参数:
        app_url: 服务根地址，例如 `http://127.0.0.1:8000`。

    返回:
        形如 `["http://127.0.0.1:8000/ui/index.html  学生端：…", ...]` 的列表。
    """
    return [f"{app_url}/ui/{name}  {desc}" for name, desc in FRONTEND_PAGES]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """应用生命周期：启动时准备资源，关闭时释放资源。"""
    settings: Settings = app.state.settings

    configure_logging(settings.app.log_level)
    settings.ensure_directories()
    await init_db()

    logger.info(
        "应用启动完成: name=%s env=%s version=%s llm_configured=%s",
        settings.app.name,
        settings.app.environment,
        settings.app.version,
        settings.llm.is_configured,
    )
    # 把"整合是否完整"直接打进启动日志：少挂一个模块一眼可见
    app.state.route_inventory = log_route_inventory(app)

    # 打印页面地址：同学双击 start.bat 后，控制台里就能看到该打开哪个页面
    base_url = f"http://{settings.app.host}:{settings.app.port}"
    for line in frontend_page_urls(base_url):
        logger.info("页面地址：%s", line)
    logger.info("接口文档：%s/docs", base_url)
    if not settings.llm.is_configured:
        logger.warning(
            "未配置 LLM__API_KEY：AI 检测/改错/注释仍可用（返回本地分析结果），"
            "配好 Key 后重启服务即可获得完整的 AI 分析。"
        )

    try:
        yield
    finally:
        await close_llm_clients()
        await dispose_engine()
        logger.info("应用已关闭，数据库连接池与 LLM 连接已释放")


def create_app(settings: Settings | None = None) -> FastAPI:
    """创建 FastAPI 应用实例（工厂模式）。"""
    settings = settings or get_settings()

    app = FastAPI(
        title=settings.app.name,
        version=settings.app.version,
        description=(
            "AI 代码导师与开源协作智能体后端服务。\n\n"
            "- **学生端**：代码文件自动分类、AI 检测、AI 改错、AI 注释生成\n"
            "- **开源协作**：仓库解析、混合检索、测试生成、沙箱执行、自动修复\n\n"
            "网页入口：`/ui/index.html`（学生端）"
        ),
        debug=settings.app.debug,
        lifespan=lifespan,
    )
    app.state.settings = settings

    # 1) Trace 中间件：为每个请求分配 trace_id，并透传到日志与响应头
    app.add_middleware(TraceIdMiddleware, header_name=settings.app.trace_header)

    # 2) CORS：为后续 Vue 3 前端预留（开发期默认放开本地端口）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.app.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[settings.app.trace_header],
    )

    # 3) 路由：**所有业务模块都在 api_router 里挂载**（见 app/api/v1/router.py），
    #    这里只 include 一次。新增模块时改路由文件即可，不用动 main.py。
    app.include_router(api_router, prefix=settings.app.api_v1_prefix)

    # 4) 前端静态页面：把 frontend/ 挂到 /ui，并把根路径重定向过去。
    #    这样同学双击 start.bat 后打开 http://127.0.0.1:8000 就能直接看到界面。
    #    目录不存在时自动跳过，不影响后端单独运行。
    frontend_dir = PROJECT_ROOT / "frontend"
    if frontend_dir.is_dir():
        app.mount("/ui", StaticFiles(directory=str(frontend_dir), html=True), name="ui")

        @app.get("/", include_in_schema=False)
        async def index_redirect() -> RedirectResponse:
            """根路径直接跳到前端页面，省得学生手动输地址。"""
            return RedirectResponse(url="/ui/index.html")

        @app.get("/favicon.ico", include_in_schema=False)
        async def favicon() -> Response:
            """浏览器会自动请求图标；返回 204 避免刷出一堆 404 日志。"""
            return Response(status_code=204)

    else:
        logger.warning("未找到 frontend 目录，前端页面不可用：%s", frontend_dir)

        @app.get("/", tags=["meta"], summary="服务元信息")
        async def root() -> dict[str, Any]:
            return {
                "service": settings.app.name,
                "version": settings.app.version,
                "environment": settings.app.environment,
                "docs": "/docs",
                "api_prefix": settings.app.api_v1_prefix,
            }

    # 5) 兜底异常处理：把 trace_id 一并返回，便于问题定位
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("未处理异常: path=%s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error", "trace_id": current_trace_id()},
        )

    return app


app = create_app()

