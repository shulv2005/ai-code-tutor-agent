"""API v1 路由聚合：后续模块（PR 草稿）在此挂载子路由。"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.endpoints import (
    agent,
    auth,
    check,
    comment,
    files,
    fix,
    health,
    library,
    models,
    repositories,
    sandbox,
    search,
    tutor,
)

api_router = APIRouter()
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(
    repositories.router, prefix="/repositories", tags=["repositories"]
)
api_router.include_router(search.router, prefix="/search", tags=["retrieval"])
api_router.include_router(agent.router, prefix="/agent", tags=["agent"])
api_router.include_router(sandbox.router, prefix="/sandbox", tags=["sandbox"])
# AI 代码导师：面向前端「学生代码学习」页面
api_router.include_router(tutor.router, prefix="/tutor", tags=["tutor"])
# 本地项目库：原地扫描并按语言分类（只读，不移动文件）
api_router.include_router(library.router, prefix="/library", tags=["library"])
# 本地代码文件自动分类：会真正把文件归档到 c/ java/ python/ 目录
api_router.include_router(files.router, prefix="/files", tags=["files"])
# AI 自动检测：先本地语法检查（AST / Tree-sitter），再 AI 深度检测
api_router.include_router(check.router, prefix="/check", tags=["check"])
# 代码改错：本地分析 → AI 修正 → 本地复检，含对比学习用的历史记录
api_router.include_router(fix.router, prefix="/fix", tags=["fix"])
# 代码注释生成：三层中文注释 + 代码未被改动的复检
api_router.include_router(comment.router, prefix="/comment", tags=["comment"])
# API Key 会话：网页上填的 Key 只存内存，这里负责存取与清除
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
# 模型清单：前端下拉框的数据源（只报"有没有 Key"，不给 Key）
api_router.include_router(models.router, prefix="/models", tags=["models"])

__all__ = ["api_router"]
