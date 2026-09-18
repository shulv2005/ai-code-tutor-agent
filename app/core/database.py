"""数据库层：SQLAlchemy 2.0 异步引擎 + 会话管理。

- 开发期使用 SQLite（aiosqlite 驱动）；生产期只需把 DATABASE__URL_OVERRIDE
  换成 PostgreSQL（如 postgresql+asyncpg://...），业务代码无需改动。
- 引擎与会话工厂采用懒加载，方便测试中切换数据库地址后再创建。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """所有 ORM 模型的基类。"""


_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """获取（并缓存）异步数据库引擎。"""
    global _engine
    if _engine is None:
        settings = get_settings()
        connect_args: dict[str, Any] = {}
        if settings.database.url.startswith("sqlite"):
            # aiosqlite 会在独立线程中执行，需关闭同线程校验
            connect_args["check_same_thread"] = False
        _engine = create_async_engine(
            settings.database.url,
            echo=settings.database.echo,
            future=True,
            pool_pre_ping=True,
            connect_args=connect_args,
        )
        logger.debug("已创建数据库引擎: %s", settings.database.url)
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """获取（并缓存）会话工厂。"""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    return _session_factory


async def init_db() -> None:
    """建表：先导入模型包完成元数据注册，再执行 create_all。"""
    import app.models  # noqa: F401  确保 ORM 模型注册到 Base.metadata

    async with get_engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.debug("数据库表结构已就绪")


async def dispose_engine() -> None:
    """释放连接池（应用关闭或测试结束时调用）。"""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """独立使用的会话上下文管理器（脚本、后台任务、Agent 内部）。"""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI 依赖注入用的会话生成器。"""
    async with get_session() as session:
        yield session

