"""数据库连接与会话依赖单元测试。"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import text

from app.core.database import dispose_engine, get_session, init_db


async def test_init_db_and_session_roundtrip(sqlite_path: Path) -> None:
    try:
        await init_db()
        async with get_session() as session:
            result = await session.execute(text("SELECT 1"))
            assert result.scalar_one() == 1
    finally:
        await dispose_engine()


async def test_sqlite_file_is_created(sqlite_path: Path) -> None:
    try:
        await init_db()
        assert sqlite_path.exists()
    finally:
        await dispose_engine()

