"""健康检查：/api/v1/health（存活）与 /api/v1/health/ready（就绪，含数据库探测）。"""

from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import text

from app.api.deps import AppSettingsDep, DbSession
from app.core.trace import current_trace_id
from app.schemas.health import HealthResponse, ReadinessResponse

router = APIRouter()


@router.get("", response_model=HealthResponse, summary="存活探针")
async def health(settings: AppSettingsDep) -> HealthResponse:
    """返回服务基础信息，不依赖外部组件。"""
    return HealthResponse(
        service=settings.app.name,
        version=settings.app.version,
        environment=settings.app.environment,
        trace_id=current_trace_id(),
    )


@router.get("/ready", response_model=ReadinessResponse, summary="就绪探针")
async def readiness(session: DbSession, settings: AppSettingsDep) -> ReadinessResponse:
    """探测数据库连通性，用于容器编排的就绪检查。"""
    await session.execute(text("SELECT 1"))
    return ReadinessResponse(
        service=settings.app.name,
        version=settings.app.version,
        environment=settings.app.environment,
        database="ok",
        trace_id=current_trace_id(),
    )

