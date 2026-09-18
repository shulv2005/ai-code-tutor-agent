"""健康检查相关响应模型。"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    return datetime.now(UTC)


class HealthResponse(BaseModel):
    """存活探针响应。"""

    status: Literal["ok"] = "ok"
    service: str
    version: str
    environment: str
    trace_id: str | None = None
    time: datetime = Field(default_factory=_utc_now)


class ReadinessResponse(HealthResponse):
    """就绪探针响应，额外携带依赖组件状态。"""

    database: Literal["ok", "error"]

