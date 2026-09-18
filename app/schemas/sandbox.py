"""执行沙箱模块的 API 契约。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SandboxRunRequestBody(BaseModel):
    """沙箱执行请求。

    `workspace` 可以是绝对路径，也可以是相对沙箱工作区的路径；
    两者都会被强制校验为"必须位于 DOCKER__WORKSPACE_DIR 之内"。
    """

    model_config = ConfigDict(
        json_schema_extra={"example": {"workspace": "1a2b3c4d5e6f7788/sandbox_repo"}}
    )

    workspace: str = Field(
        ...,
        min_length=1,
        max_length=1024,
        description="包含测试代码的目录（须位于沙箱工作区内）",
    )
    test_target: str = Field(
        default="tests",
        max_length=256,
        description="pytest 目标路径（相对 workspace）",
    )
    timeout_seconds: int | None = Field(
        default=None, ge=1, le=3600, description="执行超时（秒），缺省用配置值"
    )
    coverage_enabled: bool = Field(default=True, description="是否采集覆盖率")
    command: list[str] | None = Field(
        default=None,
        max_length=32,
        description="自定义命令；仅 Docker 后端允许（本地后端无隔离，会拒绝）",
    )


class CoverageFileRead(BaseModel):
    """单文件覆盖率。"""

    path: str
    percent_covered: float
    covered_lines: int
    num_statements: int
    missing_lines: int


class CoverageRead(BaseModel):
    """覆盖率汇总。"""

    percent_covered: float
    covered_lines: int
    num_statements: int
    missing_lines: int
    file_count: int
    files: list[CoverageFileRead] = Field(default_factory=list)


class SandboxRunResponse(BaseModel):
    """沙箱执行结果。"""

    backend: Literal["docker", "local"] = Field(
        description="实际执行的后端；local 表示**无隔离**的降级路径"
    )
    isolated: bool = Field(description="是否具备真正的隔离能力")
    exit_code: int
    succeeded: bool
    timed_out: bool
    truncated: bool
    duration_ms: float
    passed: int = 0
    failed: int = 0
    errors: int = 0
    stdout: str = ""
    stderr: str = ""
    coverage: CoverageRead | None = None
    command: list[str] = Field(default_factory=list)
    container_id: str | None = None
    notes: list[str] = Field(default_factory=list)
    trace_id: str | None = None


class SandboxStatusResponse(BaseModel):
    """沙箱可用性：前端可据此提示"当前无隔离"或"需要启动 Docker"。"""

    backend: Literal["docker", "local"] | None = None
    docker_available: bool
    isolated: bool
    image: str
    image_present: bool = False
    reason: str = ""
    warnings: list[str] = Field(default_factory=list)
    limits: dict[str, object] = Field(default_factory=dict)
    security: dict[str, object] = Field(default_factory=dict)
