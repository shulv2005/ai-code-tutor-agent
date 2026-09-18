"""执行沙箱接口：/api/v1/sandbox

把 Step 4 生成的测试放进沙箱执行，返回测试结果与覆盖率报告。

安全要点（本模块的核心职责）：
1. **路径限定**：请求里的 `workspace` 必须落在 `DOCKER__WORKSPACE_DIR` 之内，
   且经 `resolve()` 展开符号链接，防止 `../../` 与软链接穿越读取宿主机任意目录。
2. **命令限制**：自定义命令只在 Docker 后端放行；本地后端无隔离，一律拒绝。
3. **绝不挂载 docker.sock**：详见 sandbox.py 的安全说明。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from app.api.deps import AppSettingsDep, SandboxRunnerDep
from app.core.config import Settings
from app.core.trace import current_trace_id
from app.schemas.sandbox import (
    CoverageFileRead,
    CoverageRead,
    SandboxRunRequestBody,
    SandboxRunResponse,
    SandboxStatusResponse,
)
from app.services.sandbox import (
    SandboxCommandError,
    SandboxError,
    SandboxPathError,
    SandboxRunRequest,
    SandboxUnavailable,
    resolve_workspace,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _security_summary(settings: Settings) -> dict[str, object]:
    """对外暴露当前沙箱的安全配置（便于审计，也便于前端展示）。"""
    docker = settings.docker
    return {
        "network_disabled": docker.network_disabled,
        # 安装依赖必须联网，此时网络隔离实际不生效，如实反映
        "network_actually_disabled": docker.network_disabled
        and not docker.install_dependencies,
        "read_only_root": docker.read_only_root,
        "cap_drop_all": True,
        "no_new_privileges": True,
        "privileged": False,
        "docker_socket_mounted": False,
        "sandbox_user": docker.sandbox_user or "(镜像默认)",
        "pids_limit": docker.pids_limit,
    }


@router.get(
    "/status",
    response_model=SandboxStatusResponse,
    summary="沙箱可用性与安全配置",
    description="返回实际生效的后端、镜像可用性与安全限制；local 后端表示无隔离。",
)
async def sandbox_status(
    runner: SandboxRunnerDep, settings: AppSettingsDep
) -> SandboxStatusResponse:
    availability = await runner.probe()
    docker = settings.docker
    return SandboxStatusResponse(
        backend=availability.backend,
        docker_available=availability.docker_available,
        isolated=availability.backend == "docker",
        image=availability.image,
        image_present=availability.image_present,
        reason=availability.reason,
        warnings=availability.warnings,
        limits={
            "timeout_seconds": docker.timeout_seconds,
            "memory_limit": docker.memory_limit,
            "cpu_limit": docker.cpu_limit,
            "pids_limit": docker.pids_limit,
            "max_output_bytes": docker.max_output_bytes,
        },
        security=_security_summary(settings),
    )


@router.post(
    "/run",
    response_model=SandboxRunResponse,
    summary="在沙箱中执行测试",
    description=(
        "把包含测试用例的目录挂载进临时容器，在其中运行 pytest 并采集 coverage.py 覆盖率，"
        "返回 stdout / stderr / 退出码，容器执行完毕即销毁。"
    ),
)
async def run_sandbox(
    payload: SandboxRunRequestBody,
    runner: SandboxRunnerDep,
    settings: AppSettingsDep,
) -> SandboxRunResponse:
    try:
        workspace = resolve_workspace(settings.docker, payload.workspace)
    except SandboxPathError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    request = SandboxRunRequest(
        workspace=workspace,
        test_target=payload.test_target,
        command=payload.command,
        timeout_seconds=payload.timeout_seconds,
        coverage_enabled=payload.coverage_enabled,
    )

    try:
        result = await runner.run(request)
    except SandboxUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except SandboxCommandError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except SandboxError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    coverage: CoverageRead | None = None
    if result.coverage is not None:
        coverage = CoverageRead(
            percent_covered=round(result.coverage.percent_covered, 2),
            covered_lines=result.coverage.covered_lines,
            num_statements=result.coverage.num_statements,
            missing_lines=result.coverage.missing_lines,
            file_count=len(result.coverage.files),
            files=[
                CoverageFileRead(
                    path=item.path,
                    percent_covered=round(item.percent_covered, 2),
                    covered_lines=item.covered_lines,
                    num_statements=item.num_statements,
                    missing_lines=item.missing_lines,
                )
                for item in result.coverage.files[:100]
            ],
        )

    notes = list(result.notes)
    if result.backend == "local":
        notes.append(
            "本地后端不具备隔离能力：如需真正的沙箱，请安装并启动 Docker，"
            "并把 DOCKER__BACKEND 设为 docker。"
        )

    return SandboxRunResponse(
        backend=result.backend,
        isolated=result.backend == "docker",
        exit_code=result.exit_code,
        succeeded=result.succeeded,
        timed_out=result.timed_out,
        truncated=result.truncated,
        duration_ms=round(result.duration_ms, 2),
        passed=result.passed,
        failed=result.failed,
        errors=result.errors,
        stdout=result.stdout,
        stderr=result.stderr,
        coverage=coverage,
        command=result.command,
        container_id=result.container_id,
        notes=notes,
        trace_id=current_trace_id(),
    )
