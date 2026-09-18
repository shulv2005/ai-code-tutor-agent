"""执行沙箱：在隔离环境中运行生成的 pytest 测试并采集覆盖率。

双后端设计（与 Step 3 的 fastembed/hashing、Step 4 的真实/假 LLM 同一思路）：

- `DockerSandboxRunner`：**真正的安全边界**。临时容器 + 无网络 + 丢弃全部
  capabilities + 只读根文件系统 + 资源限额，用完即毁。
- `LocalSubprocessSandboxRunner`：Docker 不可用时的降级路径。
  ⚠️ **它不是安全边界**——代码直接跑在宿主机上，仅依赖超时与输出截断兜底。
  仅在开发/CI 且只执行自己生成的代码时使用。

选哪个由 `DOCKER__BACKEND` 决定：auto（默认，优先 docker）/ docker / local。
真正执行了哪个后端会体现在返回结果与 Trace 里，绝不静默混淆。
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from app.core.config import DockerSettings
from app.core.trace import trace_span

logger = logging.getLogger(__name__)

BackendName = Literal["docker", "local"]

# 沙箱内的工作目录挂载点（与 DOCKER__WORKSPACE_MOUNT 对应）
COVERAGE_JSON_NAME = "coverage.json"


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------
class SandboxError(RuntimeError):
    """沙箱执行基类异常。"""


class SandboxUnavailable(SandboxError):
    """沙箱后端不可用（Docker daemon 未运行、镜像缺失等）。"""


class SandboxPathError(SandboxError):
    """工作目录不在允许的沙箱根目录内（路径穿越防护）。"""


class SandboxCommandError(SandboxError):
    """命令不被允许。"""


# ---------------------------------------------------------------------------
# DTO
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class SandboxRunRequest:
    """一次沙箱执行的请求。"""

    # 宿主机上要挂载进沙箱的目录（必须位于 workspace_path 内）
    workspace: Path
    # 测试相对路径（相对 workspace），默认 tests
    test_target: str = "tests"
    # 自定义命令；仅 docker 后端允许
    command: list[str] | None = None
    timeout_seconds: int | None = None
    coverage_enabled: bool = True
    # 追加给 pytest 的参数，例如 ["-c", "agent_pytest.ini"] 以隔离仓库自带配置
    extra_pytest_args: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CoverageFileSummary:
    """单文件覆盖率。"""

    path: str
    percent_covered: float
    covered_lines: int
    num_statements: int
    missing_lines: int


@dataclass(slots=True)
class CoverageSummary:
    """整体覆盖率报告。"""

    percent_covered: float = 0.0
    covered_lines: int = 0
    num_statements: int = 0
    missing_lines: int = 0
    files: list[CoverageFileSummary] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "percent_covered": round(self.percent_covered, 2),
            "covered_lines": self.covered_lines,
            "num_statements": self.num_statements,
            "missing_lines": self.missing_lines,
            "file_count": len(self.files),
        }


@dataclass(slots=True)
class SandboxResult:
    """一次沙箱执行的完整结果。"""

    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float
    backend: BackendName
    timed_out: bool = False
    truncated: bool = False
    # 测试统计：从 pytest 输出里解析，便于上层（Step 7）判断是否需要修复
    passed: int = 0
    failed: int = 0
    errors: int = 0
    coverage: CoverageSummary | None = None
    # 实际执行的命令（便于复现与审计）
    command: list[str] = field(default_factory=list)
    container_id: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    @property
    def tests_failed(self) -> bool:
        return self.failed > 0 or self.errors > 0


@dataclass(slots=True)
class SandboxAvailability:
    """沙箱后端可用性探测结果。"""

    backend: BackendName | None
    docker_available: bool
    image: str
    image_present: bool = False
    reason: str = ""
    # 降级到 local 时给出的警告，API 会原样透出
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 命令构造
# ---------------------------------------------------------------------------
def build_pytest_script(
    *,
    test_target: str,
    coverage_enabled: bool,
    workspace_mount: str,
    install_dependencies: bool = False,
    pip_index_url: str = "",
    extra_pytest_args: Sequence[str] = (),
) -> str:
    """构造容器内执行的 shell 脚本。

    设计要点：
    - 用 `coverage run -m pytest` 包裹，测试失败也要产出覆盖率报告再退出，
      因此分两条命令执行并显式保存 pytest 退出码（`|| true` 会让退出码失真）。
    - 关闭 pytest 缓存与字节码写入，减少对只读根文件系统的写入需求。
    - `extra_pytest_args` 用于隔离仓库自带的 pytest 配置
      （很多仓库的 addopts 依赖沙箱里没装的插件，会让 pytest 直接报错退出）。
    """
    lines = ["set -u"]
    if install_dependencies:
        pip_args = "--no-cache-dir --quiet"
        if pip_index_url:
            pip_args += f" -i {_shell_quote(pip_index_url)}"
        lines.append(
            f"python -m pip install {pip_args} pytest coverage "
            f"|| {{ echo '__SANDBOX_SETUP_FAILED__' >&2; exit 90; }}"
        )

    target = _shell_quote(test_target)
    extra = " ".join(_shell_quote(arg) for arg in extra_pytest_args)
    extra = f" {extra}" if extra else ""
    if coverage_enabled:
        lines += [
            f"cd {workspace_mount}",
            "python -m coverage run --source=. -m pytest "
            f"{target} -q -p no:cacheprovider{extra}",
            "PYTEST_EXIT=$?",
            # 无论测试是否失败都产出覆盖率，便于反馈闭环使用
            f"python -m coverage json -o {COVERAGE_JSON_NAME} -q || true",
            "exit $PYTEST_EXIT",
        ]
    else:
        lines += [
            f"cd {workspace_mount}",
            f"python -m pytest {target} -q -p no:cacheprovider{extra}",
            "exit $?",
        ]
    return "\n".join(lines)


def _shell_quote(value: str) -> str:
    """单引号包裹并转义，防止命令注入。"""
    return "'" + value.replace("'", "'\"'\"'") + "'"


def build_local_commands(
    *, test_target: str, coverage_enabled: bool, executable: str,
    extra_pytest_args: Sequence[str] = (),
) -> list[list[str]]:
    """构造本地后端要执行的命令序列（不经 shell，跨平台）。"""
    pytest_cmd = [
        executable, "-m", "pytest", test_target,
        "-q", "-p", "no:cacheprovider", *extra_pytest_args,
    ]
    if not coverage_enabled:
        return [pytest_cmd]
    return [
        [executable, "-m", "coverage", "run", "--source=.", "-m", *pytest_cmd[1:]],
        [executable, "-m", "coverage", "json", "-o", COVERAGE_JSON_NAME, "-q"],
    ]


# ---------------------------------------------------------------------------
# 输出解析
# ---------------------------------------------------------------------------
def parse_coverage_json(path: Path) -> CoverageSummary | None:
    """解析 coverage.py 输出的 JSON 报告。"""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("覆盖率报告解析失败: %s", path, exc_info=True)
        return None

    totals = payload.get("totals") or {}
    files: list[CoverageFileSummary] = []
    for name, item in (payload.get("files") or {}).items():
        summary = item.get("summary") or {}
        files.append(
            CoverageFileSummary(
                path=name,
                percent_covered=float(summary.get("percent_covered", 0.0) or 0.0),
                covered_lines=int(summary.get("covered_lines", 0) or 0),
                num_statements=int(summary.get("num_statements", 0) or 0),
                missing_lines=int(summary.get("missing_lines", 0) or 0),
            )
        )
    files.sort(key=lambda item: (-item.missing_lines, item.path))

    return CoverageSummary(
        percent_covered=float(totals.get("percent_covered", 0.0) or 0.0),
        covered_lines=int(totals.get("covered_lines", 0) or 0),
        num_statements=int(totals.get("num_statements", 0) or 0),
        missing_lines=int(totals.get("missing_lines", 0) or 0),
        files=files,
    )


_PYTEST_SUMMARY = (
    r"(?P<count>\d+)\s+(?P<kind>passed|failed|error|errors|skipped|xfailed|xpassed|warning|warnings)"
)
# 汇总行可能在 -qq 等静默模式下消失，此时退化为统计短摘要里的 FAILED/ERROR 行
_PYTEST_SHORT_LINE = re.compile(r"^(?P<kind>FAILED|ERROR)\s+\S", re.MULTILINE)


def parse_pytest_summary(output: str) -> dict[str, int]:
    """从 pytest 输出中解析各状态用例数。

    两条路径：
    1. 汇总行 `1 failed, 2 passed in 0.12s`（主路径，最准确）；
    2. 短摘要里的 `FAILED ...` / `ERROR ...` 行（兜底）。

    需要兜底是因为静默级别提高（如 `-qq`）时汇总行会被抑制，
    而反馈闭环依赖这个计数来判断"是否修好了"。
    """
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    tail = output[-8000:] if len(output) > 8000 else output
    for match in re.finditer(_PYTEST_SUMMARY, tail):
        kind = match.group("kind")
        count = int(match.group("count"))
        if kind == "failed":
            counts["failed"] += count
        elif kind in ("error", "errors"):
            counts["errors"] += count
        elif kind == "passed":
            counts["passed"] += count
        elif kind == "skipped":
            counts["skipped"] += count

    if counts["failed"] == 0 and counts["errors"] == 0:
        # 兜底：数短摘要行
        for match in _PYTEST_SHORT_LINE.finditer(output):
            if match.group("kind") == "FAILED":
                counts["failed"] += 1
            else:
                counts["errors"] += 1
    return counts


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    """按字节上限截断输出，避免日志/响应被刷爆。"""
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= limit:
        return text, False
    head = encoded[: limit // 2].decode("utf-8", errors="replace")
    tail = encoded[-(limit // 2) :].decode("utf-8", errors="replace")
    return f"{head}\n...<输出已截断 {len(encoded) - limit} 字节>...\n{tail}", True


# ---------------------------------------------------------------------------
# 后端协议
# ---------------------------------------------------------------------------
@runtime_checkable
class SandboxRunner(Protocol):
    """沙箱后端协议。"""

    @property
    def name(self) -> BackendName: ...

    async def probe(self) -> SandboxAvailability: ...

    async def run(self, request: SandboxRunRequest) -> SandboxResult: ...


# ---------------------------------------------------------------------------
# Docker 后端
# ---------------------------------------------------------------------------
class DockerSandboxRunner:
    """基于 Docker 的隔离执行后端。

    容器生命周期：create -> start -> wait -> logs -> **remove(force=True)**
    全程在 try/finally 中，保证任何异常路径下容器都会被销毁。
    另外给容器打上 label，便于清理历史遗留的孤儿容器。
    """

    ORPHAN_LABEL = "opensource-agent.sandbox"

    def __init__(self, settings: DockerSettings) -> None:
        self._settings = settings
        self._client: Any | None = None

    @property
    def name(self) -> BackendName:
        return "docker"

    def _get_client(self) -> Any:
        """懒加载 docker 客户端（未安装/未运行 daemon 时抛 SandboxUnavailable）。"""
        if self._client is not None:
            return self._client
        try:
            import docker  # 延迟导入：未装 SDK 也不影响其它后端
        except ImportError as exc:  # pragma: no cover - 依赖已声明
            raise SandboxUnavailable("未安装 docker Python SDK（pip install docker）") from exc
        try:
            client = docker.from_env()
            client.ping()
        except Exception as exc:  # noqa: BLE001 - docker 异常类型不稳定
            raise SandboxUnavailable(f"Docker daemon 不可用: {exc}") from exc
        self._client = client
        return client

    async def probe(self) -> SandboxAvailability:
        """探测 daemon 与镜像可用性。"""
        try:
            client = await asyncio.to_thread(self._get_client)
        except SandboxUnavailable as exc:
            return SandboxAvailability(
                backend=None,
                docker_available=False,
                image=self._settings.image,
                reason=str(exc),
            )

        image_present = await asyncio.to_thread(self._image_exists, client)
        return SandboxAvailability(
            backend="docker",
            docker_available=True,
            image=self._settings.image,
            image_present=image_present,
            reason="" if image_present else f"本地不存在镜像 {self._settings.image}",
        )

    def _image_exists(self, client: Any) -> bool:
        try:
            client.images.get(self._settings.image)
            return True
        except Exception:  # noqa: BLE001 - ImageNotFound 等
            return False

    def _container_kwargs(self, request: SandboxRunRequest, script: str) -> dict[str, Any]:
        """组装容器创建参数——这里是**安全边界的全部内容**。"""
        settings = self._settings
        # 安装依赖必须联网，此时无法同时满足 network_disabled，
        # 因此仅在 install_dependencies 为真时放行网络，并记录到 notes。
        need_network = settings.install_dependencies
        network_disabled = settings.network_disabled and not need_network

        kwargs: dict[str, Any] = {
            "image": settings.image,
            "command": ["/bin/sh", "-c", script],
            "working_dir": settings.workspace_mount,
            "detach": True,
            # 不交给 docker 自动删除：需要先取日志与退出码
            "auto_remove": False,
            "network_disabled": network_disabled,
            "mem_limit": settings.memory_limit,
            "nano_cpus": int(settings.cpu_limit * 1_000_000_000),
            "pids_limit": settings.pids_limit,
            # 只挂载这一个任务目录，绝不挂载宿主机根目录或 docker.sock
            "volumes": {
                str(request.workspace): {
                    "bind": settings.workspace_mount,
                    "mode": "rw",
                }
            },
            "environment": {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1",
                "PYTHONHASHSEED": "0",  # 让测试结果可复现
                "HOME": "/tmp",
                "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            },
            "labels": {self.ORPHAN_LABEL: "true"},
            # 安全性选项
            "privileged": False,
            "cap_drop": ["ALL"],
            "security_opt": ["no-new-privileges:true"],
        }
        if settings.read_only_root:
            kwargs["read_only"] = True
            # 只读根文件系统下，/tmp 必须可写（pip、pytest 临时文件都要用）
            kwargs["tmpfs"] = {"/tmp": f"rw,size={settings.tmpfs_size}"}
        if settings.sandbox_user:
            kwargs["user"] = settings.sandbox_user
        return kwargs

    async def run(self, request: SandboxRunRequest) -> SandboxResult:
        client = await asyncio.to_thread(self._get_client)
        settings = self._settings
        timeout = request.timeout_seconds or settings.timeout_seconds
        notes: list[str] = []

        script = build_pytest_script(
            test_target=request.test_target,
            coverage_enabled=request.coverage_enabled,
            workspace_mount=settings.workspace_mount,
            install_dependencies=settings.install_dependencies,
            pip_index_url=settings.pip_index_url,
            extra_pytest_args=request.extra_pytest_args,
        )
        if settings.install_dependencies:
            notes.append(
                "已在容器启动时 pip 安装 pytest/coverage，此步骤需要网络。"
                "生产环境建议预构建镜像并设 DOCKER__INSTALL_DEPENDENCIES=false，"
                "以便全程开启 DOCKER__NETWORK_DISABLED=true。"
            )
        if not settings.network_disabled:
            notes.append("网络隔离已关闭（DOCKER__NETWORK_DISABLED=false）")

        kwargs = self._container_kwargs(request, script)
        started = time.perf_counter()
        container: Any | None = None

        with trace_span(
            "sandbox.run",
            kind="sandbox",
            payload={"workspace": str(request.workspace), "target": request.test_target},
            metadata={"backend": "docker", "image": settings.image, "timeout": timeout},
        ) as span:
            try:
                container = await asyncio.to_thread(client.containers.create, **kwargs)
                container_id = container.id
                await asyncio.to_thread(container.start)

                timed_out = False
                try:
                    status = await asyncio.to_thread(container.wait, timeout=timeout)
                    exit_code = int(status.get("StatusCode", -1))
                except Exception as exc:  # noqa: BLE001 - 超时或连接中断
                    logger.warning("等待容器结束失败，将强制终止: %s", exc)
                    timed_out = True
                    exit_code = -1
                    await asyncio.to_thread(self._kill, container)

                stdout = await self._logs(container, stdout=True)
                stderr = await self._logs(container, stdout=False)
            finally:
                # 无论如何都要销毁容器，避免资源泄漏
                if container is not None:
                    await asyncio.to_thread(self._remove, container, span)

            duration_ms = (time.perf_counter() - started) * 1000
            stdout, truncated_out = _truncate(stdout, settings.max_output_bytes)
            stderr, truncated_err = _truncate(stderr, settings.max_output_bytes)

            coverage = (
                parse_coverage_json(request.workspace / COVERAGE_JSON_NAME)
                if request.coverage_enabled
                else None
            )
            counts = parse_pytest_summary(stdout + "\n" + stderr)
            if timed_out:
                notes.append(f"执行超时（{timeout} 秒），容器已被强制终止")

            span.set_metadata(exit_code=exit_code, timed_out=timed_out)
            span.set_output(
                {
                    "exit_code": exit_code,
                    "passed": counts["passed"],
                    "failed": counts["failed"],
                    "coverage": coverage.to_dict() if coverage else None,
                }
            )

            return SandboxResult(
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                duration_ms=duration_ms,
                backend="docker",
                timed_out=timed_out,
                truncated=truncated_out or truncated_err,
                passed=counts["passed"],
                failed=counts["failed"],
                errors=counts["errors"],
                coverage=coverage,
                command=["python", "-m", "coverage", "run", "-m", "pytest", request.test_target],
                container_id=container_id,
                notes=notes,
            )

    async def _logs(self, container: Any, *, stdout: bool) -> str:
        """分别取 stdout / stderr（SDK 会做多路复用解包）。"""
        try:
            raw = await asyncio.to_thread(
                container.logs, stdout=stdout, stderr=not stdout, timestamps=False
            )
        except Exception:  # noqa: BLE001 - 容器已被删除时取日志会失败
            logger.debug("读取容器日志失败", exc_info=True)
            return ""
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        return str(raw)

    @staticmethod
    def _kill(container: Any) -> None:
        try:
            container.kill()
        except Exception:  # noqa: BLE001 - 已退出时 kill 会报错
            logger.debug("终止容器失败（可能已退出）", exc_info=True)

    @staticmethod
    def _remove(container: Any, span: Any) -> None:
        """强制删除容器与匿名卷。"""
        try:
            container.remove(force=True, v=True)
            span.set_metadata(container_removed=True)
        except Exception:  # noqa: BLE001
            logger.warning("删除沙箱容器失败，可能残留资源", exc_info=True)
            span.set_metadata(container_removed=False)

    async def cleanup_orphans(self) -> int:
        """清理历史遗留的沙箱容器（进程崩溃时可能残留）。"""
        client = await asyncio.to_thread(self._get_client)
        removed = 0
        containers = await asyncio.to_thread(
            client.containers.list, all=True, filters={"label": self.ORPHAN_LABEL}
        )
        for container in containers:
            try:
                await asyncio.to_thread(container.remove, force=True, v=True)
                removed += 1
            except Exception:  # noqa: BLE001
                logger.warning("清理孤儿容器失败: %s", container.id[:12], exc_info=True)
        return removed


# ---------------------------------------------------------------------------
# 本地降级后端
# ---------------------------------------------------------------------------
class LocalSubprocessSandboxRunner:
    """本地子进程后端——Docker 不可用时的降级路径。

    ⚠️ **这不是安全边界。** 生成的测试代码会以当前用户身份直接在宿主机上
    执行，可读写宿主机文件系统、访问网络。它只提供：
    - 超时终止
    - 输出截断
    - 独立工作目录

    因此：
    - 只应在开发/CI 且只执行自己生成代码时使用；
    - 禁止在该后端下接受外部传入的自定义命令（见 API 层的校验）。
    """

    def __init__(self, settings: DockerSettings) -> None:
        self._settings = settings

    @property
    def name(self) -> BackendName:
        return "local"

    async def probe(self) -> SandboxAvailability:
        return SandboxAvailability(
            backend="local",
            docker_available=False,
            image=self._settings.image,
            reason="使用本地子进程后端（Docker 不可用）",
            warnings=[
                "当前为本地子进程沙箱，**不具备隔离能力**："
                "被测代码可直接访问宿主机文件系统与网络。",
                "仅建议在开发/CI 环境执行自己生成的代码时使用；"
                "请安装并启动 Docker 以启用真正的隔离。",
            ],
        )

    async def run(self, request: SandboxRunRequest) -> SandboxResult:
        settings = self._settings
        timeout = request.timeout_seconds or settings.timeout_seconds
        started = time.perf_counter()

        if request.command is not None:
            # 本地后端接受任意命令等于把宿主机 shell 暴露给调用方
            raise SandboxCommandError(
                "本地沙箱后端不允许自定义命令（无隔离，风险过高）。"
                "请启动 Docker 后使用 docker 后端。"
            )

        executable = sys.executable
        notes = [
            "⚠️ 本次执行使用本地子进程后端，**无隔离**："
            "代码以当前用户身份运行在宿主机上。"
        ]

        # coverage 是本地后端的运行期依赖：缺少时只跳过覆盖率，不让整次执行失败
        coverage_enabled = request.coverage_enabled
        if coverage_enabled and not _python_module_available(executable, "coverage"):
            coverage_enabled = False
            notes.append(
                "宿主机未安装 coverage，本次已跳过覆盖率采集（pip install coverage 后可用）。"
            )

        commands = build_local_commands(
            test_target=request.test_target,
            coverage_enabled=coverage_enabled,
            executable=executable,
            extra_pytest_args=request.extra_pytest_args,
        )

        with trace_span(
            "sandbox.run",
            kind="sandbox",
            payload={"workspace": str(request.workspace), "target": request.test_target},
            metadata={"backend": "local", "timeout": timeout, "isolated": False},
        ) as span:
            stdout_parts: list[str] = []
            stderr_parts: list[str] = []
            exit_code = 0
            timed_out = False

            for index, command in enumerate(commands):
                # 只有第一条命令（pytest）的退出码代表测试结果；
                # 后续的覆盖率导出失败不应把测试判定为失败
                code, out, err, timed_out = await self._run_one(
                    command, request.workspace, timeout
                )
                stdout_parts.append(out)
                stderr_parts.append(err)
                if index == 0:
                    exit_code = code
                if timed_out:
                    exit_code = -1
                    break

            duration_ms = (time.perf_counter() - started) * 1000
            stdout, truncated_out = _truncate("\n".join(stdout_parts), settings.max_output_bytes)
            stderr, truncated_err = _truncate("\n".join(stderr_parts), settings.max_output_bytes)

            coverage = (
                parse_coverage_json(request.workspace / COVERAGE_JSON_NAME)
                if coverage_enabled
                else None
            )
            counts = parse_pytest_summary(stdout + "\n" + stderr)

            span.set_metadata(exit_code=exit_code, timed_out=timed_out, isolated=False)
            span.set_output(
                {
                    "exit_code": exit_code,
                    "passed": counts["passed"],
                    "failed": counts["failed"],
                }
            )

            return SandboxResult(
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                duration_ms=duration_ms,
                backend="local",
                timed_out=timed_out,
                truncated=truncated_out or truncated_err,
                passed=counts["passed"],
                failed=counts["failed"],
                errors=counts["errors"],
                coverage=coverage,
                command=commands[0],
                notes=notes,
            )

    async def _run_one(
        self, command: list[str], cwd: Path, timeout: int
    ) -> tuple[int, str, str, bool]:
        """执行单条命令，返回 (退出码, stdout, stderr, 是否超时)。"""
        env = {
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
        }
        try:
            completed = await asyncio.to_thread(
                subprocess.run,
                command,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                env=env,
            )
            return completed.returncode, completed.stdout or "", completed.stderr or "", False
        except subprocess.TimeoutExpired as exc:
            logger.warning("本地沙箱执行超时（%s 秒）: %s", timeout, command[:3])
            return -1, _decode(exc.stdout), _decode(exc.stderr), True
        except OSError as exc:
            return -1, "", f"无法启动进程: {exc}", False


def _decode(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


@functools.lru_cache(maxsize=8)
def _python_module_available(executable: str, module: str) -> bool:
    """检测解释器里是否可导入某模块（结果缓存，避免重复起进程）。"""
    try:
        completed = subprocess.run(
            [executable, "-c", f"import {module}"],
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


# ---------------------------------------------------------------------------
# 路径安全与工厂
# ---------------------------------------------------------------------------
def resolve_workspace(settings: DockerSettings, candidate: str | Path) -> Path:
    """把请求中的路径解析为沙箱根目录内的绝对路径。

    这是 API 层最重要的防护：`POST /sandbox/run` 接收的是"代码路径"，
    若不校验，调用方就能让服务读取/执行宿主机任意目录。

    Raises:
        SandboxPathError: 路径越界或不存在。
    """
    root = settings.workspace_path.resolve()
    raw = Path(candidate)
    if not raw.is_absolute():
        raw = root / raw
    # resolve() 会展开符号链接，因此 "workspace/link -> /etc" 这类绕过也会被拦下
    resolved = raw.resolve()

    if resolved != root and root not in resolved.parents:
        raise SandboxPathError(
            f"路径越界：{candidate} 不在沙箱工作区内（{root}）"
        )
    if not resolved.exists():
        raise SandboxPathError(f"路径不存在：{candidate}")
    if not resolved.is_dir():
        raise SandboxPathError(f"路径不是目录：{candidate}")
    return resolved


def docker_available() -> bool:
    """快速探测 Docker daemon 是否可用（不抛异常）。"""
    try:
        import docker

        client = docker.from_env()
        client.ping()
        return True
    except Exception:  # noqa: BLE001
        return False


def get_sandbox_runner(settings: DockerSettings) -> SandboxRunner:
    """按配置选择沙箱后端。

    - `auto`：优先 docker，daemon 不可用时降级 local（并给出警告）
    - `docker`：强制 docker，不可用时报错
    - `local`：强制 local
    """
    backend = settings.backend
    if backend == "local":
        return LocalSubprocessSandboxRunner(settings)
    if backend == "docker":
        return DockerSandboxRunner(settings)
    # auto
    if shutil.which("docker") and docker_available():
        return DockerSandboxRunner(settings)
    logger.warning(
        "Docker 不可用，沙箱降级为本地子进程后端（无隔离）。"
        "请安装并启动 Docker 以获得真正的隔离执行。"
    )
    return LocalSubprocessSandboxRunner(settings)


__all__ = [
    "COVERAGE_JSON_NAME",
    "BackendName",
    "CoverageFileSummary",
    "CoverageSummary",
    "DockerSandboxRunner",
    "LocalSubprocessSandboxRunner",
    "SandboxAvailability",
    "SandboxCommandError",
    "SandboxError",
    "SandboxPathError",
    "SandboxResult",
    "SandboxRunRequest",
    "SandboxRunner",
    "SandboxUnavailable",
    "build_local_commands",
    "build_pytest_script",
    "docker_available",
    "get_sandbox_runner",
    "parse_coverage_json",
    "parse_pytest_summary",
    "resolve_workspace",
]
