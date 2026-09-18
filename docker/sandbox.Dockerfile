# 沙箱执行镜像：预装 pytest + coverage，使运行期可以完全断网。
#
# 为什么需要它：
# 直接用 python:3.10-slim 跑测试时容器内没有 pytest/coverage，只能启动时
# pip install —— 而那一步必须联网，会迫使 DOCKER__NETWORK_DISABLED=false，
# 削弱隔离强度。预构建镜像可以做到「装依赖时联网、跑测试时断网」。
#
# 构建：
#   docker build -f docker/sandbox.Dockerfile -t opensource-agent-sandbox:py3.10 .
#
# 构建后设置：
#   DOCKER__IMAGE=opensource-agent-sandbox:py3.10
#   DOCKER__INSTALL_DEPENDENCIES=false
#   DOCKER__NETWORK_DISABLED=true
#   DOCKER__SANDBOX_USER=1000:1000     # 配合预先 chown 工作目录

FROM python:3.10-slim

# 固定版本，保证不同时间构建出的沙箱行为一致（可复现）
ARG PYTEST_VERSION=8.3.5
ARG COVERAGE_VERSION=7.6.12

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/tmp

RUN python -m pip install --no-cache-dir \
        "pytest==${PYTEST_VERSION}" \
        "coverage==${COVERAGE_VERSION}" \
    && python -m pip uninstall -y pip setuptools wheel \
    && rm -rf /root/.cache /var/lib/apt/lists/*

# 非 root 用户：容器内以 uid 1000 运行，进一步缩小爆炸半径。
# 注意：绑定的工作目录需要宿主机侧可写（Linux 上通常 chown 1000:1000）。
RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin runner

# 沙箱内的工作目录挂载点（与 DOCKER__WORKSPACE_MOUNT 对应）
WORKDIR /workspace

# 容器本身不做任何事：命令由运行时的 docker SDK 传入
CMD ["python", "--version"]
