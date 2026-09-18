"""Agent 模块的 API 契约。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TestGenerationRequest(BaseModel):
    """测试生成请求。

    支持三种输入方式（按优先级）：
    1. 直接给代码片段：`code`（+ 可选 `file_path` / `symbol_name`）
    2. 从索引按符号取：`repository_id` + `symbol_id`
    3. 用自然语言检索定位：`repository_id` + `query`
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "code": "def add(a: int, b: int) -> int:\n    return a + b\n",
                "file_path": "pkg/core.py",
                "symbol_name": "add",
            }
        }
    )

    # --- 输入方式 1：直接给代码 ---
    code: str | None = Field(
        default=None, max_length=200_000, description="待测代码片段（直接给定）"
    )
    file_path: str = Field(default="snippet.py", max_length=1024, description="代码所属文件路径")
    symbol_name: str | None = Field(default=None, max_length=512, description="符号名")
    language: str = Field(default="python", max_length=64, description="语言，当前支持 python")

    # --- 输入方式 2/3：从仓库索引取 ---
    repository_id: int | None = Field(default=None, ge=1, description="仓库 ID")
    symbol_id: int | None = Field(default=None, ge=1, description="代码符号 ID")
    query: str | None = Field(
        default=None, max_length=2000, description="自然语言检索，用于定位待测代码"
    )

    # --- 行为控制 ---
    max_attempts: int = Field(
        default=2, ge=1, le=5, description="最大尝试次数（校验失败会带反馈重试）"
    )
    save_to_sandbox: bool = Field(
        default=True, description="是否把生成的测试写入沙箱目录"
    )
    include_related: bool = Field(
        default=True, description="是否附上同文件相关符号作为上下文"
    )
    include_existing_tests: bool = Field(
        default=True, description="是否附上仓库现有测试作为风格参考"
    )

    @model_validator(mode="after")
    def _require_input(self) -> TestGenerationRequest:
        has_code = bool(self.code and self.code.strip())
        has_symbol = self.repository_id is not None and self.symbol_id is not None
        has_query = self.repository_id is not None and bool(self.query and self.query.strip())
        if not (has_code or has_symbol or has_query):
            raise ValueError(
                "必须提供以下之一：code；repository_id + symbol_id；repository_id + query"
            )
        return self


class CodeContextRead(BaseModel):
    """实际用于生成代码的上下文（便于调用方核对）。"""

    path: str
    qualified_name: str
    kind: str
    language: str
    signature: str
    docstring: str | None = None
    start_line: int
    end_line: int
    repository: str | None = None
    symbol_id: int | None = None
    related_count: int = 0
    existing_test_count: int = 0


class LLMUsageRead(BaseModel):
    """token 用量。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class TestGenerationResponse(BaseModel):
    """测试生成结果。"""

    run_id: str
    target: CodeContextRead
    test_code: str
    test_functions: list[str] = Field(default_factory=list)
    model: str = ""
    prompt_version: str = ""
    attempts: int = 1
    warnings: list[str] = Field(default_factory=list)
    usage: LLMUsageRead = Field(default_factory=LLMUsageRead)
    saved_path: str | None = Field(default=None, description="沙箱内生成测试的绝对路径")
    sandbox_dir: str | None = Field(default=None, description="本次运行的沙箱仓库根目录")
    timings_ms: dict[str, float] = Field(default_factory=dict)
    trace_id: str | None = None


class LLMStatusResponse(BaseModel):
    """LLM 配置状态：便于前端提示"需要先配置模型"。"""

    configured: bool
    model: str
    base_url: str
    provider: str
    max_retries: int
    timeout_seconds: int
    prompt_version: str


# ---------------------------------------------------------------------------
# 自动修复（Step 6/7）
# ---------------------------------------------------------------------------
class AutoFixRequest(BaseModel):
    """自动修复请求：接收仓库 URL 与 Issue 描述，触发完整链路。

    会依次执行：克隆 → 解析 → 建检索索引 → 定位代码 → 生成测试 →
    沙箱执行 → 失败则生成补丁并重跑，最多 max_attempts 轮。
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "repository_url": "https://gitee.com/mirrors/requests.git",
                "issue": "取消订单时没有校验状态，已取消的订单还能再次取消",
                "max_attempts": 3,
            }
        }
    )

    # 目标仓库：给 URL（自动克隆并索引）或给已注册的 repository_id
    repository_url: str | None = Field(default=None, max_length=512)
    repository_id: int | None = Field(default=None, ge=1)

    # 定位方式：Issue 描述（走混合检索）或直接指定符号
    issue: str | None = Field(default=None, max_length=4000, description="Issue 描述")
    symbol_id: int | None = Field(default=None, ge=1, description="直接指定待修复符号")

    max_attempts: int = Field(default=3, ge=1, le=5, description="最大修复轮次")
    keep_worktree: bool = Field(
        default=True, description="结束后是否保留隔离工作副本（便于查看 diff）"
    )
    test_max_attempts: int = Field(default=2, ge=1, le=5, description="测试生成的最大尝试次数")

    @model_validator(mode="after")
    def _require_target(self) -> AutoFixRequest:
        if self.repository_id is None and not (
            self.repository_url and self.repository_url.strip()
        ):
            raise ValueError("必须提供 repository_url 或 repository_id 之一")
        if self.symbol_id is None and not (self.issue and self.issue.strip()):
            raise ValueError("必须提供 issue 描述或 symbol_id 之一")
        return self


class LoopIterationRead(BaseModel):
    """一轮修复循环的记录。"""

    index: int
    passed: int = 0
    failed: int = 0
    errors: int = 0
    exit_code: int | None = None
    category: str | None = Field(default=None, description="Fix Agent 的失败归类")
    analysis: str | None = Field(default=None, description="Fix Agent 的根因分析")
    patch_files: list[str] = Field(default_factory=list)
    patch_applied: bool = False
    patch_error: str | None = None
    note: str = ""


class PrDraftRead(BaseModel):
    """PR 草稿。"""

    title: str
    body: str
    branch_name: str
    commit_message: str
    base_branch: str
    files_changed: list[str] = Field(default_factory=list)
    additions: int = 0
    deletions: int = 0
    is_draft: bool = True
    labels: list[str] = Field(default_factory=list)
    verified: bool = Field(
        default=False, description="是否基于一次真正成功的修复（未成功时为 false）"
    )
    warnings: list[str] = Field(default_factory=list)


class AutoFixResponse(BaseModel):
    """自动修复结果。"""

    run_id: str
    status: Literal[
        "passed",
        "max_attempts",
        "no_patch",
        "not_patchable",
        "patch_rejected",
        "no_progress",
        "error",
    ]
    success: bool
    attempts: int
    message: str

    target: CodeContextRead | None = None
    generated_test: str = ""
    final_diff: str = Field(default="", description="工作副本相对基线的完整 diff")
    changed_files: list[str] = Field(default_factory=list)
    worktree: str | None = None

    passed: int = 0
    failed: int = 0
    errors: int = 0
    coverage_percent: float | None = None

    iterations: list[LoopIterationRead] = Field(default_factory=list)
    # 链路终点的产物：可直接提交的 PR 草稿与 Issue 评论
    pr_draft: PrDraftRead | None = None
    issue_comment: str = ""
    timings_ms: dict[str, float] = Field(default_factory=dict)
    trace_id: str | None = None
