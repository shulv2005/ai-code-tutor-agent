"""Agent 编排层：Planner / 代码检索 / 测试生成 / 修复 Agent。

已实现：
- `test_agent`  测试生成 Agent（Step 4）
- `fix_agent`   修复 Agent（Step 6）：失败分类 + 生成 unified diff
- `planner`     反馈闭环编排（Step 6/7）：生成→执行→修复→重跑
- `patch`       补丁提取/校验/应用/回滚 + 隔离工作副本
- `context`     Step 3 检索结果 → CodeContext 的适配层
- `extraction`  LLM 自由文本 → 可运行 Python 代码的解析
- `prompts`     提示词模板（带版本号）

后续步骤：Step 8 PR / Issue 草稿生成。
"""

from app.agents.context import (
    ContextError,
    compute_import_hint,
    context_from_query,
    context_from_snippet,
    context_from_symbol,
    enrich_context,
)
from app.agents.dto import CodeContext, GeneratedTest, TestGenerationResult
from app.agents.extraction import (
    check_generated_test,
    extract_python_code,
    python_test_functions,
    syntax_error,
)
from app.agents.fix_agent import (
    FIX_PROMPT_VERSION,
    FailureContext,
    FixAgent,
    FixAgentError,
    FixProposal,
    parse_analysis,
    parse_category,
)
from app.agents.patch import (
    PatchApplyResult,
    PatchError,
    PatchValidationError,
    apply_patch,
    extract_patch,
    parse_patch_files,
    prepare_worktree,
    remove_worktree,
    revert_patch,
    revert_worktree,
    validate_patch,
    worktree_diff,
)
from app.agents.planner import AutoFixPlanner, AutoFixResult, LoopIteration, failure_signature
from app.agents.prompts import PROMPT_VERSION, build_test_generation_messages
from app.agents.test_agent import (
    GENERATED_TEST_FILENAME,
    SANDBOX_DIR_NAME,
    TestAgent,
    TestAgentError,
    TestGenerationFailed,
)

__all__ = [
    "FIX_PROMPT_VERSION",
    "GENERATED_TEST_FILENAME",
    "PROMPT_VERSION",
    "SANDBOX_DIR_NAME",
    "AutoFixPlanner",
    "AutoFixResult",
    "CodeContext",
    "ContextError",
    "FailureContext",
    "FixAgent",
    "FixAgentError",
    "FixProposal",
    "GeneratedTest",
    "LoopIteration",
    "PatchApplyResult",
    "PatchError",
    "PatchValidationError",
    "TestAgent",
    "TestAgentError",
    "TestGenerationFailed",
    "TestGenerationResult",
    "apply_patch",
    "build_test_generation_messages",
    "check_generated_test",
    "compute_import_hint",
    "context_from_query",
    "context_from_snippet",
    "context_from_symbol",
    "enrich_context",
    "extract_patch",
    "extract_python_code",
    "failure_signature",
    "parse_analysis",
    "parse_category",
    "parse_patch_files",
    "prepare_worktree",
    "python_test_functions",
    "remove_worktree",
    "revert_patch",
    "revert_worktree",
    "syntax_error",
    "validate_patch",
    "worktree_diff",
]
