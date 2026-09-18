"""Agent 层 DTO：代码上下文与生成结果的稳定契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.core.llm_client import LLMUsage


@dataclass(slots=True)
class CodeContext:
    """待测代码的上下文（Step 3 检索结果 → Step 4 输入）。"""

    path: str
    code: str
    language: str = "python"
    qualified_name: str = ""
    kind: str = "function"
    signature: str = ""
    docstring: str | None = None
    start_line: int = 0
    end_line: int = 0
    # 同文件/同模块的相关符号摘要，帮助模型理解调用约定
    related: list[str] = field(default_factory=list)
    # 仓库现有测试片段，作为风格参考（Step 3 的 include_tests=True）
    existing_tests: list[str] = field(default_factory=list)
    repository: str | None = None
    symbol_id: int | None = None
    # 被测模块的导入语句，例如 `from pkg.core import add`。
    # 关键：修复循环里测试必须导入**真实模块**而不是内联替身，
    # 否则对源码打补丁不会影响测试结果，整个反馈闭环失效。
    import_hint: str | None = None
    # 沙箱里是否有完整仓库（有则可安全依赖真实导入）
    module_available: bool = False

    @property
    def display_name(self) -> str:
        return self.qualified_name or Path(self.path).stem


@dataclass(slots=True)
class GeneratedTest:
    """一次测试生成的结果。"""

    code: str
    model: str = ""
    attempts: int = 1
    usage: LLMUsage = field(default_factory=LLMUsage)
    # 解析阶段发现的问题（如"未检测到代码块"、"内容被截断"）
    warnings: list[str] = field(default_factory=list)
    test_functions: list[str] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass(slots=True)
class TestGenerationResult:
    """测试生成任务的完整结果。"""

    # 类名以 Test 开头，显式告知 pytest 这不是测试类
    __test__ = False

    run_id: str
    target: CodeContext
    test: GeneratedTest
    saved_path: Path | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    trace_id: str | None = None


__all__ = ["CodeContext", "GeneratedTest", "TestGenerationResult"]
