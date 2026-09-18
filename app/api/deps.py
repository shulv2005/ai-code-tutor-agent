"""FastAPI 依赖注入：数据库会话、全局配置与领域服务。"""

from __future__ import annotations

import threading
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.fix_agent import FixAgent
from app.agents.planner import AutoFixPlanner
from app.agents.test_agent import TestAgent
from app.agents.tutor_agent import TutorAgent
from app.core.api_key_manager import ApiKeyManager
from app.core.api_key_manager import get_api_key_manager as _get_api_key_manager
from app.core.config import Settings, get_settings
from app.core.database import get_db
from app.core.llm_client import get_llm_client
from app.services.code_checker import CodeChecker
from app.services.code_fixer import CodeFixer
from app.services.comment_generator import CommentGenerator
from app.services.file_classifier import FileClassifier
from app.services.library_service import LibraryService
from app.services.repo.service import RepositoryService
from app.services.retrieval.service import RetrievalService
from app.services.sandbox import SandboxRunner, get_sandbox_runner

DbSession = Annotated[AsyncSession, Depends(get_db)]
AppSettingsDep = Annotated[Settings, Depends(get_settings)]


def get_api_key_manager() -> ApiKeyManager:
    """API Key 内存保管箱（进程内单例）。

    为什么必须是单例：Key 就存在这个对象的内存字典里，
    按请求新建的话"刚存进去的 Key"下一次请求就找不到了。

    测试里可以用 `reset_api_key_manager()` 清空，避免用例之间互相影响。
    """
    return _get_api_key_manager()


ApiKeyManagerDep = Annotated[ApiKeyManager, Depends(get_api_key_manager)]


def get_repository_service(settings: AppSettingsDep) -> RepositoryService:
    """仓库解析服务（无状态，可直接按请求构造）。"""
    return RepositoryService(settings)


RepoServiceDep = Annotated[RepositoryService, Depends(get_repository_service)]

# 检索服务必须进程内单例：它持有内存索引缓存与嵌入模型句柄，
# 按请求重建会导致每次检索都重新加载 ONNX 模型与 FAISS 索引。
_retrieval_service: RetrievalService | None = None
_retrieval_lock = threading.Lock()


def get_retrieval_service(settings: AppSettingsDep) -> RetrievalService:
    """检索服务（进程内单例）。"""
    global _retrieval_service
    if _retrieval_service is None:
        with _retrieval_lock:
            if _retrieval_service is None:
                _retrieval_service = RetrievalService(settings)
    return _retrieval_service


def reset_retrieval_service() -> None:
    """清空单例（测试或多环境切换时使用）。"""
    global _retrieval_service
    with _retrieval_lock:
        _retrieval_service = None


RetrievalServiceDep = Annotated[RetrievalService, Depends(get_retrieval_service)]


def get_test_agent(settings: AppSettingsDep) -> TestAgent:
    """测试生成 Agent。

    LLM 客户端走进程内单例（httpx 连接池复用）；测试可用
    `app.dependency_overrides[get_test_agent]` 注入假 LLM。
    """
    return TestAgent(settings, get_llm_client())


TestAgentDep = Annotated[TestAgent, Depends(get_test_agent)]


def get_sandbox(settings: AppSettingsDep) -> SandboxRunner:
    """沙箱执行后端（auto 模式下每次按 Docker 可用性选择）。"""
    return get_sandbox_runner(settings.docker)


SandboxRunnerDep = Annotated[SandboxRunner, Depends(get_sandbox)]


def get_fix_agent(settings: AppSettingsDep) -> FixAgent:
    """修复 Agent。"""
    return FixAgent(settings, get_llm_client())


FixAgentDep = Annotated[FixAgent, Depends(get_fix_agent)]


def get_autofix_planner(
    settings: AppSettingsDep,
    test_agent: TestAgentDep,
    fix_agent: FixAgentDep,
    sandbox: SandboxRunnerDep,
) -> AutoFixPlanner:
    """自动修复编排器（组装 Test Agent / Fix Agent / Sandbox）。"""
    return AutoFixPlanner(
        settings, test_agent=test_agent, fix_agent=fix_agent, sandbox=sandbox
    )


AutoFixPlannerDep = Annotated[AutoFixPlanner, Depends(get_autofix_planner)]


def get_tutor_agent(settings: AppSettingsDep) -> TutorAgent:
    """AI 代码导师 Agent（检测 / 注释 / 改错）。

    与其它 Agent 共用同一个进程级 LLM 客户端；
    测试可用 `app.dependency_overrides[get_tutor_agent]` 注入假客户端。
    """
    return TutorAgent(settings, get_llm_client())


TutorAgentDep = Annotated[TutorAgent, Depends(get_tutor_agent)]


def get_library(settings: AppSettingsDep) -> LibraryService:
    """本地项目库服务。

    服务本身无状态（只持有配置），按请求构造即可；
    从 settings 现取现用，测试里覆盖 LIBRARY__ROOTS 后立刻生效。
    """
    return LibraryService(settings.library)


LibraryServiceDep = Annotated[LibraryService, Depends(get_library)]


def get_file_classifier(settings: AppSettingsDep) -> FileClassifier:
    """本地代码文件分类器（扫描 + 按语言归档）。

    无状态（只持有配置），按请求构造即可；从 settings 现取现用，
    测试里覆盖 CLASSIFIER__ROOT 后立刻生效。
    """
    return FileClassifier(settings.classifier)


FileClassifierDep = Annotated[FileClassifier, Depends(get_file_classifier)]


def get_code_checker(settings: AppSettingsDep) -> CodeChecker:
    """AI 自动检测器（本地静态检查 + 大模型深度检测）。

    与其它 Agent 共用同一个进程级 LLM 客户端（连接池复用）；
    测试可用 `app.dependency_overrides[get_code_checker]` 注入假客户端。
    """
    return CodeChecker(settings, get_llm_client())


CodeCheckerDep = Annotated[CodeChecker, Depends(get_code_checker)]


def get_code_fixer(settings: AppSettingsDep) -> CodeFixer:
    """代码改错器（本地分析 + AI 修正 + 本地复检）。

    内部复用 `CodeChecker` 的本地检查能力，保证「检测」与「改错」对
    "什么算问题"的判断完全一致；两者共用同一个进程级 LLM 客户端。
    测试可用 `app.dependency_overrides[get_code_fixer]` 注入假客户端。
    """
    return CodeFixer(settings, get_llm_client())


CodeFixerDep = Annotated[CodeFixer, Depends(get_code_fixer)]


def get_comment_generator(settings: AppSettingsDep) -> CommentGenerator:
    """代码注释生成器（本地分析 + AI 写注释 + 本地复检）。

    内部复用 `CodeChecker` 的本地解析能力，保证三个学习类功能
    （检测/改错/注释）对同一份代码的结构理解一致。
    测试可用 `app.dependency_overrides[get_comment_generator]` 注入假客户端。
    """
    return CommentGenerator(settings, get_llm_client())


CommentGeneratorDep = Annotated[CommentGenerator, Depends(get_comment_generator)]


__all__ = [
    "ApiKeyManagerDep",
    "AppSettingsDep",
    "AutoFixPlannerDep",
    "CodeCheckerDep",
    "CodeFixerDep",
    "CommentGeneratorDep",
    "DbSession",
    "FileClassifierDep",
    "FixAgentDep",
    "LibraryServiceDep",
    "RepoServiceDep",
    "RetrievalServiceDep",
    "SandboxRunnerDep",
    "TestAgentDep",
    "TutorAgentDep",
    "get_autofix_planner",
    "get_api_key_manager",
    "get_code_checker",
    "get_code_fixer",
    "get_comment_generator",
    "get_file_classifier",
    "get_fix_agent",
    "get_library",
    "get_repository_service",
    "get_retrieval_service",
    "get_sandbox",
    "get_test_agent",
    "get_tutor_agent",
    "reset_retrieval_service",
]

