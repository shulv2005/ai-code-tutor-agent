"""配置管理单元测试。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.core.config import PROJECT_ROOT, Settings, get_settings

ENV_PREFIXES = ("APP__", "DATABASE__", "LLM__", "RETRIEVAL__", "DOCKER__", "TRACING__")


@pytest.fixture()
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """清理可能存在的项目环境变量，保证默认值可预测。"""
    for key in list(os.environ):
        if key.startswith(ENV_PREFIXES):
            monkeypatch.delenv(key, raising=False)


def test_defaults_load_without_env(clean_env: None) -> None:
    settings = Settings(_env_file=None)

    assert settings.app.name == "opensource-collab-agent"
    assert settings.app.api_v1_prefix == "/api/v1"
    assert settings.database.url.startswith("sqlite+aiosqlite:///")
    assert settings.retrieval.vector_backend in {"faiss", "chroma"}
    assert settings.docker.network_disabled is True


def test_nested_env_override(clean_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db_path = tmp_path / "custom.db"
    monkeypatch.setenv("DATABASE__SQLITE_PATH", str(db_path))
    monkeypatch.setenv("LLM__MODEL", "qwen2.5-coder")
    monkeypatch.setenv("DOCKER__TIMEOUT_SECONDS", "42")

    settings = Settings(_env_file=None)

    assert settings.database.sqlite_path == db_path
    assert db_path.as_posix() in settings.database.url
    assert settings.llm.model == "qwen2.5-coder"
    assert settings.docker.timeout_seconds == 42


def test_api_key_is_secret(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM__API_KEY", "sk-super-secret")
    settings = Settings(_env_file=None)

    assert settings.llm.api_key.get_secret_value() == "sk-super-secret"
    assert "sk-super-secret" not in repr(settings.llm.api_key)


def test_cors_origins_accept_comma_separated(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP__CORS_ORIGINS", "http://a.com, http://b.com")
    settings = Settings(_env_file=None)

    assert settings.app.cors_origins == ["http://a.com", "http://b.com"]


def test_url_override_takes_precedence(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE__URL_OVERRIDE", "postgresql+asyncpg://user:pwd@db:5432/agent")
    settings = Settings(_env_file=None)

    assert settings.database.url == "postgresql+asyncpg://user:pwd@db:5432/agent"


def test_directories_are_created(clean_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATABASE__SQLITE_PATH", str(tmp_path / "nested" / "app.db"))
    monkeypatch.setenv("RETRIEVAL__INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("DOCKER__WORKSPACE_DIR", str(tmp_path / "ws"))

    settings = Settings(_env_file=None)
    settings.ensure_directories()

    assert (tmp_path / "nested").is_dir()
    assert (tmp_path / "index").is_dir()
    assert (tmp_path / "ws").is_dir()


def test_get_settings_is_cached(clean_env: None) -> None:
    get_settings.cache_clear()

    assert get_settings() is get_settings()
    assert get_settings().app.version == "0.1.0"

    get_settings.cache_clear()


def test_project_root_points_to_repository_root() -> None:
    assert (PROJECT_ROOT / "app" / "main.py").is_file()

