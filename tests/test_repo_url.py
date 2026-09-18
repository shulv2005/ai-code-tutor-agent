"""仓库地址校验测试：这是面向"用户提交任意 URL"的安全边界。"""

from __future__ import annotations

import pytest

from app.services.repo.git_service import InvalidRepoUrlError, parse_repo_url

ALLOWED = ["github.com", "gitee.com"]


@pytest.mark.parametrize(
    ("url", "owner", "name", "scheme"),
    [
        ("https://github.com/psf/requests.git", "psf", "requests", "https"),
        ("https://github.com/psf/requests", "psf", "requests", "https"),
        ("https://github.com/psf/requests/", "psf", "requests", "https"),
        ("http://gitee.com/mindspore/mindspore.git", "mindspore", "mindspore", "http"),
        ("git@github.com:psf/requests.git", "psf", "requests", "https"),
    ],
)
def test_valid_urls_are_normalized(url: str, owner: str, name: str, scheme: str) -> None:
    ref = parse_repo_url(url, ALLOWED)
    assert (ref.owner, ref.name) == (owner, name)
    # 归一化成规范形式（保留调用方指定的协议；scp 风格统一转 https）
    assert ref.sanitized_url == f"{scheme}://{ref.host}/{owner}/{name}.git"


def test_ssh_style_url_is_converted_to_https() -> None:
    ref = parse_repo_url("git@github.com:psf/requests.git", ALLOWED)
    assert ref.clone_url.startswith("https://")
    assert ref.host == "github.com"


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "file://C:/Windows/System32",
        "ftp://github.com/a/b.git",
        "javascript:alert(1)",
        "https://evil.com/a/b.git",  # 主机不在白名单
        "/local/path/to/repo",
        "C:\\Users\\me\\repo",
    ],
)
def test_unsafe_urls_are_rejected(url: str) -> None:
    """挡掉本地文件读取与 SSRF：非 http(s) 协议与白名单外主机一律拒绝。"""
    with pytest.raises(InvalidRepoUrlError):
        parse_repo_url(url, ALLOWED)


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/../../etc/passwd",
        "https://github.com/onlyowner",
        "https://github.com/owner/..%2f..%2fetc",
        "https://github.com/a/b/../../../c",
    ],
)
def test_path_traversal_is_rejected(url: str) -> None:
    """owner/name 会拼进本地目录名，必须挡住目录穿越。"""
    with pytest.raises(InvalidRepoUrlError):
        parse_repo_url(url, ALLOWED)


def test_empty_url_is_rejected() -> None:
    with pytest.raises(InvalidRepoUrlError):
        parse_repo_url("   ", ALLOWED)


def test_credentials_are_stripped_from_sanitized_url() -> None:
    """带 token 的地址：克隆用带凭据的，入库/日志用脱敏的。"""
    ref = parse_repo_url("https://user:token123@github.com/psf/requests.git", ALLOWED)
    assert "token123" not in ref.sanitized_url
    assert "token123" in ref.clone_url


def test_slug_is_filesystem_safe() -> None:
    ref = parse_repo_url("https://github.com/psf/requests.git", ALLOWED)
    assert ref.slug == "github.com__psf__requests"
    assert "/" not in ref.slug and "\\" not in ref.slug and ".." not in ref.slug
