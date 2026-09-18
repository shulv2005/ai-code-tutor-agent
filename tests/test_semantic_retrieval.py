"""语义检索质量测试：用真实嵌入模型（fastembed / BGE）验证中文查询 → 代码符号。

这是整个检索层的核心价值主张，因此必须有真实模型的用例兜底，
而不是只靠 hashing 后端的确定性测试。

模型缺失或网络不可达时自动 skip（不 fail），保证 CI 在离线环境仍为绿。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.api.deps import reset_retrieval_service
from app.core.config import RetrievalSettings, get_settings
from app.services.retrieval.embedder import build_embedder, reset_shared_embedders

REPO_URL = "https://github.com/local/sample.git"


def _fastembed_or_skip() -> None:
    """探测 fastembed 是否真的可用；不可用则 skip。"""
    provider = build_embedder(RetrievalSettings())
    if provider.name != "fastembed":
        pytest.skip(
            "fastembed 不可用（模型未缓存且无法下载）。"
            "首次使用需联网拉取模型，或设置 HF_ENDPOINT 镜像 + HF_HUB_DISABLE_XET=1"
        )


@pytest.fixture(scope="module")
def embeddings() -> object:
    """模块级共享的真实嵌入模型（加载一次，约 0.5s）。"""
    _fastembed_or_skip()
    provider = build_embedder(RetrievalSettings())
    return provider


DOCUMENTS = [
    "add(a: int, b: int) -> int\n两个数相加并返回结果",
    "fetch(url: str, retries: int = 3) -> str\n异步拉取远端数据，失败时自动重试",
    "class UserRepository\n负责用户表的增删改查，封装数据库会话",
    "parse_config(path)\n读取 YAML 配置文件并校验必填字段",
]


@pytest.mark.parametrize(
    ("query", "expected_index"),
    [
        ("两个数字相加的函数", 0),
        ("怎么重试网络请求", 1),
        ("在哪里查询用户数据", 2),
        ("解析配置文件的地方", 3),
    ],
)
def test_chinese_query_retrieves_correct_symbol(
    embeddings: object, query: str, expected_index: int
) -> None:
    """中文自然语言查询应命中语义对应的代码符号（rank-1）。"""
    doc_vectors = embeddings.embed_documents(DOCUMENTS)  # type: ignore[attr-defined]
    query_vector = embeddings.embed_query(query)  # type: ignore[attr-defined]

    similarities = doc_vectors @ query_vector
    best = int(np.argmax(similarities))
    assert best == expected_index, (
        f"查询「{query}」期望命中 [{expected_index}]，实际 [{best}]；"
        f"相似度={np.round(similarities, 4).tolist()}"
    )


def test_vectors_are_normalized(embeddings: object) -> None:
    """归一化后内积才等价于余弦相似度（FAISS IndexFlatIP 的前提）。"""
    vectors = embeddings.embed_documents(DOCUMENTS)  # type: ignore[attr-defined]
    norms = np.linalg.norm(vectors, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-4)
    assert vectors.shape[1] == 512


def test_semantic_beats_pure_keyword_overlap(embeddings: object) -> None:
    """语义模型的价值：查询词与文档零字面重叠时仍能命中。"""
    docs = ["fetch_user_data 从远端拉取用户信息", "class Widget 渲染界面组件"]
    vectors = embeddings.embed_documents(docs)  # type: ignore[attr-defined]
    # 「网络请求」与文档中的 "fetch"/"远端" 无字面重叠
    query_vector = embeddings.embed_query("网络请求")  # type: ignore[attr-defined]

    similarities = vectors @ query_vector
    assert int(np.argmax(similarities)) == 0


# ---------------------------------------------------------------------------
# 端到端：真实模型 + 完整检索链路
# ---------------------------------------------------------------------------
# 说明：这里用「带中文 docstring」的仓库做端到端断言。
# 实测 bge-small-zh-v1.5 的跨语言能力有限（中文查询 -> 纯英文文档 的 rank-1
# 命中率约 40%），而中文 docstring 场景 rank-1 接近 100%。中文开源项目与
# 带中文注释的仓库正是本项目的主要场景，因此用它作为端到端基线；
# 跨语言能力受限于模型规模，需要更强模型时可换 intfloat/multilingual-e5-large。
CHINESE_SOURCES = {
    "auth.py": '''\
def verify_token(token: str) -> bool:
    """校验用户令牌是否有效，过期则返回假。"""
    if not token:
        return False
    return len(token) > 8


def hash_password(raw: str) -> str:
    """把明文密码做哈希，用于安全存储。"""
    return raw[::-1]
''',
    "order.py": '''\
def create_order(user_id: int, items: list) -> dict:
    """创建订单，计算总价并写入数据库。"""
    total = sum(item["price"] for item in items)
    return {"user_id": user_id, "total": total}


def cancel_order(order_id: int) -> bool:
    """取消订单并退回库存。"""
    return True
''',
    "http_client.py": '''\
async def request_with_retry(url: str, retries: int = 3) -> str:
    """发送网络请求，失败时自动重试。"""
    for _ in range(retries):
        pass
    return url
''',
}


@pytest.fixture()
def chinese_repo(tmp_path: Path) -> Path:
    """构造一个带中文 docstring 的本地仓库。"""
    import git

    root = tmp_path / "zh_src"
    root.mkdir()
    for filename, source in CHINESE_SOURCES.items():
        (root / filename).write_text(source, encoding="utf-8")
    repo = git.Repo.init(root, initial_branch="main")
    repo.index.add(list(CHINESE_SOURCES))
    actor = git.Actor("Tester", "tester@example.com")
    repo.index.commit("init", author=actor, committer=actor)
    repo.close()
    return root


@pytest.fixture()
def chinese_clone(monkeypatch: pytest.MonkeyPatch, chinese_repo: Path) -> None:
    """把 URL 解析指向中文样例仓库。"""
    from app.services.repo.git_service import RepoRef

    def fake_parse(url: str, allowed_hosts: object) -> RepoRef:
        return RepoRef(
            host="github.com",
            owner="local",
            name="zh",
            clone_url=str(chinese_repo),
            sanitized_url="https://github.com/local/zh.git",
        )

    monkeypatch.setattr("app.services.repo.service.parse_repo_url", fake_parse)


@pytest.fixture()
def semantic_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """把检索切换到真实语义后端，其余配置指向临时目录。"""
    _fastembed_or_skip()
    monkeypatch.setenv("RETRIEVAL__INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("RETRIEVAL__EMBEDDER", "fastembed")
    get_settings.cache_clear()
    reset_retrieval_service()
    reset_shared_embedders()
    yield
    get_settings.cache_clear()
    reset_retrieval_service()
    reset_shared_embedders()


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("怎么校验用户令牌", "verify_token"),
        ("密码如何加密存储", "hash_password"),
        ("创建订单并计算总价", "create_order"),
        ("取消订单", "cancel_order"),
        ("网络请求失败重试", "request_with_retry"),
    ],
)
def test_semantic_search_end_to_end(
    sqlite_path: Path,
    repo_workspace: Path,
    chinese_clone: None,
    semantic_env: None,
    query: str,
    expected: str,
) -> None:
    """真实模型跑通「建索引 -> 中文查询 -> 命中正确实现函数」。"""
    from fastapi.testclient import TestClient

    from app.main import create_app

    app = create_app()
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/repositories", json={"url": "https://github.com/local/zh.git"}
        )
        assert created.status_code == 201, created.text
        repo_id = created.json()["repository"]["id"]

        built = client.post(f"/api/v1/repositories/{repo_id}/index")
        assert built.status_code == 200, built.text
        info = built.json()
        assert info["embedder"] == "fastembed"
        assert info["dimension"] == 512
        assert info["chunk_count"] == 5

        hits = client.post(
            "/api/v1/search", json={"repository_id": repo_id, "query": query, "top_k": 3}
        ).json()["hits"]

    assert hits, "语义检索不应返回空结果"
    assert hits[0]["qualified_name"] == expected, (
        f"查询「{query}」期望命中 {expected}，实际 {[h['qualified_name'] for h in hits]}"
    )
    assert hits[0]["code"], "命中必须带源码片段供 Agent 使用"
