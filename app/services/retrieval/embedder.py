"""嵌入模型层：可插拔的 EmbeddingProvider。

为什么要做成可插拔：嵌入模型是整个检索链路里**唯一的外部重依赖**
（要下载数百 MB 权重、依赖特定网络）。如果把它写死，模型一拉不下来整条
检索链就废了。因此：
- `FastEmbedProvider`：生产默认，ONNX Runtime 推理，真语义检索。
- `HashingEmbedder`：零依赖确定性哈希，离线/CI/模型不可用时兜底。
  质量不如语义模型，但保证「索引与检索始终可用」。

实测踩过的两个网络坑（已在 _configure_hf_env 里处理）：
1. huggingface.co 在部分网络不可达 -> 必须走 HF_ENDPOINT 镜像。
2. huggingface-hub 1.x 默认使用 Xet 协议（cas-server.xethub.hf.co），
   镜像站不代理该域名 -> 401 Unauthorized，权重永远下不来。
   必须设 HF_HUB_DISABLE_XET=1 回退普通 HTTP。
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import threading
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from app.core.config import RetrievalSettings
from app.services.retrieval.tokenizer import tokenize_code

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingProvider(Protocol):
    """嵌入模型统一接口。"""

    @property
    def name(self) -> str:
        """后端标识（fastembed / hashing）。"""
        ...

    @property
    def model_id(self) -> str:
        """模型标识（用于索引元信息比对）。"""
        ...

    @property
    def dimension(self) -> int:
        """向量维度。"""
        ...

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """批量嵌入文档，返回 (n, dim) 的 float32 数组。"""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """嵌入单条查询，返回 (dim,) 的 float32 数组。"""
        ...


def _configure_hf_env(settings: RetrievalSettings) -> None:
    """在导入 fastembed / huggingface_hub 之前设置镜像环境变量。

    必须在导入前设置：huggingface_hub 在导入时读取这些变量并缓存。
    """
    if settings.hf_endpoint:
        os.environ.setdefault("HF_ENDPOINT", settings.hf_endpoint)
    if settings.hf_disable_xet:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HOME", str(settings.cache_path))


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """按行做 L2 归一化，使内积等价于余弦相似度（配合 FAISS IndexFlatIP）。"""
    matrix = np.asarray(matrix, dtype="float32")
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # 零向量保护：全 0 会让归一化产生 NaN
    np.maximum(norms, 1e-12, out=norms)
    return (matrix / norms).astype("float32")


class FastEmbedProvider:
    """基于 fastembed（ONNX Runtime）的语义嵌入。"""

    def __init__(self, settings: RetrievalSettings) -> None:
        self._settings = settings
        self._model: object | None = None
        self._dimension = settings.embedding_dim
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "fastembed"

    @property
    def model_id(self) -> str:
        return self._settings.embedding_model

    @property
    def dimension(self) -> int:
        self._ensure_model()
        return self._dimension

    def _ensure_model(self) -> object:
        """懒加载模型（线程安全）：首次调用会下载权重，之后走本地缓存。"""
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            _configure_hf_env(self._settings)
            from fastembed import TextEmbedding  # 延迟导入：未安装时不影响其它后端

            self._settings.cache_path.mkdir(parents=True, exist_ok=True)
            model = TextEmbedding(
                model_name=self._settings.embedding_model,
                cache_dir=str(self._settings.cache_path),
            )
            size = getattr(model, "embedding_size", None)
            if isinstance(size, int) and size > 0:
                self._dimension = size
            self._model = model
            logger.info(
                "已加载嵌入模型: %s (dim=%d)", self._settings.embedding_model, self._dimension
            )
            return model

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        model = self._ensure_model()
        # passage_embed 使用文档侧提示词；BGE 系列检索质量强依赖 query/passage 区分
        embed = getattr(model, "passage_embed", None) or model.embed  # type: ignore[attr-defined]
        vectors = list(embed(list(texts)))
        if not vectors:
            return np.zeros((0, self._dimension), dtype="float32")
        return _l2_normalize(np.vstack(vectors))

    def embed_query(self, text: str) -> np.ndarray:
        model = self._ensure_model()
        embed = getattr(model, "query_embed", None) or model.embed  # type: ignore[attr-defined]
        vector = np.asarray(next(iter(embed([text]))), dtype="float32")
        return _l2_normalize(vector)[0]


class HashingEmbedder:
    """确定性特征哈希嵌入：零外部依赖，离线可用。

    用「token -> 桶 + 符号」的哈希技巧 + 次线性词频，得到与词重叠度正相关的
    余弦相似度。质量低于语义模型，但足以支撑：
    - CI / 离线环境跑通完整检索链路
    - 嵌入模型不可用时的优雅降级
    """

    _TOKEN = re.compile(r"\S+")

    def __init__(self, dimension: int = 512) -> None:
        self._dimension = max(int(dimension), 8)

    @property
    def name(self) -> str:
        return "hashing"

    @property
    def model_id(self) -> str:
        return f"hashing-{self._dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def _bucket(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        sign = 1.0 if (value >> 63) & 1 else -1.0
        return value % self._dimension, sign

    def _embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self._dimension, dtype="float32")
        tokens = tokenize_code(text) or self._TOKEN.findall(text.lower())
        if not tokens:
            return vector
        counts: dict[int, float] = {}
        for token in tokens:
            index, sign = self._bucket(token)
            counts[index] = counts.get(index, 0.0) + sign
        for index, raw in counts.items():
            # 次线性词频：抑制高频词主导
            vector[index] = math.copysign(1.0 + math.log1p(abs(raw)), raw)
        return vector

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype="float32")
        return _l2_normalize(np.vstack([self._embed(text) for text in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _l2_normalize(self._embed(text))[0]


def build_embedder(settings: RetrievalSettings) -> EmbeddingProvider:
    """按配置构造嵌入后端。

    fastembed 不可用时自动降级到 hashing 并告警，保证检索链路不整体失败。
    """
    if settings.embedder == "hashing":
        return HashingEmbedder(dimension=settings.embedding_dim)

    provider = FastEmbedProvider(settings)
    try:
        # 显式赋值给 _：触发一次模型加载，尽早暴露缺包/网络问题
        _ = provider.dimension
        return provider
    except Exception as exc:  # noqa: BLE001 - 任何加载失败都降级
        logger.warning(
            "fastembed 不可用（%s: %s），降级为 hashing 嵌入。"
            "如需语义检索请检查 HF_ENDPOINT 镜像与网络。",
            type(exc).__name__,
            exc,
        )
        return HashingEmbedder(dimension=settings.embedding_dim)


# 嵌入模型加载代价高（ONNX 初始化 + 权重读取），必须跨请求复用
_shared: dict[tuple[str, str, int], EmbeddingProvider] = {}
_shared_lock = threading.Lock()


def get_shared_embedder(settings: RetrievalSettings) -> EmbeddingProvider:
    """获取进程内共享的嵌入后端（首次调用完成加载，之后直接复用）。"""
    key = (settings.embedder, settings.embedding_model, settings.embedding_dim)
    provider = _shared.get(key)
    if provider is not None:
        return provider
    with _shared_lock:
        if key not in _shared:
            _shared[key] = build_embedder(settings)
        return _shared[key]


def reset_shared_embedders() -> None:
    """清空共享缓存（测试或配置变更时使用）。"""
    with _shared_lock:
        _shared.clear()


__all__ = [
    "EmbeddingProvider",
    "FastEmbedProvider",
    "HashingEmbedder",
    "build_embedder",
    "get_shared_embedder",
    "reset_shared_embedders",
]
