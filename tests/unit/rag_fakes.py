"""RAG 测试共享替身（多测试文件复用，故独立成文件）。"""

from __future__ import annotations

import hashlib

import numpy as np

from nnnu.services.rag.embedder import Embedder


class FakeEmbedder(Embedder):
    """确定性哈希嵌入：同一文本恒得同一向量（sha256 字节级，跨平台稳定）。

    向量无语义相似度，只用于验证构建→保存→加载→检索链路的形状与确定性；
    检索排序正确性由手工构造向量直接注入 RagIndex 验证（见 test_rag_index）。
    """

    def __init__(self, dim: int = 16) -> None:
        self._dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise ValueError("texts 不能为空")
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()  # 32 字节
            vec = np.frombuffer(digest, dtype=np.uint8).astype(np.float32) / 255.0
            rows.append(np.resize(vec, self._dim))
        return np.stack(rows)


class StubEmbedder(Embedder):
    """查询恒返回指定向量的替身（检索排序测试用），并记录调用文本。"""

    def __init__(self, vector: np.ndarray | None = None) -> None:
        self._vector = vector if vector is not None else np.zeros(2, dtype=np.float32)
        self.calls: list[str] = []

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls.extend(texts)
        return np.stack([self._vector for _ in texts])

    def query_embed(self, text: str) -> np.ndarray:
        self.calls.append(text)
        return self._vector
