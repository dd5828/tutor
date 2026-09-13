"""RAG 索引：构建 / 持久化 / 余弦检索（Phase 2 W4）。

布局：``data/rag_index/chunks.json``（块元数据 + dim）+ ``vectors.npy``
（(N, dim) float32，行已 L2 归一化）。检索接口与实现解耦（点积 + argsort），
后期数据量大了换 FAISS 只改本文件内部，接口不变。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from nnnu.core import ChatError
from nnnu.services.rag.chunker import Chunk, chunk_documents
from nnnu.services.rag.embedder import Embedder


@dataclass
class RagIndex:
    """已构建的知识库索引（内存态）。"""

    chunks: list[Chunk]
    vectors: np.ndarray  # (N, dim) float32，行已 L2 归一化；N=0 时 shape (0, 0)

    @property
    def dim(self) -> int:
        """向量维度，从数据读出（不硬编码模型维度）。"""
        if self.vectors.size == 0:
            return 0
        return int(self.vectors.shape[1])

    def is_empty(self) -> bool:
        return len(self.chunks) == 0

    @staticmethod
    def normalize(v: np.ndarray) -> np.ndarray:
        """L2 归一化；零向量原样返回。"""
        norm = float(np.linalg.norm(v))
        if norm == 0.0:
            return v
        return v / norm

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        """余弦检索：返回 [(chunk 序号, 相似度)]，降序、至多 top_k 条。

        空索引 / top_k<=0 / 查询向量零范数 → 空列表。
        """
        if self.is_empty() or top_k <= 0:
            return []
        q = self.normalize(query_vector)
        if float(np.linalg.norm(q)) == 0.0:
            return []
        scores = self.vectors @ q
        order = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in order]

    def save(self, index_dir: Path) -> None:
        """写 chunks.json + vectors.npy（目录自动创建）。"""
        index_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "dim": self.dim,
            "count": len(self.chunks),
            "created_at": datetime.now(UTC).isoformat(),
            "chunks": [
                {
                    "text": c.text,
                    "doc": c.doc,
                    "title": c.title,
                    "section": c.section,
                    "chunk_index": c.chunk_index,
                }
                for c in self.chunks
            ],
        }
        (index_dir / "chunks.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        np.save(index_dir / "vectors.npy", self.vectors)

    @classmethod
    def load(cls, index_dir: Path) -> RagIndex | None:
        """从磁盘加载索引；目录或任一文件缺失返回 None（调用方降级）。

        结构损坏（JSON 解析失败 / 块数与向量行数不匹配 / dim 不一致）抛
        ChatError(error_code="RAG_INDEX_ERROR")。
        """
        chunks_path = index_dir / "chunks.json"
        vectors_path = index_dir / "vectors.npy"
        if not chunks_path.is_file() or not vectors_path.is_file():
            return None
        try:
            payload = json.loads(chunks_path.read_text(encoding="utf-8"))
            chunks = [Chunk(**c) for c in payload["chunks"]]
            vectors = np.load(vectors_path)
            dim = payload["dim"]
        except (ValueError, KeyError, TypeError, OSError) as exc:
            raise ChatError(
                f"知识库索引损坏（{index_dir}）: {exc}，请重建索引",
                retryable=False,
                error_code="RAG_INDEX_ERROR",
            ) from exc
        if (
            vectors.ndim != 2
            or vectors.shape[0] != len(chunks)
            or vectors.shape[1] != dim
        ):
            raise ChatError(
                f"知识库索引数据不一致（{index_dir}）："
                f"块数 {len(chunks)}，向量 {vectors.shape}，维度 {dim}。请重建索引",
                retryable=False,
                error_code="RAG_INDEX_ERROR",
            )
        return cls(chunks=chunks, vectors=vectors)


def build_index(corpus_dir: Path, embedder: Embedder) -> RagIndex:
    """解析语料 → 嵌入 → 归一化 → 组装索引。

    语料目录问题（缺失/无 .txt）由 chunk_documents 抛 ChatError。
    """
    chunks = chunk_documents(corpus_dir)
    if not chunks:
        return RagIndex(chunks=[], vectors=np.zeros((0, 0), dtype=np.float32))
    vectors = embedder.embed([c.text for c in chunks])
    with np.errstate(divide="ignore", invalid="ignore"):
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return RagIndex(chunks=chunks, vectors=vectors.astype(np.float32))
