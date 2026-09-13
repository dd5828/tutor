"""检索器（Phase 2 W4）：查询 → 嵌入 → 余弦 top-k → Hit 列表。"""

from __future__ import annotations

from dataclasses import dataclass

from nnnu.services.rag.embedder import Embedder
from nnnu.services.rag.index import RagIndex

EXCERPT_LIMIT = 120


@dataclass
class Hit:
    """一条检索命中。excerpt 为块文本前 120 字符（超长加省略号）。"""

    chunk_index: int
    text: str
    doc: str
    title: str
    section: str
    score: float
    excerpt: str


def _excerpt(text: str, limit: int = EXCERPT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


class Retriever:
    """按查询词检索索引。索引为 None（未构建）或空时一律返回空列表。"""

    def __init__(self, index: RagIndex | None, embedder: Embedder) -> None:
        self._index = index
        self._embedder = embedder

    def query(self, text: str, top_k: int = 5) -> list[Hit]:
        query = text.strip()
        if not query or self._index is None or self._index.is_empty():
            return []
        vector = self._embedder.query_embed(query)
        hits: list[Hit] = []
        for idx, score in self._index.search(vector, top_k):
            chunk = self._index.chunks[idx]
            hits.append(
                Hit(
                    chunk_index=idx,
                    text=chunk.text,
                    doc=chunk.doc,
                    title=chunk.title,
                    section=chunk.section,
                    score=round(score, 3),
                    excerpt=_excerpt(chunk.text),
                )
            )
        return hits
