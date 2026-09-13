"""test_rag_retriever：查询 → Hit 组装（字段/排序/边界）。"""

from __future__ import annotations

import numpy as np
import pytest

from nnnu.services.rag.chunker import Chunk
from nnnu.services.rag.index import RagIndex
from nnnu.services.rag.retriever import Retriever
from tests.unit.rag_fakes import StubEmbedder


def _make_index() -> RagIndex:
    return RagIndex(
        chunks=[
            Chunk(
                text="同向文本" * 30, doc="极限", title="极限", section="夹逼准则", chunk_index=0
            ),
            Chunk(text="正交文本", doc="导数", title="导数", section="定义", chunk_index=1),
            Chunk(text="斜向文本", doc="积分", title="积分", section="换元法", chunk_index=2),
        ],
        vectors=np.array(
            [[1.0, 0.0], [0.0, 1.0], [1 / np.sqrt(2), 1 / np.sqrt(2)]],
            dtype=np.float32,
        ),
    )


def test_query_returns_hits_with_fields() -> None:
    index = _make_index()
    embedder = StubEmbedder(np.array([1.0, 0.0], dtype=np.float32))
    hits = Retriever(index, embedder).query("同向查询", top_k=2)
    assert len(hits) == 2
    first = hits[0]
    assert first.chunk_index == 0
    assert first.doc == "极限"
    assert first.title == "极限"
    assert first.section == "夹逼准则"
    assert first.score == 1.0
    assert first.excerpt == first.text  # 前 120 字符内无省略
    # 降序：同向 > 斜向
    assert hits[1].score == pytest.approx(0.707, abs=0.001)
    # 查询文本原样传给嵌入器
    assert embedder.calls == ["同向查询"]


def test_query_blank_and_empty_index_return_empty() -> None:
    index = _make_index()
    embedder = StubEmbedder()
    retriever = Retriever(index, embedder)
    assert retriever.query("   ") == []
    assert Retriever(None, embedder).query("查询") == []
    empty = RagIndex(chunks=[], vectors=np.zeros((0, 0), dtype=np.float32))
    assert Retriever(empty, embedder).query("查询") == []


def test_query_top_k_exceeds_corpus_returns_all() -> None:
    index = _make_index()
    embedder = StubEmbedder(np.array([1.0, 0.0], dtype=np.float32))
    hits = Retriever(index, embedder).query("查询", top_k=99)
    assert len(hits) == 3
    assert [h.chunk_index for h in hits] == [0, 2, 1]
