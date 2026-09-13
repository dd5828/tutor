"""test_rag_index：索引构建/持久化/加载/余弦检索。"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from nnnu.core import ChatError
from nnnu.services.rag.chunker import Chunk
from nnnu.services.rag.index import RagIndex, build_index
from tests.unit.rag_fakes import FakeEmbedder


def test_build_index_shape_and_normalized(tmp_path: Any) -> None:
    (tmp_path / "a.txt").write_text("# 极限\n\n" + ("极限的运算法则。\n\n" * 10), encoding="utf-8")
    (tmp_path / "b.txt").write_text("# 导数\n\n" + ("导数的几何意义。\n\n" * 10), encoding="utf-8")
    index = build_index(tmp_path, FakeEmbedder(dim=16))
    assert len(index.chunks) >= 2
    assert index.vectors.shape == (len(index.chunks), 16)
    assert index.vectors.dtype == np.float32
    # 行已 L2 归一化
    np.testing.assert_allclose(np.linalg.norm(index.vectors, axis=1), 1.0)
    assert index.dim == 16


def test_save_load_roundtrip(tmp_path: Any) -> None:
    (tmp_path / "a.txt").write_text("# 极限\n\n" + ("极限内容。\n\n" * 10), encoding="utf-8")
    built = build_index(tmp_path, FakeEmbedder(dim=8))
    index_dir = tmp_path / "index"
    built.save(index_dir)
    loaded = RagIndex.load(index_dir)
    assert loaded is not None
    assert loaded.dim == 8  # dim 从数据读出
    assert [c.text for c in loaded.chunks] == [c.text for c in built.chunks]
    assert [c.doc for c in loaded.chunks] == [c.doc for c in built.chunks]
    np.testing.assert_array_equal(loaded.vectors, built.vectors)


def test_load_missing_returns_none(tmp_path: Any) -> None:
    assert RagIndex.load(tmp_path / "不存在") is None
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert RagIndex.load(empty_dir) is None


def test_load_corrupt_json_raises(tmp_path: Any) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "chunks.json").write_text("{ 不是 JSON", encoding="utf-8")
    (index_dir / "vectors.npy").write_bytes(b"")
    with pytest.raises(ChatError) as exc_info:
        RagIndex.load(index_dir)
    assert exc_info.value.error_code == "RAG_INDEX_ERROR"


def test_load_row_mismatch_raises(tmp_path: Any) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    payload = {"dim": 4, "count": 1, "created_at": "", "chunks": [
        {"text": "t", "doc": "d", "title": "ti", "section": "s", "chunk_index": 0}
    ]}
    (index_dir / "chunks.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    np.save(index_dir / "vectors.npy", np.zeros((2, 4), dtype=np.float32))  # 行数不符
    with pytest.raises(ChatError) as exc_info:
        RagIndex.load(index_dir)
    assert exc_info.value.error_code == "RAG_INDEX_ERROR"


def test_load_dim_mismatch_raises(tmp_path: Any) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    payload = {"dim": 8, "count": 1, "created_at": "", "chunks": [
        {"text": "t", "doc": "d", "title": "ti", "section": "s", "chunk_index": 0}
    ]}
    (index_dir / "chunks.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    np.save(index_dir / "vectors.npy", np.zeros((1, 4), dtype=np.float32))  # dim 不符
    with pytest.raises(ChatError) as exc_info:
        RagIndex.load(index_dir)
    assert exc_info.value.error_code == "RAG_INDEX_ERROR"


def _chunk(text: str, idx: int) -> Chunk:
    return Chunk(text=text, doc="d", title="ti", section="s", chunk_index=idx)


def test_search_top_k_sorted() -> None:
    # 行向量按构建约定已归一化：斜向行用单位向量
    index = RagIndex(
        chunks=[_chunk("同向", 0), _chunk("正交", 1), _chunk("斜向", 2)],
        vectors=np.array(
            [[1.0, 0.0], [0.0, 1.0], [1 / np.sqrt(2), 1 / np.sqrt(2)]],
            dtype=np.float32,
        ),
    )
    query = np.array([1.0, 0.0], dtype=np.float32)
    assert index.search(query, top_k=2) == [(0, 1.0), (2, pytest.approx(0.707, abs=0.001))]
    # top_k 超过语料数返回全部（降序）
    assert [i for i, _ in index.search(query, top_k=9)] == [0, 2, 1]


def test_search_empty_index_and_zero_query() -> None:
    empty = RagIndex(chunks=[], vectors=np.zeros((0, 0), dtype=np.float32))
    assert empty.search(np.array([1.0, 0.0], dtype=np.float32), top_k=5) == []
    index = RagIndex(chunks=[_chunk("t", 0)], vectors=np.array([[1.0, 0.0]], dtype=np.float32))
    # top_k 非正 / 查询零范数 → 空
    assert index.search(np.array([1.0, 0.0], dtype=np.float32), top_k=0) == []
    assert index.search(np.array([0.0, 0.0], dtype=np.float32), top_k=5) == []


def test_save_creates_dir(tmp_path: Any) -> None:
    index = RagIndex(chunks=[_chunk("t", 0)], vectors=np.array([[1.0]], dtype=np.float32))
    index_dir = tmp_path / "深层" / "index"
    index.save(index_dir)
    assert (index_dir / "chunks.json").is_file()
    assert (index_dir / "vectors.npy").is_file()


def test_build_empty_corpus_creates_empty_index(tmp_path: Any) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    (empty_dir / "a.txt").write_text("# 只有标题\n", encoding="utf-8")
    index = build_index(empty_dir, FakeEmbedder())
    assert index.is_empty()
    assert index.dim == 0
