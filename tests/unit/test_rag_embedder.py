"""test_rag_embedder：嵌入器协议与 FastEmbedEmbedder 延迟构造/降级。"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any, ClassVar

import numpy as np
import pytest

from nnnu.core import ChatError
from nnnu.services.rag import embedder as embedder_mod
from nnnu.services.rag.embedder import FastEmbedEmbedder
from tests.unit.rag_fakes import FakeEmbedder


def test_fake_embedder_deterministic_and_dim() -> None:
    embedder = FakeEmbedder(dim=8)
    first = embedder.embed(["极限定义", "导数"])
    second = embedder.embed(["极限定义", "导数"])
    assert first.shape == (2, 8)
    assert first.dtype == np.float32
    np.testing.assert_array_equal(first, second)


def test_fake_embedder_differs_by_text() -> None:
    embedder = FakeEmbedder(dim=8)
    vectors = embedder.embed(["极限定义", "导数定义"])
    assert not np.array_equal(vectors[0], vectors[1])


def test_fake_embedder_empty_raises() -> None:
    with pytest.raises(ValueError, match="不能为空"):
        FakeEmbedder().embed([])


def test_fastembed_import_missing_degrades(monkeypatch: Any) -> None:
    def fake_import(name: str) -> ModuleType:
        raise ImportError(f"No module named {name!r}")

    monkeypatch.setattr(importlib, "import_module", fake_import)
    with pytest.raises(ChatError) as exc_info:
        FastEmbedEmbedder().embed(["文本"])
    assert exc_info.value.error_code == "RAG_EMBEDDER_MISSING"


class _FakeFastembed:
    """记录构造调用的假 fastembed 模块。"""

    constructions: ClassVar[list[tuple[str, str]]] = []

    class TextEmbedding:
        def __init__(self, *, model_name: str, cache_dir: str) -> None:
            _FakeFastembed.constructions.append((model_name, cache_dir))

        def embed(self, texts: list[str]) -> list[np.ndarray]:
            return [np.zeros(4, dtype=np.float32) for _ in texts]


def test_fastembed_lazy_constructs_once(monkeypatch: Any) -> None:
    _FakeFastembed.constructions.clear()
    monkeypatch.setattr(importlib, "import_module", lambda name: _FakeFastembed)
    embedder = FastEmbedEmbedder(cache_dir="C:/tmp/cache")
    embedder.embed(["a", "b"])
    embedder.embed(["c"])
    # 延迟构造：两次 embed 只构造一次模型，cache_dir 透传
    assert _FakeFastembed.constructions == [
        ("BAAI/bge-small-zh-v1.5", "C:/tmp/cache")
    ]


def test_fastembed_model_error_wrapped(monkeypatch: Any) -> None:
    class _Broken:
        class TextEmbedding:
            def __init__(self, **kwargs: object) -> None:
                raise RuntimeError("下载失败")

    monkeypatch.setattr(importlib, "import_module", lambda name: _Broken)
    with pytest.raises(ChatError) as exc_info:
        FastEmbedEmbedder().embed(["文本"])
    assert exc_info.value.error_code == "RAG_MODEL_ERROR"
    assert "下载失败" in str(exc_info.value)


def test_embedder_protocol_query_embed_falls_back_to_embed() -> None:
    # 协议默认实现：query_embed 退化为 embed([text])[0]
    embedder = FakeEmbedder(dim=8)
    np.testing.assert_array_equal(embedder.query_embed("夹逼准则"), embedder.embed(["夹逼准则"])[0])


def test_default_model_name_constant() -> None:
    assert embedder_mod.DEFAULT_MODEL_NAME == "BAAI/bge-small-zh-v1.5"
