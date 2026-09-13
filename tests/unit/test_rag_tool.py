"""RagTool 单测（注入假 Retriever / 临时索引目录，不触网不下载模型）。"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from nnnu.core import ChatError
from nnnu.services.rag.index import RagIndex
from nnnu.services.rag.retriever import Hit, Retriever
from nnnu.tools.rag_tool import RagTool


def _hit(text: str, doc: str, section: str, score: float, idx: int) -> Hit:
    return Hit(
        chunk_index=idx,
        text=text,
        doc=doc,
        title=text[:4],
        section=section,
        score=score,
        excerpt=text[:20],
    )


class _FakeRetriever:
    """记录调用并返回预置命中的假检索器。"""

    def __init__(self, hits: list[Hit] | None = None, error: Exception | None = None) -> None:
        self._hits = hits if hits is not None else []
        self._error = error
        self.calls: list[tuple[str, int]] = []

    def query(self, text: str, top_k: int = 5) -> list[Hit]:
        self.calls.append((text, top_k))
        if self._error is not None:
            raise self._error
        return self._hits


async def test_get_definition_schema() -> None:
    definition = RagTool().get_definition()
    assert definition.name == "rag"
    params = {p.name: p for p in definition.parameters}
    assert params["query"].required and params["query"].type == "string"
    assert not params["top_k"].required and params["top_k"].type == "integer"
    schema = definition.to_openai_schema()
    assert schema["function"]["parameters"]["required"] == ["query"]
    assert set(schema["function"]["parameters"]["properties"]) == {"query", "top_k"}


async def test_execute_returns_numbered_content_and_sources() -> None:
    retriever = _FakeRetriever(
        [
            _hit("罗尔定理内容", "中值定理", "罗尔定理", 0.92, 0),
            _hit("辅助函数构造", "中值定理", "构造", 0.87, 1),
        ]
    )
    result = await RagTool(retriever=cast(Retriever, retriever)).execute(query="罗尔定理")
    assert result.success
    assert "找到 2 条" in result.content
    assert "[1]" in result.content and "[2]" in result.content
    assert "相关度 0.92" in result.content
    assert result.sources == [
        {
            "title": "中值定理·罗尔定理",
            "excerpt": "罗尔定理内容",
            "score": 0.92,
            "type": "rag_knowledge",
        },
        {
            "title": "中值定理·构造",
            "excerpt": "辅助函数构造",
            "score": 0.87,
            "type": "rag_knowledge",
        },
    ]
    assert result.metadata["query"] == "罗尔定理"
    assert result.metadata["top_k"] == 5
    assert result.metadata["scores"] == [0.92, 0.87]


async def test_execute_empty_query_fails() -> None:
    result = await RagTool().execute(query="   ")
    assert not result.success
    assert "不能为空" in result.content


async def test_execute_no_index_degrades_friendly(tmp_path: Any) -> None:
    tool = RagTool(index_dir=tmp_path / "不存在")
    result = await tool.execute(query="极限")
    assert not result.success
    assert "尚未构建" in result.content
    assert "nnnu.services.rag.cli" in result.content


async def test_execute_empty_index_degrades(tmp_path: Any) -> None:
    index_dir = tmp_path / "index"
    RagIndex(chunks=[], vectors=np.zeros((0, 0), dtype=np.float32)).save(index_dir)
    result = await RagTool(index_dir=index_dir).execute(query="极限")
    assert not result.success
    assert "语料为空" in result.content


async def test_execute_no_hits_success_empty_sources() -> None:
    result = await RagTool(retriever=cast(Retriever, _FakeRetriever())).execute(query="奥数题")
    assert result.success
    assert result.sources == []
    assert "未找到" in result.content


async def test_execute_embed_error_degrades() -> None:
    retriever = _FakeRetriever(error=ChatError("嵌入失败", error_code="RAG_EMBED_ERROR"))
    result = await RagTool(retriever=cast(Retriever, retriever)).execute(query="极限")
    assert not result.success
    assert "检索失败" in result.content


async def test_execute_top_k_clamped() -> None:
    retriever = _FakeRetriever()
    tool = RagTool(retriever=cast(Retriever, retriever))
    await tool.execute(query="极限", top_k=999)
    await tool.execute(query="极限", top_k=0)
    await tool.execute(query="极限", top_k="abc")
    assert retriever.calls == [("极限", 10), ("极限", 1), ("极限", 5)]


def test_format_source_shape() -> None:
    source = RagTool._format_source(_hit("内容", "文档", "小节", 0.5, 3))
    assert source == {
        "title": "文档·小节",
        "excerpt": "内容",
        "score": 0.5,
        "type": "rag_knowledge",
    }
