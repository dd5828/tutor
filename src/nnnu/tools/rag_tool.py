"""rag 知识库检索工具（Phase 2 W5）。

讲解阶段调用：检索高数知识点库（概念定义、定理条件、公式），把命中的
原文片段与出处回给调用方。所有失败路径都返回 success=False 的 ToolResult
而不是抛异常——教学主流程永不因知识库问题中断。

索引目录默认 data/rag_index（本地构建，见 services/rag/cli.py）；每次执行
都重新加载索引，运行中重建索引无需重启服务。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nnnu.core import BaseTool, ChatError, ToolDefinition, ToolParameter, ToolResult
from nnnu.services.rag import DEFAULT_INDEX_DIR
from nnnu.services.rag.embedder import Embedder, FastEmbedEmbedder
from nnnu.services.rag.index import RagIndex
from nnnu.services.rag.retriever import Hit, Retriever

DEFAULT_TOP_K = 5
MAX_TOP_K = 10

_BUILD_GUIDE = (
    "知识库索引尚未构建，请先运行 "
    "python -m nnnu.services.rag.cli --corpus data/rag_corpus --index data/rag_index"
)


class RagTool(BaseTool):
    """检索知识库工具。

    ``retriever`` 供测试注入；生产按 ``index_dir`` + 默认嵌入器在每次执行时
    现装索引。
    """

    def __init__(
        self,
        retriever: Retriever | None = None,
        *,
        embedder: Embedder | None = None,
        index_dir: Path | None = None,
    ) -> None:
        self._retriever = retriever
        self._embedder = embedder if embedder is not None else FastEmbedEmbedder()
        self._index_dir = index_dir if index_dir is not None else DEFAULT_INDEX_DIR

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="rag",
            description=(
                "检索考研数学（高等数学）知识点库：概念定义、定理条件、公式、考纲。"
                "讲解阶段需要引用教材原文佐证时调用，返回相关知识点原文与出处。"
            ),
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    required=True,
                    description="检索查询：知识点/考点/公式名称，如「泰勒公式展开条件」",
                ),
                ToolParameter(
                    name="top_k",
                    type="integer",
                    required=False,
                    description="返回条数，默认 5，范围 1-10",
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        query = str(kwargs.get("query", "")).strip()
        if not query:
            return ToolResult(content="检索查询不能为空", success=False)
        top_k = self._clamp_top_k(kwargs.get("top_k", DEFAULT_TOP_K))

        retriever = self._retriever
        if retriever is None:
            try:
                index = RagIndex.load(self._index_dir)
            except ChatError as exc:
                return ToolResult(content=str(exc), success=False)
            if index is None:
                return ToolResult(content=_BUILD_GUIDE, success=False)
            if index.is_empty():
                return ToolResult(
                    content="知识库语料为空，请先在 data/rag_corpus/ 放入知识点文档后重建索引",
                    success=False,
                )
            retriever = Retriever(index, self._embedder)

        try:
            hits = retriever.query(query, top_k=top_k)
        except ChatError as exc:
            return ToolResult(content=f"检索失败: {exc}", success=False)
        if not hits:
            return ToolResult(
                content=f"未找到与「{query}」直接相关的知识点条目", success=True
            )
        return ToolResult(
            content=self._format_content(query, hits),
            sources=[self._format_source(hit) for hit in hits],
            metadata={
                "query": query,
                "top_k": top_k,
                "scores": [hit.score for hit in hits],
            },
            success=True,
        )

    @staticmethod
    def _clamp_top_k(value: Any) -> int:
        try:
            k = int(value)
        except (TypeError, ValueError):
            return DEFAULT_TOP_K
        return max(1, min(MAX_TOP_K, k))

    @staticmethod
    def _format_content(query: str, hits: list[Hit]) -> str:
        lines = [f"找到 {len(hits)} 条与「{query}」相关的知识点："]
        for i, hit in enumerate(hits, 1):
            lines.append(
                f"[{i}] {hit.title or hit.doc}·{hit.section}（相关度 {hit.score:.2f}）"
            )
            lines.append(hit.text[:200])
        return "\n".join(lines)

    @staticmethod
    def _format_source(hit: Hit) -> dict[str, Any]:
        return {
            "title": f"{hit.doc}·{hit.section}",
            "excerpt": hit.excerpt,
            "score": hit.score,
            "type": "rag_knowledge",
        }
