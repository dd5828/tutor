"""RAG 命令行：索引构建与检索冒烟。

用法：
- 构建索引（首次运行会联网下载嵌入模型）：
  ``python -m nnnu.services.rag.cli [--corpus data/rag_corpus] [--index data/rag_index]``
- 检索冒烟：
  ``python -m nnnu.services.rag.cli --query "泰勒公式展开条件" [--top-k 3]``
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nnnu.core import ChatError
from nnnu.services.rag import DEFAULT_CORPUS_DIR, DEFAULT_INDEX_DIR
from nnnu.services.rag.embedder import FastEmbedEmbedder
from nnnu.services.rag.index import RagIndex, build_index
from nnnu.services.rag.retriever import Retriever


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m nnnu.services.rag.cli",
        description="RAG 知识库：构建索引 / 检索冒烟",
    )
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS_DIR), help="语料目录")
    parser.add_argument("--index", default=str(DEFAULT_INDEX_DIR), help="索引目录")
    parser.add_argument("--query", help="给定查询词时执行检索冒烟（否则构建索引）")
    parser.add_argument("--top-k", type=int, default=3, help="检索冒烟返回条数")
    args = parser.parse_args(argv)

    try:
        if args.query:
            index = RagIndex.load(Path(args.index))
            if index is None:
                print(
                    f"索引未构建：{args.index}\n"
                    "请先运行: python -m nnnu.services.rag.cli"
                )
                return 1
            hits = Retriever(index, FastEmbedEmbedder()).query(args.query, top_k=args.top_k)
            if not hits:
                print("未命中相关条目")
                return 0
            for i, hit in enumerate(hits, 1):
                print(f"[{i}] {hit.doc}·{hit.section}（相关度 {hit.score:.3f}）")
                print(f"    {hit.excerpt}")
                print()
            return 0
        index = build_index(Path(args.corpus), FastEmbedEmbedder())
        index.save(Path(args.index))
        print(f"索引构建完成：{index.dim} 维 × {len(index.chunks)} 块 → {args.index}")
        return 0
    except ChatError as exc:
        print(f"错误：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
