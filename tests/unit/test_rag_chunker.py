"""test_rag_chunker：中文语义分块（标题解析/聚合/硬上限/重叠/元数据）。"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import pytest

from nnnu.core import ChatError
from nnnu.services.rag.chunker import (
    HARD_MAX,
    OVERLAP,
    TARGET_CHUNK,
    chunk_documents,
    chunk_text,
)


def test_chunk_title_and_section_parsed() -> None:
    text = (
        "# 微分中值定理\n\n"
        "## 罗尔定理\n\n"
        "第一段内容。\n\n"
        "第二段内容。\n\n"
        "## 拉格朗日中值定理\n\n"
        "第三段内容。\n\n"
        "第四段内容。\n"
    )
    chunks = chunk_text(text, doc="中值定理")
    assert all(c.title == "微分中值定理" for c in chunks)
    # section 取块内最后一段所属小节：块含第一~四段 → 拉格朗日中值定理
    assert chunks[-1].section == "拉格朗日中值定理"


def test_chunk_chinese_numeral_section_parsed() -> None:
    text = "# 极限\n\n第1章 数列极限\n\n一些内容。\n"
    chunks = chunk_text(text, doc="极限")
    assert chunks[0].section == "第1章 数列极限"


def test_chunk_aggregates_paragraphs_to_target() -> None:
    para = "极限存在的一个必要条件，用于判断数列或函数的敛散性。" * 3  # 约 90 字符
    text = "\n\n".join([para] * 10)
    chunks = chunk_text(text, doc="极限")
    assert len(chunks) == 2
    # 聚合到接近 TARGET_CHUNK 才 flush（超限前触发：块长 > TARGET - 段长）
    assert len(chunks[0].text) > TARGET_CHUNK - len(para)
    assert len(chunks[0].text) <= TARGET_CHUNK


def test_chunk_hard_cap_splits_long_paragraph() -> None:
    sentence = "这是极限定义中需要逐步理解的一个完整句子，包含完整的语义内容。"  # 31 字符
    para = sentence * 30  # 930 字符 > HARD_MAX
    chunks = chunk_text(para, doc="极限定义")
    assert len(chunks) >= 2
    for i, chunk in enumerate(chunks):
        # 块 = 重叠前缀（≤OVERLAP + 换行）+ 切分片段（≤HARD_MAX）
        prefix = (OVERLAP + 1) if i > 0 else 0
        assert len(chunk.text) - prefix <= HARD_MAX


def test_chunk_overlap_between_adjacent() -> None:
    para = "数列极限与函数极限的统一定义，用 epsilon 与 delta 语言刻画。" * 3
    text = "\n\n".join([para] * 10)
    chunks = chunk_text(text, doc="极限")
    assert len(chunks) >= 2
    for prev, nxt in pairwise(chunks):
        if len(prev.text) > OVERLAP:
            assert nxt.text.startswith(prev.text[-OVERLAP:])


def test_chunk_metadata_fields_and_cross_file_index() -> None:
    text1 = "# 极限\n\n" + ("极限的运算法则内容。\n\n" * 30)
    text2 = "# 导数\n\n" + ("导数的几何意义内容。\n\n" * 30)
    chunks1 = chunk_text(text1, doc="极限", start_index=0)
    chunks2 = chunk_text(text2, doc="导数", start_index=len(chunks1))
    assert [c.chunk_index for c in chunks1] == list(range(len(chunks1)))
    assert chunks2[0].chunk_index == len(chunks1)
    assert chunks1[0].doc == "极限" and chunks2[0].doc == "导数"
    assert chunks1[0].title == "极限" and chunks2[0].title == "导数"


def test_chunk_documents_only_txt_and_sorted(tmp_path: Any) -> None:
    (tmp_path / "b.md").write_text("# 不是语料\n", encoding="utf-8")
    (tmp_path / "a.txt").write_text("# A\n\n内容甲。\n", encoding="utf-8")
    (tmp_path / "c.txt").write_text("# C\n\n内容乙。\n", encoding="utf-8")
    chunks = chunk_documents(tmp_path)
    assert [c.doc for c in chunks] == ["a", "c"]
    # 跨文件 chunk_index 连续唯一
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_empty_corpus_raises(tmp_path: Any) -> None:
    with pytest.raises(ChatError) as exc_info:
        chunk_documents(tmp_path / "不存在")
    assert exc_info.value.error_code == "RAG_CORPUS_MISSING"

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(ChatError) as exc_info:
        chunk_documents(empty_dir)
    assert exc_info.value.error_code == "RAG_CORPUS_EMPTY"


def test_chunk_text_empty_returns_no_chunks() -> None:
    assert chunk_text("", doc="空文档") == []
