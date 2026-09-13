"""检索评测运行器单测（假索引/假嵌入器，不触网不下载模型）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from nnnu.core import ChatError
from nnnu.services.rag import DEFAULT_INDEX_DIR
from nnnu.services.rag.chunker import Chunk
from nnnu.services.rag.index import build_index
from nnnu.services.rag.retriever import Retriever
from tests.evals.run_rag_evals import (
    RagCase,
    RagCaseResult,
    evaluate_case,
    load_rag_cases,
    resolve_index_dir,
    run,
    summarize,
)
from tests.unit.rag_fakes import FakeEmbedder, StubEmbedder


def _write_cases(tmp_path: Any, lines: list[dict[str, Any]]) -> Any:
    path = tmp_path / "rag_cases.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return path


def _min_case(**overrides: Any) -> dict[str, Any]:
    case = {"id": "rag-001", "query": "什么是极限？", "expected_doc": "01-函数极限连续"}
    case.update(overrides)
    return case


def test_load_rag_cases_parses_minimal(tmp_path: Any) -> None:
    path = _write_cases(tmp_path, [_min_case()])
    cases = load_rag_cases(path, doc_vocab={"01-函数极限连续"})
    assert cases == [RagCase("rag-001", "什么是极限？", "01-函数极限连续", None)]


def test_load_rag_cases_parses_with_section(tmp_path: Any) -> None:
    path = _write_cases(tmp_path, [_min_case(expected_section="极限的概念")])
    cases = load_rag_cases(path, doc_vocab={"01-函数极限连续"})
    assert cases[0].expected_section == "极限的概念"


def test_load_rag_cases_rejects_missing_field(tmp_path: Any) -> None:
    path = _write_cases(tmp_path, [{"id": "rag-001", "query": "x"}])  # 缺 expected_doc
    with pytest.raises(ValueError) as exc_info:
        load_rag_cases(path, doc_vocab={"01-函数极限连续"})
    assert "expected_doc" in str(exc_info.value)
    assert "第 1 行" in str(exc_info.value)


def test_load_rag_cases_rejects_unknown_doc(tmp_path: Any) -> None:
    path = _write_cases(tmp_path, [_min_case(expected_doc="99-不存在")])
    with pytest.raises(ValueError) as exc_info:
        load_rag_cases(path, doc_vocab={"01-函数极限连续"})
    assert "不在语料词表" in str(exc_info.value)


def test_load_rag_cases_empty_file_raises(tmp_path: Any) -> None:
    path = _write_cases(tmp_path, [])
    with pytest.raises(ValueError) as exc_info:
        load_rag_cases(path, doc_vocab={"01-函数极限连续"})
    assert "没有任何用例" in str(exc_info.value)


def _chunk(text: str, doc: str, section: str, idx: int) -> Chunk:
    return Chunk(text=text, doc=doc, title="t", section=section, chunk_index=idx)


def _retriever_with_hits(hits_docs: list[str], hit_sections: list[str]) -> Retriever:
    """构造返回指定命中序列的假检索器（StubEmbedder 向量无关紧要）。"""
    class _ScriptedRetriever(Retriever):
        def query(self, text: str, top_k: int = 5) -> list[Any]:
            from nnnu.services.rag.retriever import Hit

            return [
                Hit(
                    chunk_index=i,
                    text="x",
                    doc=doc,
                    title="t",
                    section=section,
                    score=0.9 - i * 0.1,
                    excerpt="x",
                )
                for i, (doc, section) in enumerate(zip(hits_docs, hit_sections, strict=True))
            ]

    return _ScriptedRetriever(None, StubEmbedder())


def test_evaluate_case_doc_hit_at_rank_5() -> None:
    retriever = _retriever_with_hits(
        ["d1", "d2", "d3", "d4", "01-函数极限连续"], ["s1", "s2", "s3", "s4", "s5"]
    )
    case = RagCase("rag-001", "q", "01-函数极限连续", None)
    result = evaluate_case(retriever, case)
    assert result.doc_hit  # 第 5 名命中也算
    assert result.section_hit is None


def test_evaluate_case_doc_miss() -> None:
    retriever = _retriever_with_hits(["d1", "d2", "d3", "d4", "d5"], ["s"] * 5)
    case = RagCase("rag-001", "q", "01-函数极限连续", None)
    result = evaluate_case(retriever, case)
    assert not result.doc_hit
    assert result.top_docs == ["d1", "d2", "d3", "d4", "d5"]
    assert result.top_scores[0] == 0.9


def test_evaluate_case_section_hit_and_miss() -> None:
    case = RagCase("rag-001", "q", "01-函数极限连续", "极限的概念")
    hit = evaluate_case(
        _retriever_with_hits(["01-函数极限连续"], ["极限的概念"]), case
    )
    assert hit.section_hit
    miss = evaluate_case(
        _retriever_with_hits(["01-函数极限连续"], ["连续与间断"]), case
    )
    assert not miss.section_hit
    # doc 不对但 section 同名也不算
    wrong_doc = evaluate_case(
        _retriever_with_hits(["02-无穷小与夹逼准则"], ["极限的概念"]), case
    )
    assert not wrong_doc.section_hit


def test_summarize_rates_and_zero_division() -> None:
    results = [
        RagCaseResult("a", True, True, [], []),
        RagCaseResult("b", False, False, [], []),
        RagCaseResult("c", True, None, [], []),  # 无小节标注：不计入 section 分母
    ]
    doc_rate, section_rate, section_total = summarize(results)
    assert doc_rate == 2 / 3
    assert section_rate == 0.5
    assert section_total == 2
    assert summarize([]) == (0.0, 0.0, 0)


def test_resolve_index_dir_env_override(monkeypatch: Any) -> None:
    monkeypatch.setenv("NNNU_RAG_INDEX_DIR", "D:/tmp/rag")
    assert resolve_index_dir() == Path("D:/tmp/rag")
    monkeypatch.delenv("NNNU_RAG_INDEX_DIR")
    assert resolve_index_dir() == DEFAULT_INDEX_DIR


def test_run_missing_index_friendly_error(tmp_path: Any) -> None:
    with pytest.raises(ChatError) as exc_info:
        run(tmp_path / "cases.jsonl", index_dir=tmp_path / "不存在")
    assert exc_info.value.error_code == "RAG_INDEX_MISSING"
    assert "nnnu.services.rag.cli" in str(exc_info.value)


def test_run_end_to_end_with_tmp_index(tmp_path: Any) -> None:
    # 两篇语料 → FakeEmbedder 建真索引 → query 与 chunk 全文一致精确命中。
    # 注意：哈希向量无语义，且小块索引 top-5 必含全部 doc，因此 e2e 只放
    # 命中用例（未命中逻辑由 test_evaluate_case_doc_miss 覆盖）。
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "01-函数极限连续.txt").write_text(
        "# 函数极限\n\n## 极限的概念\n\n数列极限的 ε-N 定义刻画了当 n 充分大时差距可以任意小。",
        encoding="utf-8",
    )
    (corpus / "05-微分中值定理.txt").write_text(
        "# 中值定理\n\n## 罗尔定理\n\n罗尔定理要求闭区间连续、开区间可导、端点值相等。",
        encoding="utf-8",
    )
    index = build_index(corpus, FakeEmbedder(dim=16))
    index_dir = tmp_path / "index"
    index.save(index_dir)

    cases = _write_cases(
        tmp_path,
        [
            _min_case(
                query="数列极限的 ε-N 定义刻画了当 n 充分大时差距可以任意小。",
                expected_doc="01-函数极限连续",
                expected_section="极限的概念",
            ),
            _min_case(
                id="rag-002",
                query="罗尔定理要求闭区间连续、开区间可导、端点值相等。",
                expected_doc="05-微分中值定理",
                expected_section="罗尔定理",
            ),
        ],
    )
    out_path = tmp_path / "out.jsonl"
    results = run(
        cases,
        index_dir=index_dir,
        out_path=out_path,
        embedder=FakeEmbedder(dim=16),
    )
    assert len(results) == 2
    assert results[0].doc_hit and results[0].section_hit
    assert results[1].doc_hit and results[1].section_hit
    # 落盘可解析
    written = [
        json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(written) == 2
    assert written[0]["case_id"] == "rag-001"
