"""检索质量评测运行器（Phase 2 W6）：真实索引 + 真实嵌入打分。

无 LLM 调用、无需 API key（纯本地模型与索引）。
用法：
    python -m tests.evals.run_rag_evals
    python -m tests.evals.run_rag_evals --limit 10 --out tests/evals/rag_results.jsonl
索引目录默认 data/rag_index，环境变量 NNNU_RAG_INDEX_DIR 可覆盖（测临时索引）。
指标：doc-hit@5（top-5 命中 expected_doc）、section-hit@5（仅统计带
expected_section 的条目）。基线固化 tests/evals/rag_results.jsonl，
后续检索改动必重跑不许跌破。

CI 不跑本脚本（需 [rag] 依赖与本地模型缓存）；打分逻辑由
tests/unit/test_rag_evals.py 用假件覆盖。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from nnnu.core import ChatError
from nnnu.services.rag import DEFAULT_INDEX_DIR
from nnnu.services.rag.embedder import Embedder, FastEmbedEmbedder
from nnnu.services.rag.index import RagIndex
from nnnu.services.rag.retriever import Retriever

DEFAULT_CASES_PATH = Path(__file__).with_name("rag.jsonl")
DEFAULT_OUT_PATH = Path(__file__).with_name("rag_results.jsonl")
TOP_K = 5


@dataclass
class RagCase:
    """一条检索评测用例。"""

    id: str
    query: str
    expected_doc: str
    expected_section: str | None


@dataclass
class RagCaseResult:
    """一条用例的打分结果。"""

    case_id: str
    doc_hit: bool
    section_hit: bool | None  # 未设 expected_section 时为 None（不计入分母）
    top_docs: list[str]
    top_scores: list[float]


def resolve_index_dir() -> Path:
    env = os.environ.get("NNNU_RAG_INDEX_DIR")
    return Path(env) if env else DEFAULT_INDEX_DIR


def load_rag_cases(path: Path, *, doc_vocab: set[str]) -> list[RagCase]:
    """加载并校验用例：缺字段/空值/expected_doc 不在词表 → ValueError 带行号。"""
    if not path.is_file():
        raise ValueError(f"评测文件不存在: {path}")
    cases: list[RagCase] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_no} 行 JSON 解析失败: {exc}") from exc
            for field in ("id", "query", "expected_doc"):
                value = raw.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{path} 第 {line_no} 行缺少非空字段 {field}")
            doc = str(raw["expected_doc"]).strip()
            if doc not in doc_vocab:
                raise ValueError(
                    f"{path} 第 {line_no} 行 expected_doc「{doc}」"
                    f"不在语料词表（{sorted(doc_vocab)}）"
                )
            section = raw.get("expected_section")
            cases.append(
                RagCase(
                    id=str(raw["id"]).strip(),
                    query=str(raw["query"]).strip(),
                    expected_doc=doc,
                    expected_section=str(section).strip() if section else None,
                )
            )
    if not cases:
        raise ValueError(f"{path} 没有任何用例")
    return cases


def evaluate_case(retriever: Retriever, case: RagCase, *, top_k: int = TOP_K) -> RagCaseResult:
    hits = retriever.query(case.query, top_k=top_k)
    doc_hit = any(h.doc == case.expected_doc for h in hits)
    section_hit: bool | None = None
    if case.expected_section is not None:
        section_hit = any(
            h.doc == case.expected_doc and h.section == case.expected_section for h in hits
        )
    return RagCaseResult(
        case_id=case.id,
        doc_hit=doc_hit,
        section_hit=section_hit,
        top_docs=[h.doc for h in hits],
        top_scores=[h.score for h in hits],
    )


def summarize(results: list[RagCaseResult]) -> tuple[float, float, int]:
    """返回 (doc 命中率, section 命中率, section 条目数)；空列表防除零。"""
    if not results:
        return 0.0, 0.0, 0
    doc_hits = sum(1 for r in results if r.doc_hit)
    section_results = [r for r in results if r.section_hit is not None]
    section_hits = sum(1 for r in section_results if r.section_hit)
    return doc_hits / len(results), (
        section_hits / len(section_results) if section_results else 0.0
    ), len(section_results)


def _load_index(index_dir: Path) -> RagIndex:
    """加载索引；缺失/为空给友好 ChatError（含构建指引）。"""
    index = RagIndex.load(index_dir)
    if index is None:
        raise ChatError(
            f"索引尚未构建：{index_dir}\n"
            "请先运行: python -m nnnu.services.rag.cli --corpus data/rag_corpus "
            f"--index {index_dir}",
            retryable=False,
            error_code="RAG_INDEX_MISSING",
        )
    if index.is_empty():
        raise ChatError(
            f"索引为空（{index_dir}）：语料目录可能没有文档，请重建索引",
            retryable=False,
            error_code="RAG_INDEX_EMPTY",
        )
    return index


def run(
    cases_path: Path,
    *,
    index_dir: Path,
    out_path: Path | None = None,
    limit: int | None = None,
    embedder: Embedder | None = None,
) -> list[RagCaseResult]:
    """逐条打分；``embedder`` 供测试注入假件。"""
    if embedder is None:
        embedder = FastEmbedEmbedder()
    # 语料词表取索引里的 chunk.doc 集合（比硬编码词表更抗语料改名）
    index = _load_index(index_dir)
    doc_vocab = {chunk.doc for chunk in index.chunks}
    cases = load_rag_cases(cases_path, doc_vocab=doc_vocab)
    if limit is not None:
        cases = cases[:limit]

    retriever = Retriever(index, embedder)
    results: list[RagCaseResult] = []
    for i, case in enumerate(cases, 1):
        result = evaluate_case(retriever, case)
        results.append(result)
        top = result.top_docs[0] if result.top_docs else "（无命中）"
        score = f"score={result.top_scores[0]:.2f}" if result.top_scores else "score=-"
        print(
            f"[{i}/{len(cases)}] {case.id} "
            f"{'命中' if result.doc_hit else '未命中'}({top}) {score}"
        )
    doc_rate, section_rate, section_total = summarize(results)
    print(
        f"doc-hit@{TOP_K}: {doc_rate:.1%}；"
        f"section-hit@{TOP_K}: {section_rate:.1%}（{section_total} 条带小节标注）"
    )
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
        print(f"结果已写入 {out_path}")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tests.evals.run_rag_evals",
        description="检索质量评测：50 条 query 打 doc/section 命中率（本地模型，无需 key）",
    )
    parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH), help="评测数据 jsonl")
    parser.add_argument(
        "--index",
        default=None,
        help=f"索引目录（默认 {DEFAULT_INDEX_DIR}，NNNU_RAG_INDEX_DIR 可覆盖）",
    )
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条")
    parser.add_argument(
        "--out", default=None, help=f"结果落盘路径（默认 {DEFAULT_OUT_PATH}）"
    )
    args = parser.parse_args(argv)

    index_dir = Path(args.index) if args.index else resolve_index_dir()
    out_path = Path(args.out) if args.out else DEFAULT_OUT_PATH
    try:
        run(Path(args.cases), index_dir=index_dir, out_path=out_path, limit=args.limit)
    except ChatError as exc:
        print(f"错误：{exc}")
        return 1
    except ValueError as exc:
        print(f"错误：{exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
