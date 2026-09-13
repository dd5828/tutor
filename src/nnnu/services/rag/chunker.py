"""中文语义分块（Phase 2 W4）。

语料为自写高数知识点 .txt（UTF-8），结构约定：
- 行首 ``# 标题`` 为文档标题（首个生效）；
- 行首 ``## 小节`` 或「第X章/节」为小节标题，附着到其后段落的块元数据；
- 空行分段。

分块规则：
- 段落聚合至目标块长 TARGET_CHUNK（约 400 字符）；
- 单段超过硬上限 HARD_MAX（600）在句边界（。！？；）切分，仍超则强行切；
- 相邻块共享上一块尾部 OVERLAP（50）字符，衔接语义不丢。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from nnnu.core import ChatError

TARGET_CHUNK = 400
HARD_MAX = 600
OVERLAP = 50

_SENTENCE_ENDS = "。！？；"
# 「第X章/节」小节标题（中文数字或阿拉伯数字）
_SECTION_RE = re.compile(r"第[一二三四五六七八九十百\d]+[章节]")


@dataclass
class Chunk:
    """一个可检索的文本块。

    chunk_index 在语料范围内全局唯一（跨文件递增），检索命中后凭它定位。
    """

    text: str
    doc: str
    title: str
    section: str
    chunk_index: int


def _parse_document(text: str) -> tuple[str, list[tuple[str, str]]]:
    """拆出文档标题与（段落文本, 小节标题）列表。"""
    title = ""
    section = ""
    items: list[tuple[str, str]] = []
    para: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if para:
                items.append(("\n".join(para), section))
                para = []
            continue
        if line.startswith("# ") and not title:
            title = line[2:].strip()
            continue
        if line.startswith("## ") or _SECTION_RE.match(line):
            section = line.lstrip("#").strip()
            continue
        para.append(line)
    if para:
        items.append(("\n".join(para), section))
    return title, items


def _split_long(text: str, hard_max: int) -> list[str]:
    """超长段落按句边界切分（无句边界时强行切），每段不超过 hard_max。"""
    if len(text) <= hard_max:
        return [text]
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + hard_max, len(text))
        if end < len(text):
            window = text[start:end]
            cut = -1
            for ch in _SENTENCE_ENDS:
                cut = max(cut, window.rfind(ch))
            if cut >= hard_max // 2:  # 避免句边界太靠前产生碎片
                end = start + cut + 1
        pieces.append(text[start:end])
        start = end
    return pieces


def chunk_text(text: str, *, doc: str, start_index: int = 0) -> list[Chunk]:
    """把一篇文档拆成块。``start_index`` 用于跨文件连续编号。"""
    title, items = _parse_document(text)
    chunks: list[Chunk] = []
    index = start_index
    buf: list[str] = []
    buf_len = 0
    buf_section = ""
    overlap = ""

    def flush() -> None:
        nonlocal buf, buf_len, buf_section, index, overlap
        if not buf:
            return
        body = "\n".join(buf)
        full = f"{overlap}\n{body}" if overlap else body
        chunks.append(
            Chunk(text=full, doc=doc, title=title, section=buf_section, chunk_index=index)
        )
        index += 1
        overlap = full[-OVERLAP:] if len(full) > OVERLAP else full
        buf, buf_len = [], 0

    for para, section in items:
        if len(para) > HARD_MAX:
            flush()
            for piece in _split_long(para, HARD_MAX):
                buf, buf_len, buf_section = [piece], len(piece), section
                flush()
            continue
        if buf and buf_len + 1 + len(para) > TARGET_CHUNK:
            flush()
        buf.append(para)
        buf_len += len(para) + 1
        buf_section = section
    flush()
    return chunks


def chunk_file(path: Path) -> list[Chunk]:
    """按文件拆块；doc 取文件名 stem。"""
    return chunk_text(path.read_text(encoding="utf-8"), doc=path.stem)


def chunk_documents(corpus_dir: Path) -> list[Chunk]:
    """拆语料目录下全部 .txt 文档（按文件名排序，块编号跨文件连续）。"""
    if not corpus_dir.is_dir():
        raise ChatError(
            f"语料目录不存在: {corpus_dir}，请先创建并放入 .txt 知识点文档",
            retryable=False,
            error_code="RAG_CORPUS_MISSING",
        )
    files = sorted(p for p in corpus_dir.iterdir() if p.suffix == ".txt" and p.is_file())
    if not files:
        raise ChatError(
            f"语料目录没有 .txt 文档: {corpus_dir}",
            retryable=False,
            error_code="RAG_CORPUS_EMPTY",
        )
    chunks: list[Chunk] = []
    for path in files:
        chunks.extend(
            chunk_text(path.read_text(encoding="utf-8"), doc=path.stem, start_index=len(chunks))
        )
    return chunks
