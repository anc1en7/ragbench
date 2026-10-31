"""Recursive markdown chunker: heading/paragraph units within a token budget (spec 5.2, ruling 6).

Split units: the parsed heading sections from Phase 1 (``doc.sections``, each carrying its
``heading_path`` and char offsets) — reusing the corpus parser rather than re-scanning, so
code-fence-aware heading detection stays in one place — and, within a section, the
blank-line-separated paragraphs. Chunks never split mid-paragraph and never break at a
non-boundary. Every chunk respects ``size_tokens`` measured in cl100k tokens, with one
documented tolerance: a single paragraph that alone exceeds the budget is emitted as one
oversize chunk (paragraphs are indivisible). ``overlap_tokens`` is applied at paragraph
boundaries between consecutive chunks of the same section only: the next chunk is prefixed
with whole tail paragraphs of the previous chunk while their combined token count stays
within the overlap budget, and its packing budget is reduced accordingly so the size
tolerance is not violated. A section's heading line is included only in that section's
first chunk. Only ``recursive_md`` populates ``Chunk.metadata``: ``heading_path`` from the
section (empty string for documents without parsed sections, which are treated as a single
unnamed section).
"""

from __future__ import annotations

import re
from typing import Any

from ragbench import (
    chunking as _chunking,  # make_chunk_id resolved at call time (import order safe)
)
from ragbench.chunking.tokenizer import count_tokens
from ragbench.config import ChunkerCfg
from ragbench.types import Chunk, Document, Section

_HEADING_LINE = re.compile(r"#{1,6}\s+\S")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")

Span = tuple[int, int]


class RecursiveMarkdownChunker:
    """Heading-then-paragraph greedy packing under a cl100k token budget."""

    def __init__(self, cfg: ChunkerCfg) -> None:
        if cfg.type != "recursive_md":
            raise ValueError(
                f"RecursiveMarkdownChunker requires chunker type 'recursive_md', got '{cfg.type}'"
            )
        if cfg.size_tokens is None or cfg.overlap_tokens is None:
            raise ValueError("RecursiveMarkdownChunker requires size_tokens and overlap_tokens")
        if cfg.overlap_tokens >= cfg.size_tokens:
            raise ValueError(
                f"overlap_tokens ({cfg.overlap_tokens}) must be smaller than "
                f"size_tokens ({cfg.size_tokens})"
            )
        self._size = cfg.size_tokens
        self._overlap = cfg.overlap_tokens
        self._config_dump: dict[str, Any] = cfg.model_dump(mode="json")

    def chunk(self, doc: Document) -> list[Chunk]:
        if not doc.text.strip():
            return []
        chunks: list[Chunk] = []
        sections = list(doc.sections) or [_whole_document(doc)]
        if doc.sections and doc.sections[0].start > 0:
            chunks.extend(self._chunk_section(doc, 0, doc.sections[0].start, ""))
        for section in sections:
            chunks.extend(
                self._chunk_section(doc, section.start, section.end, section.heading_path)
            )
        return chunks

    def _chunk_section(
        self, doc: Document, sec_start: int, sec_end: int, heading_path: str
    ) -> list[Chunk]:
        text = doc.text
        heading, body_start = _section_heading(text, sec_start, sec_end)
        paragraphs = _paragraphs(text, body_start, sec_end)
        chunks: list[Chunk] = []
        if not paragraphs and heading is None:
            return chunks
        if not paragraphs:
            assert heading is not None
            chunks.append(self._emit(doc, heading[0], heading[1], heading_path))
            return chunks

        group: list[Span] = []
        start: int | None = None
        used = 0
        first = True
        index = 0

        def flush() -> None:
            nonlocal group, start, used
            if group and start is not None:
                chunks.append(self._emit(doc, start, group[-1][1], heading_path))
            group = []
            start = None
            used = 0

        while index < len(paragraphs):
            p_start, p_end = paragraphs[index]
            para_tokens = count_tokens(text[p_start:p_end])
            prefix_tokens = 0
            if first and heading is not None:
                prefix_tokens = count_tokens(text[heading[0] : heading[1]])

            if group and used + para_tokens > self._size:
                previous, first = group, False
                flush()
                overlap_budget = min(self._overlap, self._size - para_tokens)
                overlap_start = _overlap_prefix(text, previous, overlap_budget)
                if overlap_start is not None:
                    for span_start, span_end in previous:
                        if span_start >= overlap_start:
                            group.append((span_start, span_end))
                            used += count_tokens(text[span_start:span_end])
                    start = overlap_start
                continue

            too_big = prefix_tokens + para_tokens > self._size
            if not group and first and heading is not None and too_big:
                chunks.append(self._emit(doc, heading[0], heading[1], heading_path))
                first = False
                continue

            if not group:
                start = heading[0] if (first and heading is not None) else p_start
                used += prefix_tokens
                if prefix_tokens:
                    first = False
            group.append((p_start, p_end))
            used += para_tokens
            index += 1

        flush()
        return chunks

    def _emit(self, doc: Document, start: int, end: int, heading_path: str) -> Chunk:
        return Chunk(
            chunk_id=_chunking.make_chunk_id(doc.doc_id, start, end, self._config_dump),
            doc_id=doc.doc_id,
            start=start,
            end=end,
            text=doc.text[start:end],
            metadata={"heading_path": heading_path},
        )


def _whole_document(doc: Document) -> Section:
    return Section(section_id=doc.doc_id, heading_path="", start=0, end=len(doc.text))


def _section_heading(text: str, sec_start: int, sec_end: int) -> tuple[Span | None, int]:
    """Heading line spanning a section's first line when it is a markdown heading."""
    line_end = text.find("\n", sec_start, sec_end)
    if line_end == -1:
        line_end = sec_end
    if _HEADING_LINE.match(text, sec_start, line_end):
        return (sec_start, line_end), min(line_end + 1, sec_end)
    return None, sec_start


def _paragraphs(text: str, start: int, end: int) -> list[Span]:
    """Blank-line-separated blocks in ``text[start:end]``, whitespace-trimmed per block."""
    spans: list[Span] = []
    pos = start
    while pos < end:
        match = _PARAGRAPH_BREAK.search(text, pos, end)
        block_end = match.start() if match else end
        block_start, trimmed_end = pos, block_end
        while block_start < trimmed_end and text[block_start].isspace():
            block_start += 1
        while trimmed_end > block_start and text[trimmed_end - 1].isspace():
            trimmed_end -= 1
        if block_start < trimmed_end:
            spans.append((block_start, trimmed_end))
        pos = match.end() if match else end
    return spans


def _overlap_prefix(text: str, previous: list[Span], budget: int) -> int | None:
    """Start offset of a whole-paragraph tail of ``previous`` within ``budget`` tokens."""
    if budget <= 0:
        return None
    total = 0
    start: int | None = None
    for span_start, span_end in reversed(previous):
        tokens = count_tokens(text[span_start:span_end])
        if total + tokens > budget:
            break
        total += tokens
        start = span_start
    return start
