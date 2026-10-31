"""Semantic chunker: sentence-level cosine breakpoints within min/max bounds (spec 5.2, ruling 5).

Algorithm: split ``doc.text`` into sentences, embed them in one batch via
``embedder.embed_documents`` (the unit of work is a document's sentences, so document
embedding fits; ``embed_query`` is never called), open a breakpoint wherever cosine
similarity between consecutive sentence embeddings falls below ``threshold`` (breakpoints
are hard chunk boundaries), pack each breakpoint-delimited run greedily under
``max_tokens``, force-split any single sentence exceeding ``max_tokens`` into
non-overlapping cl100k token windows, then absorb the following chunk into any chunk still
below ``min_tokens`` whenever that merge stays within ``max_tokens`` — otherwise the
undersized chunk keeps its size (a short tail below ``min_tokens`` is expected). Sentence
splitting uses a simple regex (``[^.!?]*[.!?]+|[^.!?]+$``) with per-sentence whitespace
trimming: abbreviations are not handled — documented simplicity per the ruling.
``Chunk.metadata`` stays empty for this chunker.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from ragbench import (
    chunking as _chunking,  # make_chunk_id resolved at call time (import order safe)
)
from ragbench.chunking.tokenizer import count_tokens, token_char_offsets
from ragbench.config import ChunkerCfg
from ragbench.types import Chunk, Document, Embedder

_SENTENCE = re.compile(r"[^.!?]*[.!?]+|[^.!?]+$")

Span = tuple[int, int]


class SemanticChunker:
    """Consecutive-sentence cosine-similarity segmentation with token bounds."""

    def __init__(self, cfg: ChunkerCfg, embedder: Embedder) -> None:
        if cfg.type != "semantic":
            raise ValueError(f"SemanticChunker requires chunker type 'semantic', got '{cfg.type}'")
        if embedder is None:
            raise ValueError("SemanticChunker requires a non-None embedder")
        self._min = cfg.min_tokens
        self._max = cfg.max_tokens
        self._threshold = 0.5 if cfg.threshold is None else cfg.threshold
        self._embedder = embedder
        self._config_dump: dict[str, Any] = cfg.model_dump(mode="json")

    def chunk(self, doc: Document) -> list[Chunk]:
        units = self._units(doc.text)
        if not units:
            return []
        vectors = self._embedder.embed_documents([doc.text[s:e] for s, e in units])
        similarities = _consecutive_cosine(vectors)
        groups: list[list[Span]] = [[units[0]]]
        for index in range(1, len(units)):
            if similarities[index - 1] < self._threshold:
                groups.append([units[index]])
            else:
                groups[-1].append(units[index])
        spans = self._merge(doc.text, groups)
        return [
            Chunk(
                chunk_id=_chunking.make_chunk_id(doc.doc_id, start, end, self._config_dump),
                doc_id=doc.doc_id,
                start=start,
                end=end,
                text=doc.text[start:end],
            )
            for start, end in spans
        ]

    def _units(self, text: str) -> list[Span]:
        """Sentence spans, with oversized sentences force-split into token windows."""
        units: list[Span] = []
        for start, end in _split_sentences(text):
            if self._max is not None and count_tokens(text[start:end]) > self._max:
                units.extend(_force_split(text, start, end, self._max))
            else:
                units.append((start, end))
        return units

    def _merge(self, text: str, groups: list[list[Span]]) -> list[Span]:
        sized: list[list[Span]] = []
        for group in groups:
            current: list[Span] = []
            for span in group:
                if current and self._over_max(text, current + [span]):
                    sized.append(current)
                    current = [span]
                else:
                    current.append(span)
            if current:
                sized.append(current)
        return [(group[0][0], group[-1][1]) for group in self._lift_undersized(text, sized)]

    def _over_max(self, text: str, group: list[Span]) -> bool:
        if self._max is None:
            return False
        return count_tokens(text[group[0][0] : group[-1][1]]) > self._max

    def _lift_undersized(self, text: str, merged: list[list[Span]]) -> list[list[Span]]:
        """Absorb the following group into chunks below ``min_tokens`` when the ceiling permits."""
        if self._min is None:
            return merged
        changed = True
        while changed:
            changed = False
            out: list[list[Span]] = []
            index = 0
            while index < len(merged):
                group = merged[index]
                tokens = count_tokens(text[group[0][0] : group[-1][1]])
                if (
                    tokens < self._min
                    and index + 1 < len(merged)
                    and not self._over_max(text, group + merged[index + 1])
                ):
                    out.append(group + merged[index + 1])
                    index += 2
                    changed = True
                else:
                    out.append(group)
                    index += 1
            merged = out
        return merged


def _split_sentences(text: str) -> list[Span]:
    """Regex sentence splitter; each span is whitespace-trimmed and an exact text slice."""
    spans: list[Span] = []
    for match in _SENTENCE.finditer(text):
        start, end = match.start(), match.end()
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            spans.append((start, end))
    return spans


def _force_split(text: str, start: int, end: int, max_tokens: int) -> list[Span]:
    """Non-overlapping ``max_tokens``-token windows covering ``text[start:end]`` exactly."""
    offsets = token_char_offsets(text[start:end])
    n_tokens = len(offsets) - 1
    spans: list[Span] = []
    first = 0
    while first < n_tokens:
        last = min(first + max_tokens, n_tokens)
        span_start = start + offsets[first]
        span_end = start + offsets[last]
        if span_start < span_end:
            spans.append((span_start, span_end))
        first = last
    return spans


def _consecutive_cosine(vectors: np.ndarray) -> np.ndarray:
    """Cosine similarity between each adjacent pair of embedding rows."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    unit = vectors / norms
    return np.sum(unit[:-1] * unit[1:], axis=1)
