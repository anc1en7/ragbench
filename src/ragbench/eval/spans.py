"""Chunker-agnostic span-overlap relevance scoring (Task 3.1, spec sections 5.6-5.7).

Relevance is defined on gold SOURCE spans, never on chunks, so chunkers of any
granularity are scored against the same labels. All spans are half-open ``[start, end)``
character offsets into the cleaned ``Document.text``.

Hit rule (spec section 5.6, "at least 20% of the chunk or 20% of the gold span"): a chunk
hits a gold span of the same document when the overlap is positive and

    overlap >= min_frac * len(chunk)   OR   overlap >= min_frac * len(gold span)

This is deliberately the OR (``min``) form: a small chunk wholly inside a large gold
section, or a large chunk wholly containing a small gold span, both count as hits. (The
implementation plan's ``max`` wording would reject both; the spec governs — see
docs/OPEN_QUESTIONS.md.)

``span_hits`` applies the nDCG crediting rule of spec section 5.7: each gold span is
credited at most once, to the FIRST ranked chunk that hits it. A chunk is relevant iff it
hits at least one not-yet-credited span; it then credits every span it hits. This stops
fine-grained chunkers from inflating nDCG by returning several slices of one section.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ragbench.types import Chunk, Hit

DEFAULT_MIN_OVERLAP_FRAC: float = 0.2


class SpanLike(Protocol):
    """Anything with ``doc_id``/``start``/``end`` (e.g. :class:`GoldSpan`)."""

    @property
    def doc_id(self) -> str: ...

    @property
    def start(self) -> int: ...

    @property
    def end(self) -> int: ...


def overlap_chars(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """Length of the intersection of half-open ranges ``[a_start, a_end)``, ``[b_start, b_end)``."""
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def _check_frac(min_frac: float) -> None:
    if not 0.0 <= min_frac <= 1.0:
        raise ValueError(f"min_frac must be within [0, 1], got {min_frac}")


def _hits_span(chunk: Chunk, span: SpanLike, min_frac: float) -> bool:
    if chunk.doc_id != span.doc_id:
        return False
    overlap = overlap_chars(chunk.start, chunk.end, span.start, span.end)
    if overlap <= 0:
        return False
    chunk_len = chunk.end - chunk.start
    span_len = span.end - span.start
    return overlap >= min_frac * chunk_len or overlap >= min_frac * span_len


def is_hit(
    chunk: Chunk, gold_spans: Sequence[SpanLike], min_frac: float = DEFAULT_MIN_OVERLAP_FRAC
) -> bool:
    """True when ``chunk`` hits ANY gold span under the overlap rule (module docstring)."""
    _check_frac(min_frac)
    return any(_hits_span(chunk, span, min_frac) for span in gold_spans)


def span_hits(
    ranked_hits: Sequence[Hit],
    gold_spans: Sequence[SpanLike],
    min_frac: float = DEFAULT_MIN_OVERLAP_FRAC,
) -> list[bool]:
    """Binary relevance per rank with first-hit-only crediting (module docstring)."""
    _check_frac(min_frac)
    credited = [False] * len(gold_spans)
    relevance: list[bool] = []
    for hit in ranked_hits:
        new_credit = False
        for idx, span in enumerate(gold_spans):
            if _hits_span(hit.chunk, span, min_frac):
                if not credited[idx]:
                    new_credit = True
                credited[idx] = True
        relevance.append(new_credit)
    return relevance
