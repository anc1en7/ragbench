"""Unit tests for chunker-agnostic span-overlap scoring (Task 3.1) — offline.

Rule (spec section 5.6): a chunk hits a gold span of the SAME document when
``overlap >= min_frac * len(chunk)`` OR ``overlap >= min_frac * len(gold span)``, i.e.
``overlap >= min(min_frac * chunk_len, min_frac * gold_len)`` with ``overlap > 0``.
Spans are half-open ``[start, end)`` character ranges.

``span_hits`` applies the nDCG rule of spec section 5.7: every gold span is credited at
most once, to the FIRST ranked chunk that hits it; a later chunk only counts as relevant
if it hits a gold span not yet credited.
"""

import pytest

from ragbench.eval.spans import is_hit, overlap_chars, span_hits
from ragbench.goldset.schema import GoldSpan
from ragbench.types import Chunk, Hit


def _chunk(start: int, end: int, doc_id: str = "a.md") -> Chunk:
    return Chunk(chunk_id=f"{doc_id}:{start}-{end}", doc_id=doc_id, start=start, end=end, text="x")


def _hit(start: int, end: int, doc_id: str = "a.md") -> Hit:
    return Hit(chunk=_chunk(start, end, doc_id), score=1.0, source="dense")


def _span(start: int, end: int, doc_id: str = "a.md") -> GoldSpan:
    return GoldSpan(doc_id=doc_id, start=start, end=end)


# --- overlap_chars ---------------------------------------------------------------------


def test_overlap_chars_half_open_ranges():
    assert overlap_chars(0, 100, 50, 150) == 50  # partial
    assert overlap_chars(0, 100, 100, 200) == 0  # touching ends do not overlap
    assert overlap_chars(10, 20, 0, 100) == 10  # contained
    assert overlap_chars(0, 10, 50, 60) == 0  # disjoint


# --- is_hit ----------------------------------------------------------------------------


def test_zero_overlap_is_never_a_hit():
    assert not is_hit(_chunk(0, 100), [_span(100, 200)], 0.2)
    # min_frac=0 still requires positive overlap
    assert not is_hit(_chunk(0, 100), [_span(200, 300)], 0.0)


def test_other_document_never_hits_even_with_identical_offsets():
    assert not is_hit(_chunk(0, 100, "a.md"), [_span(0, 100, "b.md")], 0.2)


def test_partial_overlap_hits_via_chunk_fraction():
    # chunk [0,100) len 100, gold [80,1080) len 1000; overlap 20.
    # 20 >= 0.2*100 = 20 (chunk side) -> hit even though 20 < 0.2*1000 = 200.
    assert is_hit(_chunk(0, 100), [_span(80, 1080)], 0.2)


def test_partial_overlap_hits_via_gold_fraction():
    # chunk [0,1000) len 1000, gold [990,1040) len 50; overlap 10.
    # 10 >= 0.2*50 = 10 (gold side) -> hit even though 10 < 0.2*1000 = 200.
    assert is_hit(_chunk(0, 1000), [_span(990, 1040)], 0.2)


def test_partial_overlap_below_both_thresholds_misses():
    # chunk [0,100) len 100, gold [81,281) len 200; overlap 19.
    # 19 < 0.2*100 = 20 and 19 < 0.2*200 = 40 -> miss.
    assert not is_hit(_chunk(0, 100), [_span(81, 281)], 0.2)


def test_small_chunk_contained_in_large_gold_section_hits():
    # The case a max()-rule would get wrong: overlap 200 = whole chunk,
    # 200 >= 0.2*200 = 40 even though 200 < 0.2*5000 = 1000.
    assert is_hit(_chunk(1000, 1200), [_span(0, 5000)], 0.2)


def test_large_chunk_containing_small_gold_span_hits():
    # overlap 100 = whole gold span; 100 >= 0.2*100 = 20 although 100 < 0.2*4000 = 800.
    assert is_hit(_chunk(0, 4000), [_span(500, 600)], 0.2)


def test_multiple_gold_spans_any_one_suffices():
    spans = [_span(0, 50), _span(500, 600, "b.md"), _span(900, 1000)]
    assert is_hit(_chunk(950, 1050), spans, 0.2)  # overlaps third span by 50 of 100
    assert not is_hit(_chunk(200, 300), spans, 0.2)


def test_min_frac_is_configurable():
    # chunk [0,100), gold [70,570): overlap 30; 0.2 -> 30 >= 20 hit; 0.5 -> 30 < 50 and < 250.
    assert is_hit(_chunk(0, 100), [_span(70, 570)], 0.2)
    assert not is_hit(_chunk(0, 100), [_span(70, 570)], 0.5)


def test_min_frac_out_of_range_rejected():
    with pytest.raises(ValueError, match="min_frac"):
        is_hit(_chunk(0, 10), [_span(0, 10)], 1.5)
    with pytest.raises(ValueError, match="min_frac"):
        is_hit(_chunk(0, 10), [_span(0, 10)], -0.1)


# --- span_hits -------------------------------------------------------------------------


def test_span_hits_marks_relevant_ranks():
    hits = [_hit(500, 600), _hit(0, 100), _hit(2000, 2100)]
    assert span_hits(hits, [_span(0, 100)]) == [False, True, False]


def test_span_hits_credits_each_gold_span_once_first_hit_only():
    # Three chunks all inside the single gold span: only the first is relevant, so a
    # fine-grained chunker cannot inflate nDCG by returning many slices of one section.
    hits = [_hit(0, 100), _hit(100, 200), _hit(200, 300)]
    assert span_hits(hits, [_span(0, 1000)]) == [True, False, False]


def test_span_hits_two_gold_spans_each_credited_once():
    spans = [_span(0, 100), _span(1000, 1100)]
    hits = [_hit(0, 100), _hit(10, 90), _hit(1000, 1100), _hit(1050, 1100)]
    assert span_hits(hits, spans) == [True, False, True, False]


def test_span_hits_chunk_covering_two_spans_credits_both():
    # One chunk hits both spans: it is relevant once and both spans are credited, so a
    # later chunk hitting either span is no longer relevant.
    spans = [_span(0, 100), _span(150, 250)]
    hits = [_hit(0, 250), _hit(150, 250)]
    assert span_hits(hits, spans) == [True, False]


def test_span_hits_empty_inputs():
    assert span_hits([], [_span(0, 10)]) == []
    assert span_hits([_hit(0, 10)], []) == [False]


def test_span_hits_passes_min_frac_through():
    hits = [_hit(0, 100)]
    assert span_hits(hits, [_span(70, 570)], min_frac=0.2) == [True]
    assert span_hits(hits, [_span(70, 570)], min_frac=0.5) == [False]
