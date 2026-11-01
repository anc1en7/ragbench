"""Unit tests for RRF fusion (Task 2.8, controller ruling 1) — offline, no network.

Hand-computed literals with the arithmetic spelled out in comments; ``k`` is the RRF
constant (default 60), never a result count — ``fuse`` returns ALL fused candidates and
the caller truncates.
"""

import pytest

from ragbench.retrieval import fuse
from ragbench.types import Chunk, Hit


def _hit(chunk_id: str, score: float, source: str) -> Hit:
    return Hit(
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id="docs/a.md",
            start=0,
            end=10,
            text=f"text of {chunk_id}",
        ),
        score=score,
        source=source,
    )


def test_fuse_hand_computed_two_lists_with_tie():
    dense = [_hit("beta", 0.9, "dense"), _hit("alpha", 0.8, "dense"), _hit("gamma", 0.7, "dense")]
    sparse = [_hit("alpha", 0.6, "sparse"), _hit("beta", 0.5, "sparse")]
    result = fuse([dense, sparse])
    # k=60 (RRF constant, NOT a result count — every candidate is returned).
    # 1-based ranks: dense = beta@1, alpha@2, gamma@3; sparse = alpha@1, beta@2.
    # alpha: 1/(60+2) [dense rank 2] + 1/(60+1) [sparse rank 1] = 1/62 + 1/61
    # beta:  1/(60+1) [dense rank 1] + 1/(60+2) [sparse rank 2] = 1/61 + 1/62  == alpha (TIE)
    # gamma: 1/(60+3) [dense rank 3] only; absent from sparse contributes nothing = 1/63
    # TIE alpha vs beta -> chunk_id ASC -> "alpha" < "beta".
    # expected scores: alpha = beta = 1/61 + 1/62 = 123/3782 ~= 0.0325225;
    # gamma = 1/63 ~= 0.0158730 (strictly smaller, so it ranks last).
    assert [h.chunk.chunk_id for h in result] == ["alpha", "beta", "gamma"]
    assert result[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert result[1].score == pytest.approx(1 / 61 + 1 / 62)
    assert result[2].score == pytest.approx(1 / 63)
    assert all(h.source == "fused" for h in result)


def test_fuse_empty_inputs_return_empty():
    assert fuse([]) == []
    assert fuse([[], []]) == []


def test_fuse_single_list_keeps_rank_order():
    hits = [_hit("a", 0.9, "dense"), _hit("b", 0.8, "dense"), _hit("c", 0.7, "dense")]
    result = fuse([hits])
    # ranks 1, 2, 3 -> 1/61, 1/62, 1/63; order unchanged (strictly decreasing scores).
    assert [h.chunk.chunk_id for h in result] == ["a", "b", "c"]
    assert result[0].score == pytest.approx(1 / 61)
    assert result[1].score == pytest.approx(1 / 62)
    assert result[2].score == pytest.approx(1 / 63)


def test_fuse_chunk_in_both_lists_sums_contributions():
    # Same chunk at rank 1 of each list: 1/(60+1) + 1/(60+1) = 2/61.
    result = fuse([[_hit("solo", 0.5, "dense")], [_hit("solo", 0.5, "sparse")]])
    assert len(result) == 1
    assert result[0].score == pytest.approx(2 / 61)
    assert result[0].source == "fused"


def test_fuse_duplicate_chunk_id_in_one_list_uses_first_occurrence_rank():
    # "dup" appears at positions 1 AND 3 of the same list: only the FIRST occurrence's
    # rank counts (1/61); the repeat contributes nothing. "other" at rank 2 -> 1/62.
    hits = [_hit("dup", 0.9, "dense"), _hit("other", 0.8, "dense"), _hit("dup", 0.1, "dense")]
    result = fuse([hits])
    assert [h.chunk.chunk_id for h in result] == ["dup", "other"]
    assert result[0].score == pytest.approx(1 / 61)
    assert result[1].score == pytest.approx(1 / 62)


def test_fuse_tie_resolved_by_chunk_id_ascending():
    # Symmetric ranks across two lists -> identical scores; "aaa" must beat "zzz".
    dense = [_hit("zzz", 0.9, "dense"), _hit("aaa", 0.8, "dense")]
    sparse = [_hit("aaa", 0.7, "sparse"), _hit("zzz", 0.6, "sparse")]
    # zzz: 1/61 + 1/62; aaa: 1/62 + 1/61 -> exact tie -> chunk_id ASC.
    result = fuse([dense, sparse])
    assert [h.chunk.chunk_id for h in result] == ["aaa", "zzz"]
    assert result[0].score == pytest.approx(result[1].score)


def test_fuse_k_is_rrf_constant_not_result_count():
    # k=1 changes the scores (1/(1+rank)) but still returns BOTH candidates — the
    # caller truncates, fusion never does.
    result = fuse([[_hit("a", 1.0, "dense"), _hit("b", 0.5, "dense")]], k=1)
    assert [h.chunk.chunk_id for h in result] == ["a", "b"]
    assert result[0].score == pytest.approx(1 / 2)
    assert result[1].score == pytest.approx(1 / 3)


def test_fuse_source_is_fused_for_every_candidate():
    result = fuse([[_hit("a", 0.9, "dense")], [_hit("b", 0.8, "sparse"), _hit("a", 0.7, "sparse")]])
    assert {h.source for h in result} == {"fused"}


def test_fuse_does_not_mutate_inputs():
    dense = [_hit("beta", 0.9, "dense"), _hit("alpha", 0.8, "dense")]
    sparse = [_hit("alpha", 0.6, "sparse")]
    dense_before = list(dense)
    sparse_before = list(sparse)
    fuse([dense, sparse])
    assert dense == dense_before
    assert sparse == sparse_before
    assert dense[0].source == "dense"
    assert sparse[0].source == "sparse"
