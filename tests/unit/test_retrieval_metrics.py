"""Unit tests for retrieval metrics (Task 3.2) — hand-computed, offline.

Inputs are per-rank binary relevance lists as produced by ``span_hits`` (first-hit-only
crediting already applied) plus the number of gold spans for the question.
"""

import math

import pytest

from ragbench.eval.retrieval_metrics import (
    METRIC_NAMES,
    aggregate,
    mrr_at_k,
    ndcg_at_k,
    query_metrics,
    recall_at_k,
)

TOL = 1e-9
F, T = False, True


def _rels(hit_rank: int | None, length: int = 10) -> list[bool]:
    """Relevance list of ``length`` with a single relevant item at 1-based ``hit_rank``."""
    return [hit_rank is not None and i == hit_rank for i in range(1, length + 1)]


def test_metric_names_are_fixed():
    assert METRIC_NAMES == ("recall@1", "recall@3", "recall@5", "recall@10", "mrr@10", "ndcg@10")


def test_three_query_fixture_hand_computed():
    # q1: first hit at rank 2; q2: rank 1; q3: rank 7. One gold span each.
    per_query = [query_metrics(_rels(2), 1), query_metrics(_rels(1), 1), query_metrics(_rels(7), 1)]

    # Per query by hand:
    #   q1 rank 2: R@1=0 R@3=1 R@5=1 R@10=1  MRR=1/2  nDCG=(1/log2(3))/(1/log2(2)) = 1/log2(3)
    #   q2 rank 1: R@1=1 R@3=1 R@5=1 R@10=1  MRR=1    nDCG=1
    #   q3 rank 7: R@1=0 R@3=0 R@5=0 R@10=1  MRR=1/7  nDCG=1/log2(8) = 1/3
    q1 = per_query[0]
    assert q1["recall@1"] == 0.0
    assert q1["recall@3"] == 1.0
    assert abs(q1["mrr@10"] - 0.5) < TOL
    assert abs(q1["ndcg@10"] - 1 / math.log2(3)) < TOL
    assert abs(per_query[2]["ndcg@10"] - 1 / 3) < TOL

    means = aggregate(per_query)
    assert abs(means["recall@1"] - 1 / 3) < TOL
    assert abs(means["recall@3"] - 2 / 3) < TOL
    assert abs(means["recall@5"] - 2 / 3) < TOL
    assert abs(means["recall@10"] - 1.0) < TOL
    assert abs(means["mrr@10"] - (1 / 2 + 1 + 1 / 7) / 3) < TOL
    assert abs(means["ndcg@10"] - (1 / math.log2(3) + 1 + 1 / 3) / 3) < TOL


def test_no_hit_and_hit_beyond_cutoff_score_zero():
    for rels in (_rels(None), _rels(11, length=20)):
        metrics = query_metrics(rels, 1)
        assert all(metrics[name] == 0.0 for name in METRIC_NAMES)


def test_fewer_hits_than_k_is_fine():
    assert recall_at_k([F, T], 10) == 1.0
    assert recall_at_k([], 5) == 0.0
    assert mrr_at_k([], 10) == 0.0
    assert ndcg_at_k([], 1, 10) == 0.0


def test_recall_boundary_at_k():
    assert recall_at_k(_rels(5), 5) == 1.0
    assert recall_at_k(_rels(5), 4) == 0.0


def test_mrr_uses_first_relevant_rank_only():
    assert abs(mrr_at_k([F, F, T, T], 10) - 1 / 3) < TOL


def test_ndcg_two_gold_spans_hand_computed():
    # Relevant at ranks 2 and 3, two gold spans.
    # DCG  = 1/log2(3) + 1/log2(4) = 0.6309297535714575 + 0.5
    # IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.6309297535714575   (both spans at ranks 1-2)
    expected = (1 / math.log2(3) + 0.5) / (1 + 1 / math.log2(3))
    assert abs(ndcg_at_k([F, T, T], 2, 10) - expected) < TOL
    # Perfect ranking of two spans scores exactly 1.
    assert abs(ndcg_at_k([T, T, F], 2, 10) - 1.0) < TOL


def test_ndcg_ideal_is_capped_at_k():
    # 3 gold spans but k=2: IDCG uses min(3, 2) = 2 positions, so [T, T] is perfect.
    assert abs(ndcg_at_k([T, T], 3, 2) - 1.0) < TOL


def test_invalid_arguments_rejected():
    with pytest.raises(ValueError, match="k"):
        recall_at_k([T], 0)
    with pytest.raises(ValueError, match="n_gold"):
        ndcg_at_k([T], 0, 10)
    with pytest.raises(ValueError, match="empty"):
        aggregate([])
