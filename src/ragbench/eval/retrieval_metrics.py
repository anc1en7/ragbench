"""Per-query retrieval metrics and their aggregation (Task 3.2, spec section 5.7).

Every function takes a per-rank binary relevance list as produced by
:func:`ragbench.eval.spans.span_hits` (first-hit-only crediting already applied, so a
gold span is never counted twice) and, for nDCG, the number of gold spans.

- ``recall@k``: 1.0 when any of the top-k ranks is relevant, else 0.0 (spec: "fraction
  of questions with at least one gold span hit in the top-k" after averaging).
- ``mrr@k``: reciprocal 1-based rank of the first relevant item within the top-k.
- ``ndcg@k``: binary-gain DCG ``sum(rel_i / log2(i + 1))`` over the top-k, divided by the
  ideal DCG of ``min(n_gold, k)`` relevant items at ranks ``1..min(n_gold, k)``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

RECALL_KS: tuple[int, ...] = (1, 3, 5, 10)
CUTOFF: int = 10
METRIC_NAMES: tuple[str, ...] = (
    *(f"recall@{k}" for k in RECALL_KS),
    f"mrr@{CUTOFF}",
    f"ndcg@{CUTOFF}",
)


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")


def recall_at_k(relevance: Sequence[bool], k: int) -> float:
    """1.0 if any of the first ``k`` ranks is relevant, else 0.0."""
    _check_k(k)
    return 1.0 if any(relevance[:k]) else 0.0


def mrr_at_k(relevance: Sequence[bool], k: int = CUTOFF) -> float:
    """Reciprocal rank of the first relevant item within the first ``k`` ranks (0.0 if none)."""
    _check_k(k)
    for rank, relevant in enumerate(relevance[:k], start=1):
        if relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevance: Sequence[bool], n_gold: int, k: int = CUTOFF) -> float:
    """Binary-relevance nDCG@k with an ideal of ``min(n_gold, k)`` leading hits."""
    _check_k(k)
    if n_gold < 1:
        raise ValueError(f"n_gold must be >= 1, got {n_gold}")
    dcg = sum(1.0 / math.log2(rank + 1) for rank, rel in enumerate(relevance[:k], start=1) if rel)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(n_gold, k) + 1))
    return dcg / idcg


def query_metrics(relevance: Sequence[bool], n_gold: int) -> dict[str, float]:
    """All :data:`METRIC_NAMES` for one question."""
    metrics = {f"recall@{k}": recall_at_k(relevance, k) for k in RECALL_KS}
    metrics[f"mrr@{CUTOFF}"] = mrr_at_k(relevance, CUTOFF)
    metrics[f"ndcg@{CUTOFF}"] = ndcg_at_k(relevance, n_gold, CUTOFF)
    return metrics


def aggregate(per_query: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Mean of every metric over questions."""
    if not per_query:
        raise ValueError("cannot aggregate an empty list of per-query metrics")
    return {
        name: sum(metrics[name] for metrics in per_query) / len(per_query) for name in METRIC_NAMES
    }
