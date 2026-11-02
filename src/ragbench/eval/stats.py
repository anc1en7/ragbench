"""Seeded percentile-bootstrap confidence intervals (Task 3.2, spec section 5.7).

Procedure (documented so results are reproducible): with ``rng =
numpy.random.default_rng(seed)``, draw an index matrix ``rng.integers(0, n, (n_resamples,
n))`` — resampling questions with replacement — take each row's mean, and report the
``(1 - confidence) / 2`` and ``(1 + confidence) / 2`` percentiles (numpy's default linear
interpolation). Defaults: 1000 resamples, seed 42, 95%.

``paired_bootstrap_ci`` is the same procedure over per-question differences ``a - b``;
use it for config-vs-baseline comparisons (both evaluated on the same questions, same
order). A difference whose paired CI contains 0 must not be claimed (spec section 7).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

DEFAULT_RESAMPLES: int = 1000
DEFAULT_SEED: int = 42
DEFAULT_CONFIDENCE: float = 0.95


def bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[float, float]:
    """Percentile bootstrap ``(lo, hi)`` for the mean of ``values``."""
    if len(values) == 0:
        raise ValueError("cannot bootstrap an empty sequence")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be within (0, 1), got {confidence}")
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)
    alpha = 1.0 - confidence
    lo = float(np.percentile(means, 100.0 * alpha / 2.0))
    hi = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def paired_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[float, float]:
    """Percentile bootstrap ``(lo, hi)`` for ``mean(a - b)`` over paired questions."""
    if len(a) != len(b):
        raise ValueError(f"paired samples must have equal length, got {len(a)} and {len(b)}")
    diffs = [float(x) - float(y) for x, y in zip(a, b, strict=True)]
    return bootstrap_ci(diffs, n_resamples=n_resamples, seed=seed, confidence=confidence)
