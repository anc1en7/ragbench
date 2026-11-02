"""Unit tests for seeded bootstrap confidence intervals (Task 3.2) — offline."""

import numpy as np
import pytest

from ragbench.eval.stats import bootstrap_ci, paired_bootstrap_ci

VALUES = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.5, 0.25, 1.0, 0.0, 1.0, 0.75]


def test_bootstrap_ci_brackets_the_mean():
    lo, hi = bootstrap_ci(VALUES)
    mean = float(np.mean(VALUES))
    assert lo <= mean <= hi
    assert lo < hi  # non-degenerate data gives a real interval
    assert lo >= 0.0 and hi <= 1.0


def test_bootstrap_ci_is_deterministic_for_a_seed():
    assert bootstrap_ci(VALUES, seed=42) == bootstrap_ci(VALUES, seed=42)
    assert bootstrap_ci(VALUES, seed=42) != bootstrap_ci(VALUES, seed=7)


def test_bootstrap_ci_constant_values_collapse():
    assert bootstrap_ci([0.5] * 10) == (0.5, 0.5)


def test_bootstrap_ci_matches_reference_percentile_procedure():
    # Independent re-implementation of the documented procedure: resample indices with
    # numpy default_rng(seed).integers(0, n, (n_resamples, n)), mean per row, percentiles
    # 2.5 / 97.5 (linear interpolation).
    arr = np.asarray(VALUES, dtype=np.float64)
    rng = np.random.default_rng(42)
    idx = rng.integers(0, arr.size, size=(1000, arr.size))
    means = arr[idx].mean(axis=1)
    expected = (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))
    assert bootstrap_ci(VALUES, n_resamples=1000, seed=42) == expected


def test_wider_confidence_gives_wider_interval():
    lo95, hi95 = bootstrap_ci(VALUES, confidence=0.95)
    lo80, hi80 = bootstrap_ci(VALUES, confidence=0.80)
    assert lo95 <= lo80 and hi80 <= hi95


def test_paired_bootstrap_identical_systems_is_zero():
    assert paired_bootstrap_ci(VALUES, VALUES) == (0.0, 0.0)


def test_paired_bootstrap_constant_shift():
    shifted = [v + 0.25 for v in VALUES]
    lo, hi = paired_bootstrap_ci(shifted, VALUES)
    assert abs(lo - 0.25) < 1e-12 and abs(hi - 0.25) < 1e-12


def test_paired_bootstrap_ci_of_mean_difference_brackets_it():
    a = [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    b = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    lo, hi = paired_bootstrap_ci(a, b)
    diff = float(np.mean(np.asarray(a) - np.asarray(b)))
    assert lo <= diff <= hi
    assert paired_bootstrap_ci(a, b) == paired_bootstrap_ci(a, b)  # deterministic


def test_paired_bootstrap_equals_bootstrap_of_differences():
    a = [0.9, 0.1, 0.4, 0.8, 0.3]
    b = [0.5, 0.2, 0.4, 0.1, 0.3]
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    assert paired_bootstrap_ci(a, b, seed=3) == bootstrap_ci(diffs, seed=3)


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError, match="empty"):
        bootstrap_ci([])
    with pytest.raises(ValueError, match="n_resamples"):
        bootstrap_ci(VALUES, n_resamples=0)
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_ci(VALUES, confidence=1.0)
    with pytest.raises(ValueError, match="length"):
        paired_bootstrap_ci([1.0, 2.0], [1.0])
