"""Unit tests for the CI regression gate comparison (Task 5.3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragbench.eval.regression import compare, load_thresholds

ROOT = Path(__file__).resolve().parents[2]


def test_drop_within_threshold_passes() -> None:
    (check,) = compare({"recall@5": 0.90}, {"recall@5": 0.88}, {"recall@5": 0.03})
    assert check.passed
    assert check.drop == pytest.approx(0.02)


def test_drop_exactly_at_threshold_passes() -> None:
    (check,) = compare({"recall@5": 0.90}, {"recall@5": 0.87}, {"recall@5": 0.03})
    assert check.passed


def test_drop_past_threshold_fails() -> None:
    (check,) = compare({"recall@5": 0.90}, {"recall@5": 0.80}, {"recall@5": 0.03})
    assert not check.passed


def test_improvement_passes() -> None:
    (check,) = compare({"mrr@10": 0.5}, {"mrr@10": 0.9}, {"mrr@10": 0.03})
    assert check.passed


def test_missing_metric_raises() -> None:
    with pytest.raises(ValueError, match="baseline has no value"):
        compare({}, {"recall@5": 0.9}, {"recall@5": 0.03})


def test_load_thresholds(tmp_path: Path) -> None:
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"max_drop": {"recall@5": 0.03}}), encoding="utf-8")
    assert load_thresholds(path) == {"recall@5": 0.03}
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="max_drop"):
        load_thresholds(path)


def test_committed_thresholds_and_baseline_are_consistent() -> None:
    thresholds = load_thresholds(ROOT / "configs" / "regression_thresholds.json")
    assert thresholds["recall@5"] == pytest.approx(0.03)
    baseline = json.loads((ROOT / "eval" / "baseline.json").read_text(encoding="utf-8"))
    assert set(thresholds) <= set(baseline)
    assert baseline["corpus"] == "mini"
