"""CI regression gate: compare a fresh run against ``eval/baseline.json`` (Task 5.3).

A metric regresses when ``baseline - current > max_drop`` (absolute). Thresholds live in
``configs/regression_thresholds.json`` as ``{"max_drop": {"recall@5": 0.03, ...}}``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MetricCheck:
    metric: str
    baseline: float
    current: float
    max_drop: float

    @property
    def drop(self) -> float:
        return self.baseline - self.current

    @property
    def passed(self) -> bool:
        # Small epsilon so float noise at exactly the threshold does not fail the gate.
        return self.drop <= self.max_drop + 1e-9


def load_thresholds(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    max_drop = data.get("max_drop")
    if not isinstance(max_drop, dict) or not max_drop:
        raise ValueError(f"{path}: expected a non-empty 'max_drop' object")
    return {str(name): float(value) for name, value in max_drop.items()}


def compare(
    baseline: dict[str, object], current: dict[str, float], thresholds: dict[str, float]
) -> list[MetricCheck]:
    """One :class:`MetricCheck` per thresholded metric; missing baseline values raise."""
    checks: list[MetricCheck] = []
    for metric, max_drop in thresholds.items():
        if metric not in baseline:
            raise ValueError(f"baseline has no value for {metric!r}")
        if metric not in current:
            raise ValueError(f"current run has no value for {metric!r}")
        value = baseline[metric]
        if not isinstance(value, int | float):
            raise ValueError(f"baseline {metric!r} is not a number: {value!r}")
        checks.append(MetricCheck(metric, float(value), current[metric], max_drop))
    return checks
