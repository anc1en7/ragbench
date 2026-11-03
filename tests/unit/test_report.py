"""Unit tests for the README generator and Pareto figure (Task 6.1)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from ragbench.report.figures import frontier, pareto_svg
from ragbench.report.readme import NOT_YET_RUN, inject, load_runs, render

ROOT = Path(__file__).resolve().parents[2]
SKELETON = """# T

<!-- BEGIN:RESULTS TABLE -->
old
<!-- END:RESULTS TABLE -->

<!-- BEGIN:GENERATION TABLE -->
<!-- END:GENERATION TABLE -->

<!-- BEGIN:FINDINGS -->
<!-- END:FINDINGS -->
"""


@pytest.fixture
def results(tmp_path: Path) -> Path:
    """Two committed test runs (one with RAGAS) and their two dev counterparts."""
    runs = tmp_path / "runs"
    runs.mkdir()
    for name in ("4443d42bc98a__test.json", "185f47fb8795__test.json", "185f47fb8795__dev.json"):
        shutil.copy(ROOT / "results" / "runs" / name, runs / name)
    return tmp_path


def test_render_is_deterministic_and_uses_real_numbers(results: Path) -> None:
    dev, test = load_runs(results, "dev"), load_runs(results, "test")
    first = render(SKELETON, dev, test)
    assert first == render(SKELETON, dev, test)
    assert "old" not in first
    assert "| 1 | recursive_md-256 / bge-small-en-v1.5 / hybrid" in first
    assert "0.887 [0.839, 0.931]" in first  # best finalist nDCG@10 with CI
    assert "replies |" in first
    assert "**Best finalist on test:**" in first
    assert NOT_YET_RUN not in first


def test_empty_results_say_not_yet_run(tmp_path: Path) -> None:
    (tmp_path / "runs").mkdir()
    out = render(SKELETON, load_runs(tmp_path, "dev"), load_runs(tmp_path, "test"))
    assert out.count(NOT_YET_RUN) == 3


def test_inject_requires_markers() -> None:
    with pytest.raises(ValueError, match="FINDINGS"):
        inject("no markers", "FINDINGS", "x")


def test_inject_is_idempotent() -> None:
    once = inject(SKELETON, "FINDINGS", "- a")
    assert inject(once, "FINDINGS", "- a") == once


def test_frontier_keeps_only_undominated_points() -> None:
    pts = [(10.0, 0.5, "a"), (20.0, 0.4, "b"), (30.0, 0.9, "c"), (5.0, 0.3, "d")]
    assert [k for _, _, k in frontier(pts)] == ["d", "a", "c"]


def test_pareto_svg(results: Path) -> None:
    svg = pareto_svg(load_runs(results, "dev"))
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert "<circle" in svg and "Pareto frontier" in svg
    assert "not yet run" in pareto_svg([])


def test_committed_readme_is_up_to_date() -> None:
    """README blocks must match a fresh `make report` over the committed results."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    results = ROOT / "results"
    regenerated = render(readme, load_runs(results, "dev"), load_runs(results, "test"))
    assert regenerated == readme, "README is stale: run `make report`"
