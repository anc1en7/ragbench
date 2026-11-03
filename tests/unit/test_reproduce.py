"""Unit tests for the reproduce plan and comparisons (Task 6.3)."""

from __future__ import annotations

from pathlib import Path

from ragbench.config import load_pipeline_config
from ragbench.eval.results import load_run
from ragbench.reproduce import (
    ROOT,
    compare_runs,
    plan,
    readme_is_current,
    write_finalist_configs,
)


def _names(**kw: object) -> list[str]:
    return [s.name for s in plan(scratch=Path("s"), **kw)]  # type: ignore[arg-type]


def test_fast_plan_never_sweeps() -> None:
    names = _names(finalists=False, full=False, ragas=False)
    assert names == ["fetch + verify corpus, build documents", "regression gate on mini fixture"]


def test_finalists_plan_evaluates_each_config_into_scratch() -> None:
    steps = plan(
        finalists=True, full=False, ragas=False, scratch=Path("s"), configs=[Path("a.yaml")]
    )
    assert steps[-1].argv[-2:] == ["--results-dir", str(Path("s") / "results")]
    assert "--split" in steps[-1].argv


def test_full_plan_skips_ragas_unless_asked() -> None:
    steps = plan(finalists=False, full=True, ragas=False, scratch=Path("s"))
    assert steps[-1].argv[-1] == "--skip-ragas"
    steps = plan(finalists=False, full=True, ragas=True, scratch=Path("s"))
    assert "--skip-ragas" not in steps[-1].argv


def test_finalist_configs_round_trip_to_same_config_id(tmp_path: Path) -> None:
    paths = write_finalist_configs(ROOT / "results", tmp_path)
    assert len(paths) == 5
    for path in paths:
        assert load_pipeline_config(path).config_id == path.stem


def test_compare_runs_identical_is_within_tolerance() -> None:
    run = load_run(ROOT / "results" / "runs" / "4443d42bc98a__test.json")
    assert all(good for *_, good in compare_runs(run, run, 0.0))


def test_committed_readme_is_current() -> None:
    assert readme_is_current()
