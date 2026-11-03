"""One-command reproduction from a clean clone (Task 6.3).

Three depths, each a superset of the previous one:

- **fast** (default, CPU, ~2-5 min plus first model download): fetch the pinned corpus and
  verify it against ``corpus.lock``, rebuild ``data/documents.jsonl``, run the CI regression
  gate on the mini fixture against ``eval/baseline.json``, and check that the README blocks
  regenerate identically from the committed ``results/``.
- **finalists** (``--finalists``): also re-evaluate the 5 committed test finalists into a
  scratch results dir and compare each metric with the committed run within ``--tolerance``.
  On CPU the cross-encoder makes this slow (tens of minutes); a GPU takes a few minutes.
- **full** (``--full``): also re-run the whole staged sweep and finalist selection into the
  scratch dir (``--ragas`` adds RAGAS, which needs the LLM endpoint).

Requires ``make up`` (pgvector + Qdrant) for every depth. Never writes to ``results/``.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from ragbench.eval.results import RunResult, load_run
from ragbench.eval.retrieval_metrics import METRIC_NAMES

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Step:
    name: str
    argv: list[str]


def _cli(*args: str) -> list[str]:
    return [sys.executable, "-m", "ragbench.cli", *args]


def plan(
    *, finalists: bool, full: bool, ragas: bool, scratch: Path, configs: Sequence[Path] = ()
) -> list[Step]:
    """The ordered command list for the chosen depth (pure; unit-tested)."""
    steps = [
        Step("fetch + verify corpus, build documents", _cli("corpus", "ingest")),
        Step("regression gate on mini fixture", _cli("regression-check")),
    ]
    results = scratch / "results"
    if finalists or full:
        for cfg in configs:
            steps.append(
                Step(
                    f"re-evaluate finalist {cfg.stem} on test",
                    _cli(
                        "eval",
                        "--config",
                        str(cfg),
                        "--split",
                        "test",
                        "--results-dir",
                        str(results),
                    ),
                )
            )
    if full:
        steps.append(Step("staged sweep on dev", _cli("sweep", "--results-dir", str(results))))
        finalize = _cli("finalize", "--results-dir", str(results))
        if not ragas:
            finalize.append("--skip-ragas")
        steps.append(Step("select finalists, evaluate on test", finalize))
    return steps


def write_finalist_configs(results_dir: Path, out_dir: Path) -> list[Path]:
    """Write each committed test finalist's exact config as YAML; return the paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for run_file in sorted((results_dir / "runs").glob("*__test.json")):
        run = load_run(run_file)
        if run.config.corpus != "fastapi_docs":
            continue
        path = out_dir / f"{run.config_id}.yaml"
        path.write_text(yaml.safe_dump(run.config.model_dump(mode="json")), encoding="utf-8")
        paths.append(path)
    return paths


def compare_runs(
    committed: RunResult, fresh: RunResult, tolerance: float
) -> list[tuple[str, float, float, bool]]:
    """Per metric: ``(name, committed, fresh, within tolerance)``."""
    rows = []
    for name in METRIC_NAMES:
        a, b = committed.metrics[name].mean, fresh.metrics[name].mean
        rows.append((name, a, b, abs(a - b) <= tolerance))
    return rows


def readme_is_current(root: Path = ROOT) -> bool:
    from ragbench.report.readme import load_runs, render

    readme = (root / "README.md").read_text(encoding="utf-8")
    results = root / "results"
    return render(readme, load_runs(results, "dev"), load_runs(results, "test")) == readme


def run(
    *,
    finalists: bool,
    full: bool,
    ragas: bool,
    scratch: Path,
    tolerance: float,
    echo: Callable[[str], None] = print,
) -> bool:
    """Execute the plan; return True when every step and comparison passed."""
    configs = (
        write_finalist_configs(ROOT / "results", scratch / "configs") if finalists or full else []
    )
    ok = True
    for step in plan(finalists=finalists, full=full, ragas=ragas, scratch=scratch, configs=configs):
        echo(f"==> {step.name}")
        if subprocess.run(step.argv, cwd=ROOT, check=False).returncode != 0:
            echo(f"FAILED: {step.name}")
            return False

    echo("==> README blocks regenerate identically from results/")
    if readme_is_current():
        echo("README is current")
    else:
        echo("FAILED: README differs from `make report` output")
        ok = False

    for cfg in configs:
        committed = load_run(ROOT / "results" / "runs" / f"{cfg.stem}__test.json")
        fresh = load_run(scratch / "results" / "runs" / f"{cfg.stem}__test.json")
        for name, a, b, good in compare_runs(committed, fresh, tolerance):
            flag = "ok" if good else "DIFF"
            echo(f"{flag} {cfg.stem} {name}: committed={a:.4f} fresh={b:.4f}")
            ok = ok and good
    echo("REPRODUCED" if ok else "NOT REPRODUCED (see DIFF/FAILED lines)")
    return ok
