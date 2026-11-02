"""Staged sweep runner with reuse and resume (Task 4.2, spec section 5.8).

For each stage in order: take the top ``top_n`` configs of the parent stage (dev split,
ranked by the grid's selection metric), expand the stage's configs, and evaluate each
one with ``evaluate_fn``. A config whose run file ``results/runs/{config_id}__{split}.json``
already exists is LOADED instead of re-run. That is the resume mechanism (the JSON files
are the source of truth) and it also reuses configs shared between stages: a stage-B
"dense" variant is the stage-A parent itself.

Every evaluated run already records config, seed, corpus SHA, package versions and the
hardware reference (``ragbench.eval.runner.evaluate``). Document embeddings are cached on
disk by ``(model, revision, text)``, so re-indexing a chunker/embedder pair already seen
costs no model compute.

``results/.sweep_state.json`` is written after every run (``status`` running /
interrupted / complete, current stage, completed config_ids). On ``KeyboardInterrupt``
the state is saved as ``interrupted`` and the interrupt re-raised; re-running the same
command resumes.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ragbench.config import PipelineConfig
from ragbench.eval.results import RunResult, Split, load_run, run_path
from ragbench.sweep.stages import Grid, expand_stage, rank

EvaluateFn = Callable[[PipelineConfig, Split], RunResult]
STATE_FILE = ".sweep_state.json"


@dataclass
class StageReport:
    name: str
    config_ids: list[str] = field(default_factory=list)
    ran: int = 0
    reused: int = 0
    selected: list[str] = field(default_factory=list)
    seconds: float = 0.0


@dataclass
class SweepReport:
    stages: list[StageReport]
    finalists: list[str]
    elapsed_seconds: float

    @property
    def total_ran(self) -> int:
        return sum(s.ran for s in self.stages)

    @property
    def total_reused(self) -> int:
        return sum(s.reused for s in self.stages)


def _grid_sha(grid: Grid) -> str:
    canonical = json.dumps(grid.model_dump(mode="json", by_alias=True), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


class _State:
    def __init__(self, results_dir: Path, grid: Grid) -> None:
        self.path = results_dir / STATE_FILE
        self.data: dict[str, object] = {
            "grid_sha": _grid_sha(grid),
            "status": "running",
            "stage": None,
            "completed": [],
            "started_at": datetime.now(UTC).isoformat(),
        }

    def save(self, **updates: object) -> None:
        self.data.update(updates)
        self.data["updated_at"] = datetime.now(UTC).isoformat()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")

    def complete(self, config_id: str) -> None:
        completed = self.data["completed"]
        assert isinstance(completed, list)
        if config_id not in completed:
            completed.append(config_id)
        self.save()


def run_sweep(
    grid: Grid,
    results_dir: Path,
    *,
    evaluate_fn: EvaluateFn | None = None,
    log: Callable[[str], None] = lambda _msg: None,
) -> SweepReport:
    """Run every stage of ``grid``; returns per-stage counts, selections and finalists."""
    if evaluate_fn is None:
        from ragbench.eval.runner import evaluate

        def evaluate_fn(cfg: PipelineConfig, split: Split) -> RunResult:
            return evaluate(cfg, split, results_dir=results_dir)[0]

    split = grid.split
    state = _State(results_dir, grid)
    started = time.perf_counter()
    stage_results: dict[str, list[RunResult]] = {}
    all_results: dict[str, RunResult] = {}
    reports: list[StageReport] = []

    try:
        for stage in grid.stages:
            stage_start = time.perf_counter()
            report = StageReport(name=stage.name)
            state.save(stage=stage.name)
            parents = None
            if stage.from_stage is not None:
                parent_stage = grid.stage(stage.from_stage)
                ranked = rank(stage_results[stage.from_stage], grid.selection)
                parents = [r.config for r in ranked[: parent_stage.top_n]]
            configs = expand_stage(grid, stage, parents)
            results: list[RunResult] = []
            for idx, cfg in enumerate(configs, start=1):
                path = run_path(results_dir, cfg.config_id, split)
                label = f"[{stage.name} {idx}/{len(configs)}] {cfg.config_id}"
                if path.exists():
                    result = load_run(path)
                    report.reused += 1
                    log(f"{label} reused")
                else:
                    log(f"{label} running ({_describe(cfg)})")
                    result = evaluate_fn(cfg, split)
                    report.ran += 1
                results.append(result)
                all_results[cfg.config_id] = result
                report.config_ids.append(cfg.config_id)
                state.complete(cfg.config_id)
            stage_results[stage.name] = results
            report.selected = [r.config_id for r in rank(results, grid.selection)[: stage.top_n]]
            report.seconds = time.perf_counter() - stage_start
            reports.append(report)
            log(
                f"stage {stage.name}: ran={report.ran} reused={report.reused} "
                f"top={report.selected} ({report.seconds:.0f}s)"
            )
    except KeyboardInterrupt:
        state.save(status="interrupted")
        raise

    finalists = [r.config_id for r in rank(list(all_results.values()), grid.selection)]
    state.save(status="complete", finalists=finalists[: grid.finalists])
    return SweepReport(
        stages=reports,
        finalists=finalists[: grid.finalists],
        elapsed_seconds=time.perf_counter() - started,
    )


def _describe(cfg: PipelineConfig) -> str:
    size = cfg.chunker.size_tokens if cfg.chunker.type != "semantic" else "sem"
    rerank = cfg.rerank.type if cfg.rerank.type == "none" else f"ce@{cfg.rerank.top_n}"
    return (
        f"{cfg.chunker.type}/{size} {cfg.embedder.name.split('/')[-1]} "
        f"{cfg.backend} {cfg.retrieval.mode} {rerank}"
    )
