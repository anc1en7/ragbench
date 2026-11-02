"""Unit tests for the staged sweep: grid parsing, expansion, estimate, selection, resume.

Offline: the runner is driven with a fake ``evaluate_fn`` that writes real, schema-valid
run JSONs with scripted metrics, so resume and selection are exercised end to end.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragbench.config import PipelineConfig
from ragbench.eval.results import (
    CorpusInfo,
    GoldsetInfo,
    IndexStats,
    MetricCI,
    RunResult,
    StageLatency,
    run_path,
    write_run,
)
from ragbench.eval.retrieval_metrics import METRIC_NAMES
from ragbench.sweep.runner import run_sweep
from ragbench.sweep.stages import Grid, apply_overrides, estimate, expand_stage, load_grid, rank

ROOT = Path(__file__).resolve().parents[2]


def _result(cfg: PipelineConfig, ndcg: float, mrr: float = 0.5, split: str = "dev") -> RunResult:
    metrics = {name: MetricCI(mean=0.5, ci_lo=0.5, ci_hi=0.5) for name in METRIC_NAMES}
    metrics["ndcg@10"] = MetricCI(mean=ndcg, ci_lo=ndcg, ci_hi=ndcg)
    metrics["mrr@10"] = MetricCI(mean=mrr, ci_lo=mrr, ci_hi=mrr)
    return RunResult(
        config_id=cfg.config_id,
        config=cfg,
        seed=cfg.seed,
        split=split,
        corpus=CorpusInfo(name=cfg.corpus, repo="r", tag="t", commit_sha="c", sha256_of_files="s"),
        goldset=GoldsetInfo(n_questions=1, all_verified=True, provisional=False),
        package_versions={},
        hardware_ref="results/hardware.json",
        min_overlap_frac=0.2,
        bootstrap_resamples=10,
        bootstrap_seed=42,
        metrics=metrics,
        latency_ms={"total": StageLatency(p50=1.0, p95=2.0)},
        latency_passes=3,
        index=IndexStats(
            docs=1,
            chunks=1,
            embed_seconds=0.0,
            index_seconds=0.0,
            wall_seconds=0.0,
            peak_rss_bytes=1,
            chunks_per_sec=None,
            size_bytes=0,
        ),
        ann_sanity_recall=None,
        per_query=[],
        created_at="2026-11-02T00:00:00+00:00",
    )


TINY_GRID = {
    "corpus": "mini",
    "split": "dev",
    "seed": 42,
    "selection": {"metric": "ndcg@10", "tiebreak": ["mrr@10"]},
    "finalists": 2,
    "base": {
        "retrieval": {"mode": "dense"},
        "rerank": {"type": "none", "top_n": 50},
        "generation": {"enabled": False},
    },
    "stages": [
        {
            "name": "A",
            "top_n": 2,
            "set": {"backend": "pgvector"},
            "vary": [
                [
                    {"chunker": {"type": "fixed", "size_tokens": 256, "overlap_tokens": 32}},
                    {"chunker": {"type": "fixed", "size_tokens": 512, "overlap_tokens": 64}},
                ],
                [{"embedder": {"name": "m1"}}, {"embedder": {"name": "m2"}}],
            ],
        },
        {
            "name": "B",
            "from": "A",
            "top_n": 1,
            "vary": [[{"retrieval.mode": "dense"}, {"retrieval.mode": "hybrid"}]],
        },
    ],
}


# --- grid parsing and expansion --------------------------------------------------------


def test_repo_grid_loads_and_estimates_exact_counts():
    grid = load_grid(ROOT / "configs" / "grid.yaml")
    assert [s.name for s in grid.stages] == ["A", "B", "C", "D"]
    assert grid.selection.metric == "ndcg@10"
    # A: 5 chunkers x 3 embedders; B: 3 parents x 4; C: 3 x 3; D: 3 x 2
    assert estimate(grid) == [("A", 15), ("B", 12), ("C", 9), ("D", 6)]


def test_repo_grid_stage_a_is_dense_pgvector_and_unique():
    grid = load_grid(ROOT / "configs" / "grid.yaml")
    configs = expand_stage(grid, grid.stages[0], None)
    assert len(configs) == len({c.config_id for c in configs}) == 15
    assert {c.backend for c in configs} == {"pgvector"}
    assert {c.retrieval.mode for c in configs} == {"dense"}
    assert {c.corpus for c in configs} == {"fastapi_docs"}
    assert len({c.embedder.name for c in configs}) == 3


def test_apply_overrides_dotted_leaf_and_mapping_replace():
    base = {"retrieval": {"mode": "dense", "k_dense": 50}, "rerank": {"type": "x", "model": "m"}}
    out = apply_overrides(base, {"retrieval.mode": "hybrid", "rerank": {"type": "none"}})
    assert out["retrieval"] == {"mode": "hybrid", "k_dense": 50}  # leaf set, siblings kept
    assert out["rerank"] == {"type": "none"}  # mapping replaced wholesale
    assert base["retrieval"]["mode"] == "dense"  # input not mutated


def test_child_stage_expands_each_parent_by_every_variant():
    grid = load_grid(ROOT / "configs" / "grid.yaml")
    parents = expand_stage(grid, grid.stages[0], None)[:3]
    children = expand_stage(grid, grid.stages[1], parents)
    assert len(children) == 12
    assert sum(c.backend == "control_bm25" for c in children) == 3
    assert {c.retrieval.mode for c in children} == {"dense", "sparse", "hybrid"}
    # the dense variant of a parent IS the parent (same config_id -> reused, not re-run)
    assert parents[0].config_id in {c.config_id for c in children}


def test_only_backends_carries_other_parents_unchanged():
    grid = load_grid(ROOT / "configs" / "grid.yaml")
    stage_d = grid.stages[3]
    bm25 = PipelineConfig.model_validate(
        expand_stage(grid, grid.stages[0], None)[0].model_dump()
        | {"backend": "control_bm25", "retrieval": {"mode": "sparse"}}
    )
    dense = expand_stage(grid, grid.stages[0], None)[1]
    children = expand_stage(grid, stage_d, [bm25, dense])
    assert bm25.config_id in {c.config_id for c in children}
    assert sum(c.backend == "control_bm25" for c in children) == 1
    assert {c.backend for c in children if c.backend != "control_bm25"} == {"pgvector", "qdrant"}


def test_grid_rejects_unknown_parent_stage_and_bad_metric():
    bad = json.loads(json.dumps(TINY_GRID))
    bad["stages"][1]["from"] = "Z"
    with pytest.raises(ValueError, match="Z"):
        Grid.model_validate(bad)
    bad = json.loads(json.dumps(TINY_GRID))
    bad["selection"]["metric"] = "accuracy"
    with pytest.raises(ValueError, match="accuracy"):
        Grid.model_validate(bad)


# --- selection -------------------------------------------------------------------------


def test_rank_by_metric_then_tiebreak_then_config_id():
    grid = Grid.model_validate(TINY_GRID)
    cfgs = expand_stage(grid, grid.stages[0], None)
    results = [
        _result(cfgs[0], ndcg=0.7, mrr=0.1),
        _result(cfgs[1], ndcg=0.9, mrr=0.1),
        _result(cfgs[2], ndcg=0.7, mrr=0.9),  # ties cfgs[0] on ndcg, wins on mrr
        _result(cfgs[3], ndcg=0.7, mrr=0.1),  # full tie with cfgs[0] -> config_id order
    ]
    ranked = rank(results, grid.selection)
    assert ranked[0].config_id == cfgs[1].config_id
    assert ranked[1].config_id == cfgs[2].config_id
    assert [r.config_id for r in ranked[2:]] == sorted([cfgs[0].config_id, cfgs[3].config_id])


# --- runner: selection flow, resume, interruption --------------------------------------


class FakeEvaluate:
    """Writes a real run JSON; ndcg is scripted per config so selection is predictable."""

    def __init__(self, results_dir: Path, score: dict[str, float] | None = None) -> None:
        self.results_dir = results_dir
        self.score = score or {}
        self.calls: list[str] = []
        self.raise_on: str | None = None

    def __call__(self, cfg: PipelineConfig, split: str) -> RunResult:
        if cfg.config_id == self.raise_on:
            raise KeyboardInterrupt
        self.calls.append(cfg.config_id)
        ndcg = self.score.get(cfg.config_id, 0.5)
        if cfg.retrieval.mode == "hybrid":
            ndcg += 0.2
        result = _result(cfg, ndcg=ndcg, split=split)
        write_run(result, self.results_dir)
        return result


def test_run_sweep_selects_top_n_and_expands_children(tmp_path: Path):
    grid = Grid.model_validate(TINY_GRID)
    a_cfgs = expand_stage(grid, grid.stages[0], None)
    scores = {a_cfgs[2].config_id: 0.9, a_cfgs[1].config_id: 0.8}
    fake = FakeEvaluate(tmp_path, scores)

    report = run_sweep(grid, tmp_path, evaluate_fn=fake)

    stage_a, stage_b = report.stages
    assert stage_a.selected == [a_cfgs[2].config_id, a_cfgs[1].config_id]
    # B takes A's top_n (2) parents x {dense (reused: it IS the parent), hybrid (new)}
    assert stage_b.ran == 2 and stage_b.reused == 2
    assert len(fake.calls) == 4 + 2
    assert report.total_ran == 6
    assert len(stage_b.selected) == 1  # B's own top_n
    # finalists = top `finalists` (2) over every config of every stage: the scripted
    # A configs (0.9, 0.8) beat the unscripted hybrid children (0.5 + 0.2 = 0.7)
    assert report.finalists == [a_cfgs[2].config_id, a_cfgs[1].config_id]
    state = json.loads((tmp_path / ".sweep_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "complete"


def test_run_sweep_skips_completed_runs(tmp_path: Path):
    grid = Grid.model_validate(TINY_GRID)
    a_cfgs = expand_stage(grid, grid.stages[0], None)
    write_run(_result(a_cfgs[0], ndcg=0.1), tmp_path)  # pre-existing completed run
    fake = FakeEvaluate(tmp_path)

    report = run_sweep(grid, tmp_path, evaluate_fn=fake)

    assert a_cfgs[0].config_id not in fake.calls
    assert report.stages[0].reused == 1 and report.stages[0].ran == 3


def test_interrupt_leaves_resumable_state(tmp_path: Path):
    grid = Grid.model_validate(TINY_GRID)
    a_cfgs = expand_stage(grid, grid.stages[0], None)
    fake = FakeEvaluate(tmp_path)
    fake.raise_on = a_cfgs[2].config_id

    with pytest.raises(KeyboardInterrupt):
        run_sweep(grid, tmp_path, evaluate_fn=fake)

    state = json.loads((tmp_path / ".sweep_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "interrupted"
    assert state["stage"] == "A"
    assert set(state["completed"]) == {a_cfgs[0].config_id, a_cfgs[1].config_id}
    assert run_path(tmp_path, a_cfgs[1].config_id, "dev").exists()

    fake.raise_on = None
    fake.calls.clear()
    report = run_sweep(grid, tmp_path, evaluate_fn=fake)
    assert a_cfgs[0].config_id not in fake.calls and a_cfgs[1].config_id not in fake.calls
    assert report.stages[0].reused == 2


# --- cold throughput benchmark ---------------------------------------------------------


def test_measure_throughput_warms_up_then_times_full_batch(tmp_path: Path):
    from ragbench.sweep.throughput import measure_throughput, write_throughput

    calls: list[int] = []
    stats = measure_throughput(lambda batch: calls.append(len(batch)), ["t"] * 20, warmup=4)
    assert calls == [4, 20]  # warm-up excluded from timing, then the whole set
    assert stats["chunks"] == 20.0 and stats["chunks_per_sec"] > 0

    path = write_throughput(tmp_path / "tp.json", {"m1": stats}, "src")
    write_throughput(path, {"m2": stats}, "src")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data["embedders"]) == {"m1", "m2"}  # merged, not overwritten
