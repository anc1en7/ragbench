"""Unit tests for dev->test finalist flow, leaderboard, ablations and findings (Task 4.4)."""

from __future__ import annotations

from pathlib import Path

from ragbench.config import PipelineConfig
from ragbench.eval.results import (
    CorpusInfo,
    GoldsetInfo,
    IndexStats,
    MetricCI,
    QueryResult,
    RunResult,
    StageLatency,
    run_path,
    write_run,
)
from ragbench.eval.retrieval_metrics import METRIC_NAMES
from ragbench.sweep.select import (
    ablations,
    attach_generation,
    describe,
    evaluate_finalists,
    findings_lines,
    leaderboard,
    single_factor,
)
from ragbench.sweep.stages import Grid

BASE = {
    "corpus": "mini",
    "chunker": {"type": "recursive_md", "size_tokens": 512, "overlap_tokens": 64},
    "embedder": {"name": "BAAI/bge-small-en-v1.5"},
    "backend": "pgvector",
    "retrieval": {"mode": "dense"},
    "rerank": {"type": "none"},
    "generation": {"enabled": False},
}
GRID = Grid.model_validate(
    {
        "corpus": "mini",
        "selection": {"metric": "ndcg@10", "tiebreak": ["mrr@10"]},
        "finalists": 2,
        "base": {},
        "stages": [{"name": "A", "vary": []}],
    }
)


def _cfg(**over: object) -> PipelineConfig:
    return PipelineConfig.model_validate(BASE | over)


def _run(
    cfg: PipelineConfig, per_q: list[float], split: str = "dev", p95: float = 10.0
) -> RunResult:
    mean = sum(per_q) / len(per_q)
    metrics = {n: MetricCI(mean=mean, ci_lo=min(per_q), ci_hi=max(per_q)) for n in METRIC_NAMES}
    return RunResult(
        config_id=cfg.config_id,
        config=cfg,
        seed=42,
        split=split,
        corpus=CorpusInfo(name="mini", repo="r", tag="t", commit_sha="c", sha256_of_files="s"),
        goldset=GoldsetInfo(n_questions=len(per_q), all_verified=True, provisional=False),
        package_versions={},
        hardware_ref="results/hardware.json",
        min_overlap_frac=0.2,
        bootstrap_resamples=200,
        bootstrap_seed=42,
        metrics=metrics,
        latency_ms={"total": StageLatency(p50=p95 / 2, p95=p95)},
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
        ann_sanity_recall=1.0,
        per_query=[
            QueryResult(
                qid=f"q{i:02d}",
                question_type="factual",
                relevance=[],
                metrics=dict.fromkeys(METRIC_NAMES, v),
                hits=[],
            )
            for i, v in enumerate(per_q)
        ],
        created_at="2026-11-02T00:00:00+00:00",
    )


def test_describe_labels():
    assert describe(_cfg()) == "recursive_md-512 / bge-small-en-v1.5 / dense / no-rerank / pgvector"
    bm25 = _cfg(backend="control_bm25", retrieval={"mode": "sparse"})
    assert describe(bm25) == "recursive_md-512 / bge-small-en-v1.5 / bm25-control / no-rerank"


def test_single_factor_detection():
    a = _cfg()
    assert single_factor(a, _cfg(embedder={"name": "x"})) == "embedder"
    assert single_factor(a, _cfg(backend="qdrant")) == "backend"
    assert single_factor(a, _cfg(retrieval={"mode": "hybrid"})) == "retrieval"
    assert (
        single_factor(a, _cfg(backend="control_bm25", retrieval={"mode": "sparse"})) == "retrieval"
    )
    assert single_factor(a, _cfg(backend="qdrant", embedder={"name": "x"})) is None
    assert single_factor(a, a) is None


def test_ablations_paired_and_significance():
    better = [1.0] * 20
    worse = [0.0] * 10 + [1.0] * 10
    a = _run(_cfg(), worse)
    b = _run(_cfg(retrieval={"mode": "hybrid"}), better, p95=25.0)
    unrelated = _run(_cfg(backend="qdrant", embedder={"name": "x"}), better)
    (abl,) = ablations([a, b, unrelated], "ndcg@10", seed=42)
    assert abl.factor == "retrieval"
    assert abl.a.config_id == b.config_id  # better config first
    assert abs(abl.delta - 0.5) < 1e-12
    assert abl.significant and abl.lo > 0
    assert abl.latency_delta_ms == 15.0


def test_findings_flag_backend_difference_and_saturation(tmp_path: Path):
    pg = _run(_cfg(), [1.0] * 20)
    qd = _run(_cfg(backend="qdrant"), [0.0] * 10 + [1.0] * 10)
    test_best = _run(_cfg(), [1.0] * 30, split="test")
    lines = findings_lines(GRID, [pg, qd], [test_best], n_dev_runs=2)
    text = "\n".join(lines)
    assert "# Findings draft (generated)" in text
    assert "## Best finalist on test" in text and test_best.config_id in text
    assert "**backend**: 1/1" in text
    assert "hypothesis: pgvector beat qdrant" in text
    assert "hypothesis: recall@5 is saturated" in text


def test_findings_no_claim_when_ci_includes_zero():
    a = _run(_cfg(), [1.0, 0.0] * 10)
    b = _run(_cfg(embedder={"name": "x"}), [0.0, 1.0] * 10)
    text = "\n".join(findings_lines(GRID, [a, b], [], n_dev_runs=2))
    assert "**embedder**: 0/1" in text
    assert "no detectable difference" in text
    assert "- none" in text  # no flags


def test_leaderboard_ranks_by_selection():
    low = _run(_cfg(), [0.2] * 5, split="test")
    high = _run(_cfg(backend="qdrant"), [0.9] * 5, split="test")
    rows = leaderboard([low, high], GRID.selection)
    assert [r.config_id for r in rows] == [high.config_id, low.config_id]
    assert rows[0].rank == 1 and rows[0].metrics["ndcg@10"][0] == 0.9


def test_evaluate_finalists_reuses_existing_test_runs(tmp_path: Path):
    c1, c2 = _cfg(), _cfg(backend="qdrant")
    dev = {c1.config_id: _run(c1, [1.0]), c2.config_id: _run(c2, [1.0])}
    write_run(_run(c1, [1.0], split="test"), tmp_path)
    calls: list[str] = []

    def fake_eval(cfg: PipelineConfig, split: str) -> RunResult:
        calls.append(cfg.config_id)
        assert split == "test"
        result = _run(cfg, [0.5], split="test")
        write_run(result, tmp_path)
        return result

    runs = evaluate_finalists(
        GRID, [c1.config_id, c2.config_id], dev, tmp_path, evaluate_fn=fake_eval
    )
    assert calls == [c2.config_id]
    assert [r.split for r in runs] == ["test", "test"]
    assert run_path(tmp_path, c2.config_id, "test").exists()


def test_attach_generation_skips_runs_that_have_it(tmp_path: Path):
    run = _run(_cfg(), [1.0], split="test")
    seen: list[str] = []

    def fake_ragas(r: RunResult) -> RunResult:
        seen.append(r.config_id)
        return r

    attach_generation([run], tmp_path, ragas_fn=fake_ragas)
    assert seen == [run.config_id]
    assert (tmp_path / "summary.csv").exists()
