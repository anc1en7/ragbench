"""Dev selection -> test evaluation, RAGAS, leaderboard and generated findings (Task 4.4).

Leakage rule (spec section 7): finalists are chosen from DEV results only (the sweep's
ranking), and only those finalists are ever evaluated on TEST. Nothing here reads test
metrics to choose a config.

``write_findings`` produces ``results/findings_draft.md`` purely from run JSONs. Every
number is read from ``results/`` and formatted by code. Ablations compare dev runs that
differ in exactly one factor (chunker, embedder, retrieval, rerank, backend) with a paired
bootstrap on per-question selection-metric values. A difference is stated only when its
95% CI excludes 0. Surprising patterns are emitted as lines starting ``hypothesis:``; they
are hypotheses for the human to review, not findings.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ragbench.config import PipelineConfig
from ragbench.eval.results import (
    GENERATION_METRICS,
    RunResult,
    Split,
    load_run,
    run_path,
    upsert_summary,
    write_run,
)
from ragbench.eval.stats import paired_bootstrap_ci
from ragbench.sweep.stages import Grid, Selection, rank

EvaluateFn = Callable[[PipelineConfig, Split], RunResult]
RagasFn = Callable[[RunResult], RunResult]

FACTORS: tuple[str, ...] = ("chunker", "embedder", "retrieval", "rerank", "backend")


def describe(cfg: PipelineConfig) -> str:
    """Short human label for a config."""
    ch = cfg.chunker
    chunker = f"{ch.type}-{ch.size_tokens}" if ch.type != "semantic" else "semantic"
    rerank = "no-rerank" if cfg.rerank.type == "none" else f"cross-encoder@{cfg.rerank.top_n}"
    mode = "bm25-control" if cfg.backend == "control_bm25" else cfg.retrieval.mode
    backend = "" if cfg.backend == "control_bm25" else f" / {cfg.backend}"
    return f"{chunker} / {cfg.embedder.name.split('/')[-1]} / {mode} / {rerank}{backend}"


def evaluate_finalists(
    grid: Grid,
    finalists: Sequence[str],
    dev_results: dict[str, RunResult],
    results_dir: Path,
    *,
    evaluate_fn: EvaluateFn,
    log: Callable[[str], None] = lambda _m: None,
) -> list[RunResult]:
    """Test-split run for each finalist (reusing an existing test run JSON)."""
    out: list[RunResult] = []
    for config_id in finalists:
        path = run_path(results_dir, config_id, "test")
        if path.exists():
            log(f"[test] {config_id} reused")
            out.append(load_run(path))
            continue
        cfg = dev_results[config_id].config
        log(f"[test] {config_id} running ({describe(cfg)})")
        out.append(evaluate_fn(cfg, "test"))
    return out


def attach_generation(
    test_runs: Sequence[RunResult],
    results_dir: Path,
    *,
    ragas_fn: RagasFn,
    log: Callable[[str], None] = lambda _m: None,
) -> list[RunResult]:
    """Run RAGAS for test runs lacking a generation block; persist JSON + summary row."""
    updated: list[RunResult] = []
    for run in test_runs:
        if run.generation is not None:
            log(f"[ragas] {run.config_id} reused")
            updated.append(run)
            continue
        log(f"[ragas] {run.config_id} running")
        new = ragas_fn(run)
        write_run(new, results_dir)
        upsert_summary(new, results_dir)
        updated.append(new)
    return updated


# --- leaderboard -----------------------------------------------------------------------


@dataclass(frozen=True)
class LeaderRow:
    rank: int
    config_id: str
    label: str
    metrics: dict[str, tuple[float, float, float]]
    p95_total_ms: float
    generation: dict[str, float]


def leaderboard(runs: Sequence[RunResult], selection: Selection) -> list[LeaderRow]:
    rows = []
    for idx, run in enumerate(rank(runs, selection), start=1):
        gen = run.generation
        rows.append(
            LeaderRow(
                rank=idx,
                config_id=run.config_id,
                label=describe(run.config),
                metrics={
                    name: (m.mean, m.ci_lo, m.ci_hi)
                    for name, m in run.metrics.items()
                    if name in ("recall@5", "mrr@10", "ndcg@10")
                },
                p95_total_ms=run.latency_ms["total"].p95,
                generation={n: m.mean for n, m in gen.metrics.items()} if gen else {},
            )
        )
    return rows


# --- ablations + findings --------------------------------------------------------------


def _factor_values(cfg: PipelineConfig) -> dict[str, Any]:
    dump = cfg.model_dump(mode="json")
    return {
        "chunker": dump["chunker"],
        "embedder": dump["embedder"],
        # the BM25 control changes backend AND mode together: treat it as a retrieval change
        "retrieval": (dump["retrieval"], cfg.backend == "control_bm25"),
        "rerank": dump["rerank"],
        "backend": dump["backend"] if cfg.backend != "control_bm25" else "pgvector",
    }


def single_factor(a: PipelineConfig, b: PipelineConfig) -> str | None:
    """The one factor in which ``a`` and ``b`` differ, or ``None`` (0 or 2+ factors)."""
    if a.corpus != b.corpus or a.final_k != b.final_k or a.seed != b.seed:
        return None
    va, vb = _factor_values(a), _factor_values(b)
    diff = [f for f in FACTORS if va[f] != vb[f]]
    return diff[0] if len(diff) == 1 else None


@dataclass(frozen=True)
class Ablation:
    factor: str
    a: RunResult
    b: RunResult
    delta: float
    lo: float
    hi: float
    latency_delta_ms: float

    @property
    def significant(self) -> bool:
        return self.lo > 0 or self.hi < 0


def ablations(runs: Sequence[RunResult], metric: str, seed: int) -> list[Ablation]:
    """All single-factor pairs among ``runs`` (same split), ``a`` the better of each pair."""
    out: list[Ablation] = []
    ordered = sorted(runs, key=lambda r: r.config_id)
    for i, x in enumerate(ordered):
        for y in ordered[i + 1 :]:
            factor = single_factor(x.config, y.config)
            if factor is None or [q.qid for q in x.per_query] != [q.qid for q in y.per_query]:
                continue
            a, b = (x, y) if x.metrics[metric].mean >= y.metrics[metric].mean else (y, x)
            va = [q.metrics[metric] for q in a.per_query]
            vb = [q.metrics[metric] for q in b.per_query]
            lo, hi = paired_bootstrap_ci(va, vb, seed=seed)
            out.append(
                Ablation(
                    factor=factor,
                    a=a,
                    b=b,
                    delta=a.metrics[metric].mean - b.metrics[metric].mean,
                    lo=lo,
                    hi=hi,
                    latency_delta_ms=a.latency_ms["total"].p95 - b.latency_ms["total"].p95,
                )
            )
    return out


def _fmt_ci(mean: float, lo: float, hi: float) -> str:
    return f"{mean:.3f} (95% CI [{lo:.3f}, {hi:.3f}])"


def findings_lines(
    grid: Grid,
    dev_runs: Sequence[RunResult],
    test_runs: Sequence[RunResult],
    *,
    n_dev_runs: int,
    elapsed_minutes: float | None = None,
) -> list[str]:
    """Markdown lines of the generated findings draft (see module docstring)."""
    metric = grid.selection.metric
    lines = [
        "# Findings draft (generated)",
        "",
        "Generated by `ragbench finalize` from `results/runs/*.json`; no number below is "
        "hand-written. Lines starting `hypothesis:` are unverified explanations for review.",
        "",
        f"- Sweep: {n_dev_runs} configs evaluated on dev "
        f"({dev_runs[0].goldset.n_questions if dev_runs else 0} questions), "
        f"selection by {metric}; {len(test_runs)} finalists evaluated on test"
        + (f"; sweep wall time {elapsed_minutes:.0f} min." if elapsed_minutes else "."),
    ]
    hyps: list[str] = []
    if test_runs:
        best = rank(test_runs, grid.selection)[0]
        m = best.metrics
        lines += [
            "",
            "## Best finalist on test",
            "",
            f"- {describe(best.config)} (`{best.config_id}`, n={best.goldset.n_questions}): "
            f"nDCG@10 {_fmt_ci(m['ndcg@10'].mean, m['ndcg@10'].ci_lo, m['ndcg@10'].ci_hi)}, "
            f"MRR@10 {_fmt_ci(m['mrr@10'].mean, m['mrr@10'].ci_lo, m['mrr@10'].ci_hi)}, "
            f"Recall@5 {_fmt_ci(m['recall@5'].mean, m['recall@5'].ci_lo, m['recall@5'].ci_hi)}, "
            f"p95 total latency {best.latency_ms['total'].p95:.1f} ms.",
        ]
        if best.generation is not None:
            parts = [
                f"{n} {best.generation.metrics[n].mean:.3f} (n={best.generation.n_valid.get(n, 0)})"
                for n in GENERATION_METRICS
                if n in best.generation.metrics
            ]
            lines.append(
                f"- RAGAS on {best.generation.n_questions} test questions: " + ", ".join(parts)
            )
        for name in ("recall@5", "recall@10"):
            if m[name].mean >= 1.0:
                hyps.append(
                    f"hypothesis: {name} is saturated (1.000) for the best finalist, so it cannot "
                    "separate strong configs; section-derived questions may be easy at k>=5."
                )

    abl = ablations(dev_runs, metric, grid.seed)
    lines += ["", f"## Single-factor ablations on dev ({metric}, paired bootstrap)", ""]
    for factor in FACTORS:
        pairs = [x for x in abl if x.factor == factor]
        if not pairs:
            continue
        sig = [x for x in pairs if x.significant]
        top = max(pairs, key=lambda x: x.delta)
        verdict = "CI excludes 0" if top.significant else "CI includes 0: no detectable difference"
        lines.append(
            f"- **{factor}**: {len(sig)}/{len(pairs)} single-factor pairs differ (CI excludes 0). "
            f"Largest: {describe(top.a.config)} over {describe(top.b.config)}: "
            f"Δ{metric} {top.delta:+.3f} [{top.lo:+.3f}, {top.hi:+.3f}] ({verdict}); "
            f"Δp95 latency {top.latency_delta_ms:+.1f} ms."
        )
        for x in sig:
            ca, cb = x.a.config, x.b.config
            if factor == "backend":
                hyps.append(
                    f"hypothesis: {ca.backend} beat {cb.backend} on {metric} by {x.delta:+.3f} "
                    f"[{x.lo:+.3f}, {x.hi:+.3f}] for {describe(ca)}; dense quality should "
                    "match across backends, so check sparse-scorer differences (FTS vs BM25) "
                    "or ANN recall."
                )
            if (
                factor == "retrieval"
                and ca.retrieval.mode == "sparse"
                and cb.retrieval.mode == "dense"
            ):
                hyps.append(
                    f"hypothesis: sparse beat dense ({describe(ca)}) by {x.delta:+.3f}; "
                    "questions may share more vocabulary with their sections than the "
                    "lexical-overlap filter implies."
                )
            if factor == "rerank" and cb.rerank.type == "cross_encoder":
                hyps.append(
                    f"hypothesis: the cross-encoder LOWERED {metric} by {x.delta:.3f} for "
                    f"{describe(cb)}; reranker truncation (512 tokens) may cut long chunks."
                )

    for run in [*dev_runs, *test_runs]:
        if run.config.retrieval.mode == "sparse" and run.per_query:
            covered = sum(1 for q in run.per_query if q.hits)
            if covered < 0.5 * len(run.per_query):
                hyps.append(
                    f"hypothesis: {run.config.backend} sparse retrieval returned hits for only "
                    f"{covered}/{len(run.per_query)} questions (`{run.config_id}`, {run.split}); "
                    "its lexical query likely requires every term to match (AND semantics), so "
                    f"{run.config.backend} hybrid degenerates to dense. Not a verdict on lexical "
                    "retrieval: compare the control_bm25 rows."
                )
        if run.ann_sanity_recall is not None and run.ann_sanity_recall < 0.98:
            hyps.append(
                f"hypothesis: ANN sanity recall {run.ann_sanity_recall:.3f} < 0.98 for "
                f"`{run.config_id}` ({run.split}); HNSW ef_search may be too low for this index."
            )
        gen = run.generation
        if gen is not None:
            for name, valid in gen.n_valid.items():
                if valid < 0.9 * gen.n_questions:
                    hyps.append(
                        f"hypothesis: judge produced only {valid}/{gen.n_questions} valid "
                        f"{name} scores for `{run.config_id}`; provider errors or "
                        "unparsable output."
                    )
    lines += ["", "## Flags for review", ""]
    lines += [f"- {h}" for h in dict.fromkeys(hyps)] or ["- none"]
    return lines


def write_findings(path: Path, lines: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
