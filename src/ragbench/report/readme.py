"""Generated README sections (Task 6.1).

``make report`` (``ragbench report``) rebuilds the parts of ``README.md`` between marker
comments from ``results/runs/*.json`` so that no headline number is typed by hand:

- ``<!-- BEGIN:RESULTS TABLE -->``: test leaderboard of the dev-selected finalists
- ``<!-- BEGIN:GENERATION TABLE -->``: RAGAS metrics of the finalists
- ``<!-- BEGIN:FINDINGS -->``: 4-6 findings bullets, each with a number and a 95% CI

Each block ends with ``<!-- END:<NAME> -->``. When a stage has not run, its block holds the
literal text ``not yet run``, never a guess.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from ragbench.eval.results import GENERATION_METRICS, RunResult, load_run
from ragbench.sweep.select import ablations, describe

NOT_YET_RUN = "_not yet run_"
REFUSAL = "does not contain the answer"
SELECTION_METRIC = "ndcg@10"


def load_runs(results_dir: Path, split: str, corpus: str = "fastapi_docs") -> list[RunResult]:
    runs = [load_run(p) for p in sorted((results_dir / "runs").glob(f"*__{split}.json"))]
    return [r for r in runs if r.config.corpus == corpus]


def _ci(run: RunResult, name: str) -> str:
    m = run.metrics[name]
    return f"{m.mean:.3f} [{m.ci_lo:.3f}, {m.ci_hi:.3f}]"


def _ranked(test_runs: Sequence[RunResult]) -> list[RunResult]:
    return sorted(
        test_runs,
        key=lambda r: (
            -r.metrics[SELECTION_METRIC].mean,
            -r.metrics["mrr@10"].mean,
            -r.metrics["recall@5"].mean,
            r.config_id,
        ),
    )


def results_table(test_runs: Sequence[RunResult]) -> str:
    if not test_runs:
        return NOT_YET_RUN
    lines = [
        "| # | Config | nDCG@10 (95% CI) | MRR@10 (95% CI) | Recall@5 (95% CI) | p95 latency |",
        "|---|---|---|---|---|---|",
    ]
    for i, run in enumerate(_ranked(test_runs), start=1):
        lines.append(
            f"| {i} | {describe(run.config)} (`{run.config_id}`) | {_ci(run, 'ndcg@10')} | "
            f"{_ci(run, 'mrr@10')} | {_ci(run, 'recall@5')} | "
            f"{run.latency_ms['total'].p95:.1f} ms |"
        )
    n = test_runs[0].goldset.n_questions
    lines += ["", f"Test split, n = {n} questions. Ranked by nDCG@10, then MRR@10 and Recall@5."]
    return "\n".join(lines)


def refusals(run: RunResult) -> int | None:
    if run.generation is None:
        return None
    return sum(REFUSAL in q.answer.lower() for q in run.generation.per_query)


def generation_table(test_runs: Sequence[RunResult]) -> str:
    runs = [r for r in _ranked(test_runs) if r.generation is not None]
    if not runs:
        return NOT_YET_RUN
    header = " | ".join(n.replace("_", " ").capitalize() for n in GENERATION_METRICS)
    lines = [
        f'| # | Config | {header} | "Context does not contain the answer" replies |',
        "|---|---|" + "---|" * len(GENERATION_METRICS) + "---|",
    ]
    for i, run in enumerate(runs, start=1):
        gen = run.generation
        assert gen is not None
        cells = []
        for name in GENERATION_METRICS:
            m = gen.metrics.get(name)
            cells.append(f"{m.mean:.3f} [{m.ci_lo:.2f}, {m.ci_hi:.2f}]" if m else "n/a")
        lines.append(
            f"| {i} | {describe(run.config)} | {' | '.join(cells)} | "
            f"{refusals(run)} / {gen.n_questions} |"
        )
    gen0 = runs[0].generation
    assert gen0 is not None
    lines += [
        "",
        f"`ragas=={gen0.ragas_version}`, {gen0.n_questions} seeded test questions per "
        f"finalist, top {gen0.context_k} chunks as context, generator `{gen0.generator_model}` "
        f"and judge `{gen0.judge_model}`, prompt `{gen0.prompt_version}`.",
    ]
    return "\n".join(lines)


def findings(dev_runs: Sequence[RunResult], test_runs: Sequence[RunResult]) -> str:
    if not dev_runs and not test_runs:
        return NOT_YET_RUN
    bullets: list[str] = []
    if test_runs:
        ranked = _ranked(test_runs)
        best = ranked[0]
        lo = max(r.metrics[SELECTION_METRIC].ci_lo for r in ranked)
        hi = min(r.metrics[SELECTION_METRIC].ci_hi for r in ranked)
        tied = lo <= hi
        bullets.append(
            f"**Best finalist on test:** {describe(best.config)}, nDCG@10 {_ci(best, 'ndcg@10')}"
            f", Recall@5 {_ci(best, 'recall@5')}. "
            + (
                f"All {len(ranked)} finalists' nDCG@10 CIs share the interval "
                f"[{lo:.3f}, {hi:.3f}], so none is detectably better."
                if tied
                else "Not every finalist's CI overlaps the others'."
            )
        )
    abl = ablations(dev_runs, SELECTION_METRIC, seed=42)
    for factor, text in (
        ("rerank", "**Reranking**"),
        ("embedder", "**Embedder**"),
        ("chunker", "**Chunker**"),
        ("backend", "**Backend**"),
    ):
        pairs = [a for a in abl if a.factor == factor]
        if not pairs:
            continue
        sig = [a for a in pairs if a.significant]
        top = max(pairs, key=lambda a: a.delta)
        bullets.append(
            f"{text}: {len(sig)}/{len(pairs)} single-factor pairs on dev differ (95% paired "
            f"bootstrap CI excludes 0). Largest: {describe(top.a.config)} over "
            f"{describe(top.b.config)}, ΔnDCG@10 {top.delta:+.3f} "
            f"[{top.lo:+.3f}, {top.hi:+.3f}], Δp95 latency {top.latency_delta_ms:+.1f} ms."
        )
    gen_runs = [r for r in test_runs if r.generation is not None]
    if gen_runs:
        counts = [(refusals(r) or 0) / r.generation.n_questions for r in gen_runs if r.generation]
        recall = [
            r.generation.metrics["context_recall"].mean
            for r in gen_runs
            if r.generation and "context_recall" in r.generation.metrics
        ]
        bullets.append(
            f"**Generation, not retrieval, limits answers:** RAGAS context recall is "
            f"{min(recall):.3f}-{max(recall):.3f}, yet the generator replied that the context "
            f"does not contain the answer in {min(counts):.0%}-{max(counts):.0%} of answers."
        )
    return "\n".join(f"- {b}" for b in bullets[:6])


def inject(text: str, name: str, content: str) -> str:
    """Replace the body between ``<!-- BEGIN:name -->`` and ``<!-- END:name -->``."""
    pattern = re.compile(
        rf"(<!-- BEGIN:{re.escape(name)} -->).*?(<!-- END:{re.escape(name)} -->)", re.DOTALL
    )
    if not pattern.search(text):
        raise ValueError(f"README markers for {name!r} not found")
    return pattern.sub(lambda m: f"{m.group(1)}\n{content}\n{m.group(2)}", text, count=1)


def render(readme: str, dev_runs: Sequence[RunResult], test_runs: Sequence[RunResult]) -> str:
    out = inject(readme, "RESULTS TABLE", results_table(test_runs))
    out = inject(out, "GENERATION TABLE", generation_table(test_runs))
    return inject(out, "FINDINGS", findings(dev_runs, test_runs))
