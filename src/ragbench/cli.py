"""RAGBench command-line interface."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer

from ragbench.hardware import collect_hardware

app = typer.Typer(
    name="ragbench",
    help="RAGBench: benchmark Retrieval-Augmented Generation pipelines.",
    no_args_is_help=True,
)
corpus_app = typer.Typer(name="corpus", help="Corpus fetch and build.", no_args_is_help=True)
app.add_typer(corpus_app, name="corpus")
goldset_app = typer.Typer(
    name="goldset", help="Gold-set generation and review.", no_args_is_help=True
)
app.add_typer(goldset_app, name="goldset")

HARDWARE_PATH = Path("results/hardware.json")


@app.callback()
def root() -> None:
    """RAGBench command-line interface."""


@app.command("hardware")
def hardware_cmd(
    out: Path = typer.Option(HARDWARE_PATH, "--out", help="Where to write hardware.json"),
) -> None:
    """Detect hardware and package versions, writing results/hardware.json."""
    data = collect_hardware(out)
    typer.echo(json.dumps(data, indent=2))


@corpus_app.command("ingest")
def corpus_ingest_cmd(
    raw: Path = typer.Option(Path("data/raw"), "--raw", help="Raw corpus directory"),
    out: Path = typer.Option(Path("data/documents.jsonl"), "--out", help="Documents JSONL"),
    lock: Path = typer.Option(Path("corpus.lock"), "--lock", help="Corpus lock file"),
    force: bool = typer.Option(False, "--force", help="Re-resolve pin and re-download"),
) -> None:
    """Fetch the pinned corpus (no-op when fresh) and build documents.jsonl."""
    from ragbench.corpus.build import build_documents
    from ragbench.corpus.fetch import fetch

    lock_obj = fetch(lock_path=lock, raw_dir=raw, force=force)
    if lock_obj is None:
        typer.echo(f"corpus up to date (lock: {lock})")
    else:
        typer.echo(
            f"fetched tag {lock_obj.tag} @ {lock_obj.commit_sha[:12]} ({lock_obj.file_count} files)"
        )
    documents = build_documents(raw_dir=raw, out=out)
    sections = sum(len(doc.sections) for doc in documents)
    approx_tokens = sum(len(doc.text.split()) for doc in documents)
    typer.echo(
        f"documents={len(documents)} sections={sections} approx_tokens={approx_tokens} -> {out}"
    )


@goldset_app.command("review")
def goldset_review_cmd(
    candidates: Path = typer.Option(
        Path("data/goldset/candidates.jsonl"), "--candidates", help="Candidates JSONL"
    ),
    verified: Path = typer.Option(
        Path("data/goldset/verified.jsonl"), "--verified", help="Verified JSONL (output)"
    ),
    documents: Path = typer.Option(
        Path("data/documents.jsonl"), "--documents", help="Corpus for span excerpts"
    ),
    log: Path = typer.Option(
        Path("data/goldset/review_log.jsonl"), "--log", help="Reject/flag log (output)"
    ),
) -> None:
    """Review gold-set candidates interactively (accept/edit/reject/flag)."""
    from rich.console import Console

    from ragbench.goldset.review import review

    try:
        summary = review(
            candidates_path=candidates,
            verified_path=verified,
            documents_path=documents,
            log_path=log,
            console=Console(),
        )
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg="red", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"accepted={summary['accepted']} rejected={summary['rejected']} "
        f"flagged={summary['flagged']} skipped={summary['skipped']} "
        f"remaining={summary['remaining']}"
    )


@goldset_app.command("split")
def goldset_split_cmd(
    verified: Path = typer.Option(
        Path("data/goldset/verified.jsonl"), "--verified", help="Verified JSONL"
    ),
    out: Path = typer.Option(
        Path("data/goldset/splits.json"), "--out", help="Splits JSON (output)"
    ),
    dev_frac: float = typer.Option(0.3, "--dev-frac", help="Dev fraction"),
    seed: int = typer.Option(42, "--seed", help="Split seed"),
) -> None:
    """Split verified questions into stratified dev/test sets."""
    from ragbench.goldset.split import split_verified

    try:
        result = split_verified(verified_path=verified, out_path=out, dev_frac=dev_frac, seed=seed)
    except (ValueError, FileNotFoundError) as exc:
        typer.secho(str(exc), fg="red", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"seed={result['seed']} dev={len(result['dev'])} test={len(result['test'])} -> {out}"
    )


@goldset_app.command("generate")
def goldset_generate_cmd(
    documents: Path = typer.Option(
        Path("data/documents.jsonl"), "--documents", help="Corpus JSONL"
    ),
    candidates: Path = typer.Option(
        Path("data/goldset/candidates.jsonl"), "--candidates", help="Candidates JSONL (output)"
    ),
    stats: Path = typer.Option(
        Path("data/goldset/generate_stats.json"), "--stats", help="Filter stats (output)"
    ),
    target: int = typer.Option(800, "--target", help="Sections to sample"),
    concurrency: int = typer.Option(3, "--concurrency", help="LLM workers"),
    seed: int = typer.Option(42, "--seed", help="Sampling seed"),
    pacer: float = typer.Option(
        0.5, "--pacer", help="Min seconds between request starts (rate-limit safety)"
    ),
) -> None:
    """Generate filtered gold-set candidates via the configured LLM."""
    from ragbench.goldset.generate import fastembed_embedder, generate_candidates
    from ragbench.llm_client import OpenAICompatClient, settings_from_env

    try:
        settings = settings_from_env()
    except Exception as exc:  # noqa: BLE001
        typer.secho(str(exc), fg="red", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo("embedding model for dedup loading (first run downloads ~90MB)...")
    embedder = fastembed_embedder()
    with OpenAICompatClient(settings) as client:
        result = generate_candidates(
            documents_path=documents,
            out_path=candidates,
            stats_path=stats,
            client=client,
            embedder=embedder,
            sample_target=target,
            concurrency=concurrency,
            seed=seed,
            pacer_interval=pacer,
        )
    drops = result["drops"]
    typer.echo(
        f"sampled={result['sampled']} survivors={result['survivors']} "
        f"drops={ {k: v for k, v in drops.items() if v} } -> {candidates}"
    )


@app.command("llm-probe")
def llm_probe_cmd(
    out: Path = typer.Option(HARDWARE_PATH, "--out", help="hardware.json to merge results into"),
) -> None:
    """Probe the configured endpoint, verify a chat completion, pin it in hardware.json."""
    from ragbench.llm_client import LLMConfigError, probe_endpoint, settings_from_env

    try:
        settings = settings_from_env()
    except LLMConfigError as exc:
        typer.secho(str(exc), fg="red", err=True)
        raise typer.Exit(code=1) from exc

    result = probe_endpoint(settings.base_url, settings.api_key)
    if not result.get("chat_ok"):
        typer.secho(str(result.get("error", "probe failed")), fg="red", err=True)
        raise typer.Exit(code=1)

    models = list(result.get("models", []))
    pinned = settings.model
    if pinned and pinned not in models:
        typer.secho(
            f"warning: LLM_MODEL={pinned!r} not in endpoint model list "
            f"(first 10: {models[:10]}). Proceeding anyway.",
            fg="yellow",
        )

    if out.exists():
        hardware = json.loads(out.read_text(encoding="utf-8"))
    else:
        hardware = collect_hardware(out)
    hardware["llm"] = {
        "base_url_host": settings.host,
        "model": pinned,
        "resolved_at": datetime.now(UTC).isoformat(),
        "chat_ok": True,
        "model_count": len(models),
    }
    out.write_text(json.dumps(hardware, indent=2) + "\n", encoding="utf-8")
    typer.echo(
        f"OK: {settings.host} reachable, {len(models)} models, "
        f"pinned model={pinned!r} recorded in {out}"
    )


@app.command("query")
def query_cmd(
    question: str = typer.Argument(..., help="Question to retrieve ranked chunks for"),
    config: Path = typer.Option(..., "--config", "-c", help="Pipeline YAML config"),
) -> None:
    """Retrieve ranked chunks for a question and print hits with stage timings."""
    from rich.console import Console
    from rich.table import Table

    from ragbench.config import load_pipeline_config
    from ragbench.pipeline import build_pipeline

    try:
        cfg = load_pipeline_config(config)
        pipeline = build_pipeline(cfg)
        hits, timings = pipeline.retrieve(question)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        typer.secho(str(exc), fg="red", err=True)
        raise typer.Exit(code=1) from exc

    console = Console()
    table = Table()
    table.add_column("#", justify="right")
    table.add_column("score", justify="right")
    table.add_column("source")
    table.add_column("doc_id")
    table.add_column("chunk_id")
    table.add_column("text")
    for row, hit in enumerate(hits, start=1):
        text = " ".join(hit.chunk.text.split())
        table.add_row(
            str(row),
            f"{hit.score:.6f}",
            hit.source,
            hit.chunk.doc_id,
            hit.chunk.chunk_id[:8],
            text[:80] + ("..." if len(text) > 80 else ""),
        )
    console.print(table)
    if not hits:
        typer.echo(f"no chunks retrieved — run `ragbench index --config {config}` first")

    timings_table = Table(title="timings")
    timings_table.add_column("stage")
    timings_table.add_column("ms", justify="right")
    for stage in ("embed_query", "dense", "sparse", "fusion", "rerank", "total"):
        timings_table.add_row(stage, f"{timings[stage]:.3f}")
    console.print(timings_table)


@app.command("index")
def index_cmd(
    config: Path = typer.Option(..., "--config", "-c", help="Pipeline YAML config"),
    documents: Path = typer.Option(
        Path("data/documents.jsonl"), "--documents", help="Corpus JSONL to index"
    ),
    limit: int | None = typer.Option(
        None, "--limit", help="Index only the first N documents (quick runs)"
    ),
) -> None:
    """Chunk, embed and store the corpus for a pipeline config."""
    from ragbench.config import load_pipeline_config
    from ragbench.goldset.generate import load_documents
    from ragbench.pipeline import build_pipeline, index_documents

    try:
        cfg = load_pipeline_config(config)
        docs = load_documents(documents)
        if limit is not None:
            docs = docs[:limit]
        stats = index_documents(docs, build_pipeline(cfg))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        typer.secho(str(exc), fg="red", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"config_id={cfg.config_id} backend={cfg.backend} docs={stats['docs']} "
        f"chunks={stats['chunks']} embed_seconds={stats['embed_seconds']:.3f} "
        f"index_seconds={stats['index_seconds']:.3f}"
    )


@app.command("eval")
def eval_cmd(
    config: Path = typer.Option(..., "--config", "-c", help="Pipeline YAML config"),
    split: str = typer.Option("dev", "--split", help="Gold-set split: dev | test"),
    passes: int = typer.Option(3, "--passes", help="Latency passes (pass 0 is warm-up, >= 3)"),
    min_overlap_frac: float = typer.Option(
        0.2, "--min-overlap-frac", help="Span-overlap hit threshold (fraction of chunk or gold)"
    ),
    resamples: int = typer.Option(1000, "--resamples", help="Bootstrap resamples"),
    ann_sample: int = typer.Option(50, "--ann-sample", help="Questions for the ANN check"),
    results_dir: Path = typer.Option(Path("results"), "--results-dir", help="Results root"),
    update_baseline: bool = typer.Option(
        False, "--update-baseline", help="Also write this run's Recall@5/MRR@10 as CI baseline"
    ),
    baseline: Path = typer.Option(
        Path("eval/baseline.json"), "--baseline", help="Baseline file for --update-baseline"
    ),
) -> None:
    """Index, evaluate and record one config on a gold-set split."""
    from rich.console import Console
    from rich.table import Table

    from ragbench.config import load_pipeline_config
    from ragbench.eval.retrieval_metrics import METRIC_NAMES
    from ragbench.eval.runner import evaluate, write_baseline
    from ragbench.pipeline import TIMING_KEYS

    if split not in ("dev", "test"):
        typer.secho(f"--split must be 'dev' or 'test', got {split!r}", fg="red", err=True)
        raise typer.Exit(code=1)
    try:
        cfg = load_pipeline_config(config)
        result, path = evaluate(
            cfg,
            "dev" if split == "dev" else "test",
            results_dir=results_dir,
            passes=passes,
            min_overlap_frac=min_overlap_frac,
            n_resamples=resamples,
            ann_sample=ann_sample,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        typer.secho(str(exc), fg="red", err=True)
        raise typer.Exit(code=1) from exc

    console = Console()
    flag = " [PROVISIONAL: unverified gold set]" if result.goldset.provisional else ""
    console.print(
        f"config_id={result.config_id} corpus={cfg.corpus} backend={cfg.backend} "
        f"mode={cfg.retrieval.mode} split={result.split} "
        f"questions={result.goldset.n_questions}{flag}"
    )
    metrics_table = Table(title=f"retrieval metrics (95% bootstrap CI, n={resamples})")
    metrics_table.add_column("metric")
    metrics_table.add_column("mean", justify="right")
    metrics_table.add_column("95% CI", justify="right")
    for name in METRIC_NAMES:
        m = result.metrics[name]
        metrics_table.add_row(name, f"{m.mean:.3f}", f"[{m.ci_lo:.3f}, {m.ci_hi:.3f}]")
    console.print(metrics_table)

    latency_table = Table(title=f"latency ms ({passes - 1} measured passes, warm-up dropped)")
    latency_table.add_column("stage")
    latency_table.add_column("p50", justify="right")
    latency_table.add_column("p95", justify="right")
    for stage in TIMING_KEYS:
        lat = result.latency_ms[stage]
        latency_table.add_row(stage, f"{lat.p50:.2f}", f"{lat.p95:.2f}")
    console.print(latency_table)

    idx = result.index
    throughput = f"{idx.chunks_per_sec:.1f}" if idx.chunks_per_sec is not None else "n/a"
    ann = f"{result.ann_sanity_recall:.4f}" if result.ann_sanity_recall is not None else "n/a"
    console.print(
        f"index: docs={idx.docs} chunks={idx.chunks} wall={idx.wall_seconds:.2f}s "
        f"embed={idx.embed_seconds:.2f}s ({throughput} chunks/s) "
        f"size={idx.size_bytes / 2**20:.2f} MiB peak_rss={idx.peak_rss_bytes / 2**20:.0f} MiB"
    )
    console.print(f"ann_sanity_recall@{cfg.final_k}={ann}")
    console.print(f"wrote {path} and {results_dir / 'summary.csv'}")
    if update_baseline:
        console.print(f"baseline updated: {write_baseline(result, baseline)}")


@app.command("sweep")
def sweep_cmd(
    grid: Path = typer.Option(Path("configs/grid.yaml"), "--grid", help="Sweep grid YAML"),
    results_dir: Path = typer.Option(Path("results"), "--results-dir", help="Results root"),
    estimate_only: bool = typer.Option(
        False, "--estimate-only", help="Print the planned run counts and exit"
    ),
) -> None:
    """Run the staged sweep on the dev split (resumes; reuses completed runs)."""
    from ragbench.eval.results import run_path
    from ragbench.sweep.runner import run_sweep
    from ragbench.sweep.stages import estimate, expand_stage, load_grid

    try:
        grid_obj = load_grid(grid)
    except (FileNotFoundError, ValueError) as exc:
        typer.secho(str(exc), fg="red", err=True)
        raise typer.Exit(code=1) from exc

    plan = estimate(grid_obj)
    stage_a = expand_stage(grid_obj, grid_obj.stages[0], None)
    done_a = sum(run_path(results_dir, c.config_id, grid_obj.split).exists() for c in stage_a)
    typer.echo(f"sweep plan ({grid_obj.split} split, select by {grid_obj.selection.metric}):")
    for name, count in plan:
        typer.echo(f"  stage {name}: {count} configs (before reuse of completed runs)")
    typer.echo(f"  total planned: {sum(c for _, c in plan)}; stage A already complete: {done_a}")
    typer.echo(f"  then: top {grid_obj.finalists} finalists -> test split (`ragbench finalize`)")
    if estimate_only:
        return

    try:
        report = run_sweep(grid_obj, results_dir, log=typer.echo)
    except KeyboardInterrupt:
        typer.secho(
            f"\ninterrupted — state saved; rerun `ragbench sweep --grid {grid}` to resume",
            fg="yellow",
            err=True,
        )
        raise typer.Exit(code=130) from None
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        typer.secho(str(exc), fg="red", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        f"sweep complete: ran={report.total_ran} reused={report.total_reused} "
        f"elapsed={report.elapsed_seconds / 60:.1f} min"
    )
    typer.echo(f"finalists (dev, by {grid_obj.selection.metric}): {report.finalists}")


def _print_leaderboard(rows: list[Any], title: str) -> None:
    from rich.console import Console
    from rich.table import Table

    table = Table(title=title)
    for col in ("#", "config", "nDCG@10", "MRR@10", "Recall@5", "p95 ms", "faith.", "ans.rel."):
        table.add_column(col, justify="left" if col == "config" else "right")
    for row in rows:

        def ci(name: str, r: Any = row) -> str:
            mean, lo, hi = r.metrics[name]
            return f"{mean:.3f} [{lo:.2f},{hi:.2f}]"

        gen = row.generation
        table.add_row(
            str(row.rank),
            f"{row.label}\n{row.config_id}",
            ci("ndcg@10"),
            ci("mrr@10"),
            ci("recall@5"),
            f"{row.p95_total_ms:.1f}",
            f"{gen['faithfulness']:.3f}" if "faithfulness" in gen else "-",
            f"{gen['answer_relevancy']:.3f}" if "answer_relevancy" in gen else "-",
        )
    Console().print(table)


@app.command("finalize")
def finalize_cmd(
    grid: Path = typer.Option(Path("configs/grid.yaml"), "--grid", help="Sweep grid YAML"),
    results_dir: Path = typer.Option(Path("results"), "--results-dir", help="Results root"),
    skip_ragas: bool = typer.Option(False, "--skip-ragas", help="Skip RAGAS generation metrics"),
    concurrency: int = typer.Option(3, "--concurrency", help="Concurrent LLM requests"),
    spotcheck: Path = typer.Option(
        Path("docs/judge_spotcheck.md"), "--spotcheck", help="Judge spot-check file (output)"
    ),
) -> None:
    """Dev-selected finalists -> test eval -> RAGAS -> leaderboard + findings draft."""
    from ragbench.embedding.st_embedder import SentenceTransformerEmbedder
    from ragbench.eval.ragas_runner import (
        RAGAS_EMBEDDER,
        build_ragas_scorer,
        judge_model_name,
        run_generation_eval,
        select_subset,
        write_spotcheck,
    )
    from ragbench.eval.runner import SPLITS_PATH, evaluate, load_split_questions, resolve_dataset
    from ragbench.goldset.generate import load_documents
    from ragbench.llm_client import OpenAICompatClient, settings_from_env
    from ragbench.pipeline import build_pipeline, index_documents
    from ragbench.sweep.runner import run_sweep
    from ragbench.sweep.select import (
        attach_generation,
        evaluate_finalists,
        findings_lines,
        leaderboard,
        write_findings,
    )
    from ragbench.sweep.stages import load_grid

    try:
        grid_obj = load_grid(grid)
        sweep = run_sweep(grid_obj, results_dir, log=typer.echo)  # reuses every dev run
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        typer.secho(str(exc), fg="red", err=True)
        raise typer.Exit(code=1) from exc

    from ragbench.eval.results import load_run, run_path

    dev_ids = list(dict.fromkeys(cid for st in sweep.stages for cid in st.config_ids))
    dev_runs = {cid: load_run(run_path(results_dir, cid, grid_obj.split)) for cid in dev_ids}
    typer.echo(f"finalists (dev, {grid_obj.selection.metric}): {sweep.finalists}")

    def evaluate_fn(cfg: Any, split: Any) -> Any:
        return evaluate(cfg, split, results_dir=results_dir)[0]

    test_runs = evaluate_finalists(
        grid_obj, sweep.finalists, dev_runs, results_dir, evaluate_fn=evaluate_fn, log=typer.echo
    )

    if not skip_ragas:
        settings = settings_from_env()
        ragas_embedder = SentenceTransformerEmbedder(RAGAS_EMBEDDER)
        scorer = build_ragas_scorer(settings, ragas_embedder, seed=grid_obj.seed)  # type: ignore[arg-type]
        dataset = resolve_dataset(grid_obj.corpus)
        documents = load_documents(dataset.documents)
        questions = load_split_questions(dataset.questions, SPLITS_PATH, "test")
        subset = select_subset(questions, seed=grid_obj.seed)

        def ragas_fn(run: Any) -> Any:
            pipe = build_pipeline(run.config)
            index_documents(documents, pipe)  # control_bm25 indexes live in-process only
            with OpenAICompatClient(settings) as client:
                gen, _samples = run_generation_eval(
                    pipe,
                    subset,
                    generator=client,
                    scorer=scorer,
                    generator_model=settings.model,
                    judge_model=judge_model_name(settings),
                    subset_seed=grid_obj.seed,
                    concurrency=concurrency,
                    log=typer.echo,
                )
            return run.model_copy(update={"generation": gen})

        test_runs = attach_generation(test_runs, results_dir, ragas_fn=ragas_fn, log=typer.echo)

        # spot-check: rebuild context texts deterministically from chunk ids
        qmap = {q.qid: q for q in questions}
        items = []
        for run in test_runs:
            if run.generation is None:
                continue
            chunker = build_pipeline(run.config).chunker
            texts = {c.chunk_id: c.text for doc in documents for c in chunker.chunk(doc)}
            for gq in run.generation.per_query:
                ctx = [texts.get(cid, "") for cid in gq.context_chunk_ids]
                items.append((run.config_id, gq, qmap[gq.qid], ctx))
        write_spotcheck(spotcheck, items, judge_model=judge_model_name(settings))
        typer.echo(f"wrote {spotcheck} (20 items for human review)")

    _print_leaderboard(
        leaderboard(test_runs, grid_obj.selection),
        f"test leaderboard (finalists selected on dev by {grid_obj.selection.metric})",
    )
    lines = findings_lines(
        grid_obj,
        list(dev_runs.values()),
        test_runs,
        n_dev_runs=len(dev_runs),
        elapsed_minutes=None,
    )
    path = write_findings(results_dir / "findings_draft.md", lines)
    typer.echo(f"wrote {path}")


@app.command("leaderboard")
def leaderboard_cmd(
    grid: Path = typer.Option(Path("configs/grid.yaml"), "--grid", help="Sweep grid YAML"),
    results_dir: Path = typer.Option(Path("results"), "--results-dir", help="Results root"),
    split: str = typer.Option("test", "--split", help="dev | test"),
    top: int = typer.Option(10, "--top", help="Rows to show"),
) -> None:
    """Print a leaderboard of stored runs for a split (ranked by the grid's selection)."""
    from ragbench.eval.results import load_run
    from ragbench.sweep.select import leaderboard
    from ragbench.sweep.stages import load_grid

    grid_obj = load_grid(grid)
    runs = [load_run(p) for p in sorted((results_dir / "runs").glob(f"*__{split}.json"))]
    runs = [r for r in runs if r.config.corpus == grid_obj.corpus]
    if not runs:
        typer.echo(f"no {split} runs in {results_dir / 'runs'}")
        raise typer.Exit(code=1)
    _print_leaderboard(leaderboard(runs, grid_obj.selection)[:top], f"{split} leaderboard")


@app.command("bench-embed")
def bench_embed_cmd(
    embedder: list[str] = typer.Option(
        [
            "BAAI/bge-small-en-v1.5",
            "sentence-transformers/all-MiniLM-L6-v2",
            "intfloat/e5-small-v2",
        ],
        "--embedder",
        help="Embedder(s) to benchmark",
    ),
    documents: Path = typer.Option(
        Path("data/documents.jsonl"), "--documents", help="Corpus JSONL"
    ),
    out: Path = typer.Option(
        Path("results/embedder_throughput.json"), "--out", help="Throughput JSON (merged)"
    ),
) -> None:
    """Cold (cache-bypassing) embedding throughput on recursive_md-512 chunks of the corpus."""
    from ragbench.chunking import get_chunker
    from ragbench.config import ChunkerCfg
    from ragbench.embedding.base import document_prefix
    from ragbench.embedding.st_embedder import SentenceTransformerEmbedder
    from ragbench.goldset.generate import load_documents
    from ragbench.sweep.throughput import measure_throughput, write_throughput

    chunker = get_chunker(ChunkerCfg(type="recursive_md", size_tokens=512, overlap_tokens=64))
    texts = [c.text for doc in load_documents(documents) for c in chunker.chunk(doc)]
    entries = {}
    for name in embedder:
        emb = SentenceTransformerEmbedder(name)
        model = emb._ensure_loaded()
        prefix = document_prefix(name)

        def encode(batch: list[str], m: Any = model, p: str = prefix) -> Any:
            return m.encode([p + t for t in batch], normalize_embeddings=True)

        stats = measure_throughput(encode, texts)
        entries[name] = {**stats, "model_revision": emb.model_revision, "device": "cpu"}
        typer.echo(f"{name}: {stats['chunks_per_sec']:.1f} chunks/s ({len(texts)} chunks)")
    write_throughput(out, entries, "recursive_md size 512 overlap 64 over data/documents.jsonl")
    typer.echo(f"wrote {out}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
