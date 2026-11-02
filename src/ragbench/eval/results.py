"""Schema-validated run results and the ``summary.csv`` writer (Task 3.4).

One evaluation of one config on one split is a :class:`RunResult`, written to
``results/runs/{config_id}__{split}.json``. The split is part of the file name (the plan
wrote ``{config_id}.json``) so a config's test-split run never overwrites its dev run.
Every run records what is needed to reproduce it: the full config and its id, seed,
corpus pin (from ``corpus.lock``), gold-set provenance, package versions, a hardware
reference, the overlap threshold, bootstrap settings, metrics with CIs, per-stage
latency, index stats, the ANN sanity recall, and per-question ranked hits (offsets, no
text: chunk text is recoverable from ``documents.jsonl``) for dashboard replay.

``results/summary.csv`` holds one flattened row per ``(config_id, split)``: re-running a
config replaces its row in place; the header is written once.
"""

from __future__ import annotations

import csv
import io
import json
from importlib import metadata
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from ragbench.config import PipelineConfig
from ragbench.eval.retrieval_metrics import METRIC_NAMES

SCHEMA_VERSION: int = 1
Split = Literal["dev", "test"]

KEY_PACKAGES: tuple[str, ...] = (
    "ragbench",
    "numpy",
    "torch",
    "sentence-transformers",
    "transformers",
    "qdrant-client",
    "pgvector",
    "psycopg",
    "fastembed",
    "bm25s",
    "tiktoken",
    "pydantic",
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MetricCI(_Strict):
    mean: float
    ci_lo: float
    ci_hi: float

    @model_validator(mode="after")
    def _ci_brackets_mean(self) -> MetricCI:
        # small tolerance: a percentile bound can differ from the mean by float rounding
        tol = 1e-12
        if not self.ci_lo - tol <= self.mean <= self.ci_hi + tol:
            raise ValueError(
                f"expected ci_lo <= mean <= ci_hi, got {self.ci_lo} / {self.mean} / {self.ci_hi}"
            )
        return self


class StageLatency(_Strict):
    p50: float
    p95: float


class IndexStats(_Strict):
    docs: int
    chunks: int
    embed_seconds: float
    index_seconds: float
    wall_seconds: float
    peak_rss_bytes: int
    chunks_per_sec: float | None
    size_bytes: int


class RetrievedChunk(_Strict):
    chunk_id: str
    doc_id: str
    start: int
    end: int
    score: float
    source: str


class QueryResult(_Strict):
    qid: str
    question_type: str
    relevance: list[bool]
    metrics: dict[str, float]
    hits: list[RetrievedChunk]


class CorpusInfo(_Strict):
    name: str
    repo: str
    tag: str
    commit_sha: str
    sha256_of_files: str


class GoldsetInfo(_Strict):
    n_questions: int
    all_verified: bool
    provisional: bool


class RunResult(_Strict):
    schema_version: int = SCHEMA_VERSION
    config_id: str
    config: PipelineConfig
    seed: int
    split: Split
    corpus: CorpusInfo
    goldset: GoldsetInfo
    package_versions: dict[str, str]
    hardware_ref: str
    min_overlap_frac: float
    bootstrap_resamples: int
    bootstrap_seed: int
    metrics: dict[str, MetricCI]
    latency_ms: dict[str, StageLatency]
    latency_passes: int
    index: IndexStats
    ann_sanity_recall: float | None
    per_query: list[QueryResult]
    created_at: str

    @model_validator(mode="after")
    def _consistent(self) -> RunResult:
        if self.config_id != self.config.config_id:
            raise ValueError(
                f"config_id {self.config_id!r} does not match config ({self.config.config_id!r})"
            )
        missing = [name for name in METRIC_NAMES if name not in self.metrics]
        if missing:
            raise ValueError(f"metrics missing required entries: {missing}")
        return self


def corpus_info_from_lock(lock_path: Path, *, name: str) -> CorpusInfo:
    """Corpus pin recorded in ``corpus.lock``, labeled with the config's corpus ``name``."""
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    return CorpusInfo(
        name=name,
        repo=str(lock["repo"]),
        tag=str(lock["tag"]),
        commit_sha=str(lock["commit_sha"]),
        sha256_of_files=str(lock["sha256_of_files"]),
    )


def package_versions(names: tuple[str, ...] = KEY_PACKAGES) -> dict[str, str]:
    """Installed versions of the packages that affect results (missing ones omitted)."""
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def run_path(results_dir: Path, config_id: str, split: str) -> Path:
    """``<results_dir>/runs/{config_id}__{split}.json``."""
    return results_dir / "runs" / f"{config_id}__{split}.json"


def write_run(result: RunResult, results_dir: Path) -> Path:
    """Write ``result`` as indented JSON; returns the file path."""
    path = run_path(results_dir, result.config_id, result.split)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def load_run(path: Path) -> RunResult:
    """Load and validate a run JSON file."""
    return RunResult.model_validate_json(path.read_text(encoding="utf-8"))


SUMMARY_COLUMNS: tuple[str, ...] = (
    "config_id",
    "split",
    "corpus",
    "chunker",
    "chunk_size",
    "embedder",
    "backend",
    "mode",
    "rerank",
    "rerank_top_n",
    "final_k",
    "n_questions",
    "provisional",
    *(col for name in METRIC_NAMES for col in (name, f"{name}_lo", f"{name}_hi")),
    "p50_total_ms",
    "p95_total_ms",
    "index_wall_seconds",
    "chunks",
    "size_bytes",
    "ann_sanity_recall",
    "created_at",
)


def summary_row(result: RunResult) -> dict[str, str]:
    """Flatten ``result`` into one ``summary.csv`` row keyed by :data:`SUMMARY_COLUMNS`."""
    cfg = result.config
    chunk_size = cfg.chunker.size_tokens if cfg.chunker.type != "semantic" else None
    total = result.latency_ms.get("total")
    row: dict[str, object] = {
        "config_id": result.config_id,
        "split": result.split,
        "corpus": cfg.corpus,
        "chunker": cfg.chunker.type,
        "chunk_size": chunk_size,
        "embedder": cfg.embedder.name,
        "backend": cfg.backend,
        "mode": cfg.retrieval.mode,
        "rerank": cfg.rerank.type,
        "rerank_top_n": cfg.rerank.top_n,
        "final_k": cfg.final_k,
        "n_questions": result.goldset.n_questions,
        "provisional": result.goldset.provisional,
        "p50_total_ms": total.p50 if total else None,
        "p95_total_ms": total.p95 if total else None,
        "index_wall_seconds": result.index.wall_seconds,
        "chunks": result.index.chunks,
        "size_bytes": result.index.size_bytes,
        "ann_sanity_recall": result.ann_sanity_recall,
        "created_at": result.created_at,
    }
    for name in METRIC_NAMES:
        metric = result.metrics[name]
        row[name] = metric.mean
        row[f"{name}_lo"] = metric.ci_lo
        row[f"{name}_hi"] = metric.ci_hi
    return {col: "" if row[col] is None else str(row[col]) for col in SUMMARY_COLUMNS}


def upsert_summary(result: RunResult, results_dir: Path) -> Path:
    """Insert or replace the ``(config_id, split)`` row of ``results/summary.csv``."""
    path = results_dir / "summary.csv"
    rows: list[dict[str, str]] = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as fh:
            rows = [dict(r) for r in csv.DictReader(fh)]
    new_row = summary_row(result)
    key = (new_row["config_id"], new_row["split"])
    for idx, existing in enumerate(rows):
        if (existing.get("config_id"), existing.get("split")) == key:
            rows[idx] = new_row
            break
    else:
        rows.append(new_row)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow({col: r.get(col, "") for col in SUMMARY_COLUMNS})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(buffer.getvalue(), encoding="utf-8", newline="")
    return path
