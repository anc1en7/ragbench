"""End-to-end evaluation of one pipeline config on one split (Task 3.5).

``evaluate`` indexes the config's corpus in-process (so the in-process ``control_bm25``
backend works too), profiles retrieval over the split's questions (``passes >= 3``, pass 0
discarded), scores every question by span overlap, aggregates metrics with seeded
bootstrap CIs, runs the ANN exact-search sanity check, and writes a schema-validated
:class:`~ragbench.eval.results.RunResult` plus its ``summary.csv`` row.

Datasets are selected by ``cfg.corpus`` (so ``config_id`` always identifies the data):

- ``fastapi_docs``: ``data/documents.jsonl`` + ``data/goldset/verified.jsonl``
- ``mini``: ``data/fixtures/mini/{documents,questions}.jsonl`` (CI fixture)

Both use ``data/goldset/splits.json``; a question belongs to a split iff its qid is listed
there (the mini fixture's 40 questions split 9 dev / 31 test). ``provisional`` is set
when the human-verified gold set has fewer than 150 questions or any evaluated question
is unverified (spec section 5.6 USER ACTION rule).

ANN sanity (spec section 5.3): for up to ``ann_sample`` seeded split questions, compare
the store's HNSW ``dense_search`` top-k against brute-force cosine over every chunk
vector. Agreement counts an ANN result as correct when its exact cosine is at least the
exact k-th best cosine (minus 1e-6), so ties between identical chunk texts are not
counted as misses. ``None`` for sparse-only modes and the BM25 control.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ragbench.config import PipelineConfig
from ragbench.eval.latency import measure_indexing, profile_queries
from ragbench.eval.results import (
    GoldsetInfo,
    IndexStats,
    MetricCI,
    QueryResult,
    RetrievedChunk,
    RunResult,
    Split,
    StageLatency,
    corpus_info_from_lock,
    package_versions,
    upsert_summary,
    write_run,
)
from ragbench.eval.retrieval_metrics import METRIC_NAMES, aggregate, query_metrics
from ragbench.eval.spans import DEFAULT_MIN_OVERLAP_FRAC, span_hits
from ragbench.eval.stats import DEFAULT_RESAMPLES, bootstrap_ci
from ragbench.goldset.generate import load_documents
from ragbench.goldset.review import load_candidates
from ragbench.goldset.schema import Candidate
from ragbench.pipeline import Pipeline, build_pipeline, index_documents
from ragbench.types import Hit

REPO_ROOT = Path(__file__).resolve().parents[3]
MIN_VERIFIED: int = 150
ANN_TIE_EPS: float = 1e-6


@dataclass(frozen=True)
class Dataset:
    documents: Path
    questions: Path


DATASETS: dict[str, Dataset] = {
    "fastapi_docs": Dataset(
        documents=REPO_ROOT / "data" / "documents.jsonl",
        questions=REPO_ROOT / "data" / "goldset" / "verified.jsonl",
    ),
    "mini": Dataset(
        documents=REPO_ROOT / "data" / "fixtures" / "mini" / "documents.jsonl",
        questions=REPO_ROOT / "data" / "fixtures" / "mini" / "questions.jsonl",
    ),
}
SPLITS_PATH = REPO_ROOT / "data" / "goldset" / "splits.json"
GOLDSET_PATH = REPO_ROOT / "data" / "goldset" / "verified.jsonl"
CORPUS_LOCK_PATH = REPO_ROOT / "corpus.lock"
HARDWARE_REF = "results/hardware.json"


def resolve_dataset(corpus: str) -> Dataset:
    """Dataset paths for a config's ``corpus`` name."""
    try:
        return DATASETS[corpus]
    except KeyError:
        known = ", ".join(sorted(DATASETS))
        raise ValueError(f"unknown corpus {corpus!r}; known corpora: {known}") from None


def load_split_questions(questions_path: Path, splits_path: Path, split: str) -> list[Candidate]:
    """Questions of ``questions_path`` whose qid is in ``splits_path[split]``, qid-sorted."""
    if not splits_path.exists():
        raise FileNotFoundError(f"splits file not found: {splits_path} (run goldset split)")
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    if split not in ("dev", "test"):
        raise ValueError(f"split must be 'dev' or 'test', got {split!r}")
    wanted = set(splits[split])
    questions = sorted(
        (q for q in load_candidates(questions_path) if q.qid in wanted), key=lambda q: q.qid
    )
    if not questions:
        raise ValueError(f"no questions from {questions_path} are in the {split!r} split")
    return questions


def ann_applicable(cfg: PipelineConfig) -> bool:
    """The ANN check needs a dense index: dense/hybrid modes on a vector backend."""
    return cfg.retrieval.mode in ("dense", "hybrid") and cfg.backend != "control_bm25"


def ann_sanity_recall(
    pipeline: Pipeline,
    chunk_ids: Sequence[str],
    vectors: np.ndarray,
    questions: Sequence[str],
    *,
    k: int = 10,
) -> float:
    """Mean tie-aware agreement of the store's ANN top-k with exact cosine top-k."""
    if not questions:
        raise ValueError("ann_sanity_recall needs at least one question")
    matrix = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.where(norms == 0, 1.0, norms)
    index_of = {chunk_id: i for i, chunk_id in enumerate(chunk_ids)}
    k_eff = min(k, len(chunk_ids))
    namespace = pipeline.cfg.config_id
    agreements: list[float] = []
    for question in questions:
        qvec = np.asarray(pipeline.embedder.embed_query(question), dtype=np.float64)
        qnorm = np.linalg.norm(qvec)
        sims = matrix @ (qvec / (qnorm if qnorm else 1.0))
        kth_best = float(np.sort(sims)[::-1][k_eff - 1])
        ann_hits = pipeline.store.dense_search(namespace, qvec.astype(np.float32), k_eff)
        correct = sum(
            1
            for hit in ann_hits[:k_eff]
            if hit.chunk.chunk_id in index_of
            and sims[index_of[hit.chunk.chunk_id]] >= kth_best - ANN_TIE_EPS
        )
        agreements.append(correct / k_eff)
    return float(np.mean(agreements))


def _retrieved(hits: Sequence[Hit]) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=hit.chunk.chunk_id,
            doc_id=hit.chunk.doc_id,
            start=hit.chunk.start,
            end=hit.chunk.end,
            score=hit.score,
            source=hit.source,
        )
        for hit in hits
    ]


def _goldset_info(questions: Sequence[Candidate], goldset_path: Path) -> GoldsetInfo:
    n_verified = sum(1 for q in load_candidates(goldset_path) if q.verified)
    all_verified = all(q.verified for q in questions)
    return GoldsetInfo(
        n_questions=len(questions),
        all_verified=all_verified,
        provisional=(not all_verified) or n_verified < MIN_VERIFIED,
    )


def evaluate(
    cfg: PipelineConfig,
    split: Split,
    *,
    results_dir: Path,
    passes: int = 3,
    min_overlap_frac: float = DEFAULT_MIN_OVERLAP_FRAC,
    n_resamples: int = DEFAULT_RESAMPLES,
    ann_sample: int = 50,
    pipeline: Pipeline | None = None,
    splits_path: Path = SPLITS_PATH,
    goldset_path: Path = GOLDSET_PATH,
    corpus_lock: Path = CORPUS_LOCK_PATH,
) -> tuple[RunResult, Path]:
    """Index, retrieve, score and persist one run; returns ``(result, run_json_path)``."""
    dataset = resolve_dataset(cfg.corpus)
    documents = load_documents(dataset.documents)
    questions = load_split_questions(dataset.questions, splits_path, split)
    pipe = pipeline if pipeline is not None else build_pipeline(cfg)
    namespace = cfg.config_id

    index = measure_indexing(
        lambda: index_documents(documents, pipe),
        size_fn=lambda: pipe.store.size_bytes(namespace),
    )

    texts = [q.question for q in questions]
    ranked, latency = profile_queries(pipe.retrieve, texts, passes=passes)

    per_query: list[QueryResult] = []
    for question, hits in zip(questions, ranked, strict=True):
        relevance = span_hits(hits, question.gold_spans, min_overlap_frac)
        per_query.append(
            QueryResult(
                qid=question.qid,
                question_type=question.question_type,
                relevance=relevance,
                metrics=query_metrics(relevance, len(question.gold_spans)),
                hits=_retrieved(hits),
            )
        )

    means = aggregate([q.metrics for q in per_query])
    metrics: dict[str, MetricCI] = {}
    for name in METRIC_NAMES:
        lo, hi = bootstrap_ci([q.metrics[name] for q in per_query], n_resamples, cfg.seed)
        metrics[name] = MetricCI(mean=means[name], ci_lo=lo, ci_hi=hi)

    ann_recall: float | None = None
    if ann_applicable(cfg):
        chunks = [chunk for doc in documents for chunk in pipe.chunker.chunk(doc)]
        vectors = pipe.embedder.embed_documents([chunk.text for chunk in chunks])
        sample = list(texts)
        random.Random(cfg.seed).shuffle(sample)
        ann_recall = ann_sanity_recall(
            pipe,
            [chunk.chunk_id for chunk in chunks],
            vectors,
            sorted(sample[:ann_sample]),
            k=cfg.final_k,
        )

    result = RunResult(
        config_id=cfg.config_id,
        config=cfg,
        seed=cfg.seed,
        split=split,
        corpus=corpus_info_from_lock(corpus_lock, name=cfg.corpus),
        goldset=_goldset_info(questions, goldset_path),
        package_versions=package_versions(),
        hardware_ref=HARDWARE_REF,
        min_overlap_frac=min_overlap_frac,
        bootstrap_resamples=n_resamples,
        bootstrap_seed=cfg.seed,
        metrics=metrics,
        latency_ms={stage: StageLatency(**summary) for stage, summary in latency.items()},
        latency_passes=passes,
        index=IndexStats(**_index_fields(index)),
        ann_sanity_recall=ann_recall,
        per_query=per_query,
        created_at=datetime.now(UTC).isoformat(),
    )
    path = write_run(result, results_dir)
    upsert_summary(result, results_dir)
    close = getattr(pipe.store, "close", None)
    if callable(close):
        close()
    return result, path


def _index_fields(stats: dict[str, Any]) -> dict[str, Any]:
    return {name: stats[name] for name in IndexStats.model_fields}


def write_baseline(result: RunResult, path: Path) -> Path:
    """Write the CI regression baseline (Recall@5, MRR@10) from ``result``."""
    payload = {
        "config_id": result.config_id,
        "corpus": result.config.corpus,
        "split": result.split,
        "n_questions": result.goldset.n_questions,
        "recall@5": result.metrics["recall@5"].mean,
        "mrr@10": result.metrics["mrr@10"].mean,
        "created_at": result.created_at,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
