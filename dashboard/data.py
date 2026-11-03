"""Shared data loading and labels for the replay dashboard.

Replay mode: everything comes from committed files (``results/runs/*.json``,
``data/goldset/verified.jsonl``, ``docs/``). No vector database, model or LLM is touched.
Only light ragbench modules (pydantic schemas and bootstrap stats) are imported, so the
dashboard does not need torch.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # run without installing the package (deploys)
    sys.path.insert(0, str(ROOT / "src"))

from ragbench.config import PipelineConfig  # noqa: E402
from ragbench.eval.results import RunResult, load_run  # noqa: E402
from ragbench.sweep.select import ablations  # noqa: E402

RESULTS = Path(os.environ.get("RAGBENCH_RESULTS_DIR", ROOT / "results"))
METRICS = ["ndcg@10", "mrr@10", "recall@1", "recall@3", "recall@5", "recall@10"]
GEN_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
REFUSAL = "does not contain the answer"

EMBEDDERS = {
    "BAAI/bge-small-en-v1.5": "bge-small",
    "intfloat/e5-small-v2": "e5-small",
    "sentence-transformers/all-MiniLM-L6-v2": "MiniLM",
}
BACKENDS = {"pgvector": "pgvector", "qdrant": "Qdrant", "control_bm25": "BM25 control"}


# --- labels ---------------------------------------------------------------------------


def mode(cfg: PipelineConfig) -> str:
    return "sparse" if cfg.backend == "control_bm25" else cfg.retrieval.mode


def chunker_label(cfg: PipelineConfig) -> str:
    ch = cfg.chunker
    name = ch.type.replace("_", "-")
    return name if ch.type == "semantic" else f"{name} {ch.size_tokens}"


def embedder_label(cfg: PipelineConfig) -> str:
    return EMBEDDERS.get(cfg.embedder.name, cfg.embedder.name.split("/")[-1])


def rerank_label(cfg: PipelineConfig) -> str:
    return "no rerank" if cfg.rerank.type == "none" else f"rerank@{cfg.rerank.top_n}"


def backend_label(cfg: PipelineConfig) -> str:
    return BACKENDS.get(cfg.backend, cfg.backend)


def retrieval_label(cfg: PipelineConfig) -> str:
    """Backend · mode · rerank (the part that differs between finalists)."""
    if cfg.backend == "control_bm25":
        return f"BM25 control · {rerank_label(cfg)}"
    return f"{backend_label(cfg)} · {cfg.retrieval.mode} · {rerank_label(cfg)}"


def full_label(cfg: PipelineConfig) -> str:
    return f"{chunker_label(cfg)} · {embedder_label(cfg)} · {retrieval_label(cfg)}"


FACTOR_VALUE = {
    "chunker": chunker_label,
    "embedder": embedder_label,
    "rerank": rerank_label,
    "backend": backend_label,
    "retrieval": lambda c: "BM25 control" if c.backend == "control_bm25" else c.retrieval.mode,
}


def context_label(cfg: PipelineConfig, factor: str) -> str:
    """Everything except ``factor``, for the 'held fixed' part of an ablation row."""
    parts = {
        "chunker": chunker_label(cfg),
        "embedder": embedder_label(cfg),
        "retrieval": FACTOR_VALUE["retrieval"](cfg),
        "rerank": rerank_label(cfg),
        "backend": backend_label(cfg) if cfg.backend != "control_bm25" else "",
    }
    return " · ".join(v for k, v in parts.items() if k != factor and v)


# --- data -----------------------------------------------------------------------------


@st.cache_resource
def runs(split: str) -> list[RunResult]:
    out = []
    for path in sorted((RESULTS / "runs").glob(f"*__{split}.json")):
        result = load_run(path)
        if result.config.corpus == "fastapi_docs":
            out.append(result)
    return out


def ranked(split: str) -> list[RunResult]:
    return sorted(
        runs(split),
        key=lambda r: (-r.metrics["ndcg@10"].mean, -r.metrics["mrr@10"].mean, r.config_id),
    )


@st.cache_data
def table(split: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rank, r in enumerate(ranked(split), start=1):
        cfg = r.config
        row: dict[str, Any] = {
            "rank": rank,
            "pipeline": retrieval_label(cfg),
            "chunker": chunker_label(cfg),
            "embedder": embedder_label(cfg),
            "mode": mode(cfg),
            "rerank": rerank_label(cfg),
            "backend": backend_label(cfg),
            "p95_ms": r.latency_ms["total"].p95,
            "config_id": r.config_id,
        }
        for m in METRICS:
            row[m] = r.metrics[m].mean
            row[f"{m}_lo"] = r.metrics[m].ci_lo
            row[f"{m}_hi"] = r.metrics[m].ci_hi
        rows.append(row)
    return pd.DataFrame(rows)


@st.cache_data
def ablation_table(split: str, metric: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ab in ablations(runs(split), metric, seed=42):
        value = FACTOR_VALUE[ab.factor]
        rows.append(
            {
                "factor": ab.factor,
                "change": f"{value(ab.a.config)} vs {value(ab.b.config)}",
                "held fixed": context_label(ab.a.config, ab.factor),
                "delta": ab.delta,
                "lo": ab.lo,
                "hi": ab.hi,
                "significant": ab.significant,
                "latency_delta_ms": ab.latency_delta_ms,
            }
        )
    return pd.DataFrame(rows)


def refusal_rate(result: RunResult) -> float | None:
    gen = result.generation
    if gen is None or not gen.per_query:
        return None
    return sum(REFUSAL in q.answer.lower() for q in gen.per_query) / len(gen.per_query)


@st.cache_data
def gold_questions() -> dict[str, dict[str, Any]]:
    path = ROOT / "data" / "goldset" / "verified.jsonl"
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            q = json.loads(line)
            out[q["qid"]] = q
    return out


@st.cache_data
def documents() -> dict[str, str] | None:
    """Full document texts if the corpus has been ingested locally, else ``None``."""
    path = ROOT / "data" / "documents.jsonl"
    if not path.exists():
        return None
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            doc = json.loads(line)
            out[doc["doc_id"]] = doc["text"]
    return out
