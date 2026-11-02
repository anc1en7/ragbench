"""Unit tests for the evaluation runner (Task 3.5) — offline.

The end-to-end test evaluates the committed mini fixture with the in-process
``control_bm25`` backend (real bm25s + cached tiktoken, no docker, no model download).
ANN sanity is tested with a fake store/embedder over hand-built vectors.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from ragbench.config import PipelineConfig
from ragbench.eval.results import load_run
from ragbench.eval.retrieval_metrics import METRIC_NAMES
from ragbench.eval.runner import (
    ann_applicable,
    ann_sanity_recall,
    evaluate,
    load_split_questions,
    resolve_dataset,
    write_baseline,
)
from ragbench.pipeline import TIMING_KEYS, Pipeline
from ragbench.rerank import RerankNone
from ragbench.types import Chunk, Document, Hit


def _cfg(**overrides: object) -> PipelineConfig:
    data: dict[str, object] = {
        "corpus": "mini",
        "chunker": {"type": "recursive_md", "size_tokens": 256, "overlap_tokens": 32},
        "embedder": {"name": "BAAI/bge-small-en-v1.5"},
        "backend": "control_bm25",
        "retrieval": {"mode": "sparse"},
        "rerank": {"type": "none"},
        "generation": {"enabled": False},
    }
    data.update(overrides)
    return PipelineConfig.model_validate(data)


# --- datasets / splits -----------------------------------------------------------------


def test_resolve_dataset_known_and_unknown():
    assert resolve_dataset("mini").documents.name == "documents.jsonl"
    with pytest.raises(ValueError, match="unknown corpus"):
        resolve_dataset("nope")


def _question(qid: str) -> dict[str, object]:
    return {
        "qid": qid,
        "question": f"question {qid}",
        "reference_answer": "a",
        "gold_sections": [],
        "gold_spans": [{"doc_id": "a.md", "start": 0, "end": 10}],
        "question_type": "factual",
        "source_doc": "a.md",
        "verified": True,
    }


def test_load_split_questions_filters_and_sorts(tmp_path: Path):
    questions = tmp_path / "q.jsonl"
    questions.write_text(
        "\n".join(json.dumps(_question(q)) for q in ("q3", "q1", "q2")), encoding="utf-8"
    )
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"seed": 42, "dev": ["q3", "q1", "q9"], "test": ["q2"]}))

    dev = load_split_questions(questions, splits, "dev")
    assert [q.qid for q in dev] == ["q1", "q3"]
    assert [q.qid for q in load_split_questions(questions, splits, "test")] == ["q2"]
    with pytest.raises(ValueError, match="split"):
        load_split_questions(questions, splits, "train")


def test_mini_fixture_dev_split_has_nine_questions():
    ds = resolve_dataset("mini")
    from ragbench.eval.runner import SPLITS_PATH

    assert len(load_split_questions(ds.questions, SPLITS_PATH, "dev")) == 9
    assert len(load_split_questions(ds.questions, SPLITS_PATH, "test")) == 31


# --- ANN sanity ------------------------------------------------------------------------


class FakeEmbedder:
    name = "fake"
    dim = 2

    def __init__(self, qvecs: dict[str, np.ndarray]) -> None:
        self.qvecs = qvecs

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        raise AssertionError("not used")

    def embed_query(self, text: str) -> np.ndarray:
        return self.qvecs[text]


class FakeStore:
    """Returns a scripted ranking of chunk ids for every dense query."""

    def __init__(self, ranking: list[str]) -> None:
        self.ranking = ranking

    def reset(self, namespace: str, dim: int) -> None: ...

    def add(self, namespace: str, chunks: Sequence[Chunk], vectors: np.ndarray) -> None: ...

    def dense_search(self, namespace: str, qvec: np.ndarray, k: int) -> list[Hit]:
        return [
            Hit(Chunk(chunk_id=cid, doc_id="d", start=0, end=1, text=""), 1.0, "dense")
            for cid in self.ranking[:k]
        ]

    def sparse_search(self, namespace: str, query: str, k: int) -> list[Hit]:
        return []

    def size_bytes(self, namespace: str) -> int:
        return 0


class NullChunker:
    def chunk(self, doc: Document) -> list[Chunk]:
        return []


# Unit vectors at angles 0, 10, 20, 90 degrees; the query points along 0 degrees, so the
# exact cosine order is a, b, c, d.
_VECS = np.array(
    [[np.cos(np.radians(a)), np.sin(np.radians(a))] for a in (0, 10, 20, 90)], dtype=np.float32
)
_IDS = ["a", "b", "c", "d"]


def _pipe(ranking: list[str]) -> Pipeline:
    cfg = _cfg(backend="pgvector", retrieval={"mode": "dense"})
    return Pipeline(
        cfg,
        store=FakeStore(ranking),
        embedder=FakeEmbedder({"q": np.array([1.0, 0.0], dtype=np.float32)}),
        reranker=RerankNone(),
        chunker=NullChunker(),
    )


def test_ann_sanity_perfect_agreement():
    assert ann_sanity_recall(_pipe(["a", "b", "c"]), _IDS, _VECS, ["q"], k=2) == 1.0


def test_ann_sanity_partial_agreement():
    # exact top-2 = {a, b}; ANN returned {a, d} -> 1 of 2 correct
    assert ann_sanity_recall(_pipe(["a", "d"]), _IDS, _VECS, ["q"], k=2) == 0.5


def test_ann_sanity_ties_are_not_misses():
    # b and b2 are identical vectors (duplicate chunk text): either one is a correct 2nd hit
    vecs = np.vstack([_VECS[:2], _VECS[1:2], _VECS[2:]])
    ids = ["a", "b", "b2", "c", "d"]
    assert ann_sanity_recall(_pipe(["a", "b2"]), ids, vecs, ["q"], k=2) == 1.0


def test_ann_applicability():
    assert ann_applicable(_cfg(backend="pgvector", retrieval={"mode": "dense"}))
    assert ann_applicable(_cfg(backend="qdrant", retrieval={"mode": "hybrid"}))
    assert not ann_applicable(_cfg(backend="pgvector", retrieval={"mode": "sparse"}))
    assert not ann_applicable(_cfg())


# --- end to end (control_bm25, offline) ------------------------------------------------


def test_evaluate_mini_dev_with_control_bm25(tmp_path: Path):
    cfg = _cfg()
    result, path = evaluate(cfg, "dev", results_dir=tmp_path, passes=3, n_resamples=200)

    assert path == tmp_path / "runs" / f"{cfg.config_id}__dev.json"
    assert load_run(path) == result
    assert result.goldset.n_questions == 9
    assert result.goldset.all_verified and not result.goldset.provisional
    assert set(result.metrics) == set(METRIC_NAMES)
    for metric in result.metrics.values():
        assert 0.0 <= metric.ci_lo <= metric.mean <= metric.ci_hi <= 1.0
    assert result.metrics["recall@10"].mean > 0.0  # BM25 finds some gold sections
    assert set(result.latency_ms) == set(TIMING_KEYS)
    assert result.latency_passes == 3
    assert result.ann_sanity_recall is None  # sparse-only control
    assert result.index.docs == 30 and result.index.chunks > 30
    assert result.index.chunks_per_sec is None  # no embedding for BM25
    assert len(result.per_query) == 9
    assert all(len(q.relevance) == len(q.hits) for q in result.per_query)
    assert result.corpus.commit_sha  # pinned from corpus.lock

    with (tmp_path / "summary.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [(r["config_id"], r["split"]) for r in rows] == [(cfg.config_id, "dev")]


def test_write_baseline(tmp_path: Path):
    result, _ = evaluate(_cfg(), "dev", results_dir=tmp_path, n_resamples=50)
    path = write_baseline(result, tmp_path / "eval" / "baseline.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["config_id"] == result.config_id
    assert data["recall@5"] == result.metrics["recall@5"].mean
    assert data["mrr@10"] == result.metrics["mrr@10"].mean
    assert data["split"] == "dev" and data["corpus"] == "mini"
