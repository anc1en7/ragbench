"""Integration tests for the assembled pipeline and ``ragbench query`` (Task 2.9, ruling 10).

Requires the docker compose stack (``make up``): Postgres and Qdrant must be reachable or
the suite fails explicitly rather than skipping. Per backend (pgvector=hybrid,
qdrant=dense, control_bm25=sparse) a tiny synthetic corpus is indexed through
``index_documents`` with a small fixed-window config and retrieved for a question
containing a distinctive planted term; then ``CliRunner`` drives ``ragbench query``.

``rerank.type`` is ``none`` in EVERY config here — no cross-encoder model download
(Phase 4 concern). The two vector backends embed with the real
``BAAI/bge-small-en-v1.5`` SentenceTransformer (one-time download into ``.hf_cache``);
control_bm25 skips dense embedding entirely (ruling 10) and its CLI query shows the
index hint because its index is in-process only.
"""

from __future__ import annotations

import math
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import psycopg
import pytest
from typer.testing import CliRunner

from ragbench.cli import app
from ragbench.config import load_pipeline_config
from ragbench.pipeline import build_pipeline, index_documents
from ragbench.stores.pgvector_store import DEFAULT_PG_DSN
from ragbench.stores.qdrant_store import DEFAULT_QDRANT_URL
from ragbench.types import Document

pytestmark = pytest.mark.integration

TIMING_KEYS = {"embed_query", "dense", "sparse", "fusion", "rerank", "total"}

PLANTED_DOC_ID = "query_planted.md"
QUESTION = "Where does the rare xylotomic term appear in the documentation?"

# Vocabulary deliberately avoids rare/term/appear*/documentation so the sparse side of
# the hybrid config matches ONLY the planted document (keeps the fused rank-1 proof exact).
_GUIDE_FILLER = (
    "The routing layer maps HTTP verbs onto handler functions, validates incoming "
    "payloads against the request model, and dispatches the call into the service "
    "layer. Middleware wraps every handler with logging, exception translation and "
    "timeout enforcement so failures surface as consistent error responses."
)


def _corpus() -> list[Document]:
    documents = [
        Document(
            doc_id=f"guide_{i}.md",
            title=f"Guide {i}",
            text=(
                f"Guide chapter {i}. {_GUIDE_FILLER} "
                f"Chapter {i} also covers setup ordering, lazy imports and shutdown hooks."
            ),
            sections=[],
        )
        for i in range(5)
    ]
    planted_text = (
        "Reference appendix. The rare xylotomic term appears only in this planted "
        f"appendix chunk so the integration suite can verify retrieval end to end. {_GUIDE_FILLER}"
    )
    documents.append(
        Document(
            doc_id=PLANTED_DOC_ID,
            title="Appendix",
            text=planted_text,
            sections=[],
        )
    )
    return documents


def _write_config(tmp_path: Path, backend: str, mode: str) -> Path:
    path = tmp_path / f"pipeline_{backend}.yaml"
    path.write_text(
        (
            f"corpus: it_query_{backend}\n"
            "chunker: {type: fixed, size_tokens: 64, overlap_tokens: 8}\n"
            "embedder: {name: BAAI/bge-small-en-v1.5}\n"
            f"backend: {backend}\n"
            f"retrieval: {{mode: {mode}, k_dense: 50, k_sparse: 50, rrf_k: 60}}\n"
            "rerank: {type: none, top_n: 50}\n"
            "final_k: 10\n"
            "generation: {enabled: false}\n"
            "seed: 42\n"
        ),
        encoding="utf-8",
    )
    return path


def _reachable() -> list[str]:
    missing: list[str] = []
    try:
        with psycopg.connect(DEFAULT_PG_DSN, connect_timeout=3) as conn:
            conn.execute("SELECT 1")
    except psycopg.OperationalError:
        missing.append("Postgres")
    try:
        with urlopen(f"{DEFAULT_QDRANT_URL}/collections", timeout=3) as resp:
            if resp.status != 200:
                missing.append("Qdrant")
    except (URLError, TimeoutError, OSError):
        missing.append("Qdrant")
    return missing


@pytest.fixture(autouse=True)
def _require_backends() -> None:
    missing = _reachable()
    if missing:
        pytest.fail(f"{' and '.join(missing)} unreachable — run `make up` first", pytrace=False)


@pytest.mark.parametrize(
    ("backend", "mode", "expected_source"),
    [
        ("pgvector", "hybrid", "fused"),
        ("qdrant", "dense", "dense"),
        ("control_bm25", "sparse", "sparse"),
    ],
)
def test_index_then_retrieve_returns_ranked_hits_with_six_timings(
    tmp_path: Path,
    backend: str,
    mode: str,
    expected_source: str,
) -> None:
    cfg = load_pipeline_config(_write_config(tmp_path, backend, mode))
    pipeline = build_pipeline(cfg)
    documents = _corpus()

    stats = index_documents(documents, pipeline)

    assert stats["docs"] == len(documents)
    assert stats["chunks"] > stats["docs"]  # 64-token windows split docs into chunks
    assert stats["index_seconds"] > 0.0
    if backend == "control_bm25":
        assert stats["embed_seconds"] == 0.0  # dense embedding skipped (ruling 10)
    else:
        assert stats["embed_seconds"] > 0.0

    hits, timings = pipeline.retrieve(QUESTION)

    assert hits, "expected at least one ranked hit"
    scores = [hit.score for hit in hits]
    assert all(math.isfinite(score) for score in scores)
    assert scores == sorted(scores, reverse=True)
    assert set(timings) == TIMING_KEYS
    assert timings["total"] > 0.0

    if mode == "hybrid":
        assert timings["embed_query"] > 0.0
        assert timings["dense"] > 0.0
        assert timings["sparse"] > 0.0
        assert timings["fusion"] > 0.0
        # sparse matches ONLY the planted doc and k_dense covers every chunk, so the
        # planted doc (rank 1 in sparse + present in dense) must win the RRF fusion
        assert hits[0].chunk.doc_id == PLANTED_DOC_ID
    elif mode == "dense":
        assert timings["embed_query"] > 0.0
        assert timings["dense"] > 0.0
        assert timings["sparse"] == 0.0
        assert timings["fusion"] == 0.0
        # bge similarity: question and planted chunk share the distinctive phrasing
        assert hits[0].chunk.doc_id == PLANTED_DOC_ID
    else:  # sparse
        assert timings["embed_query"] == 0.0
        assert timings["dense"] == 0.0
        assert timings["fusion"] == 0.0
        assert timings["sparse"] > 0.0
        assert hits[0].chunk.doc_id == PLANTED_DOC_ID

    assert {hit.source for hit in hits} == {expected_source}


def test_query_cli_prints_ranked_chunks_and_timings_table(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path, "pgvector", "hybrid")
    cfg = load_pipeline_config(cfg_path)
    index_documents(_corpus(), build_pipeline(cfg))

    result = CliRunner().invoke(
        app, ["query", "--config", str(cfg_path), QUESTION], env={"COLUMNS": "160"}
    )

    output = result.output
    assert result.exit_code == 0, output
    assert PLANTED_DOC_ID in output  # the expected doc is printed
    assert "fused" in output
    for stage in ("embed_query", "dense", "sparse", "fusion", "rerank", "total"):
        assert stage in output
    assert "total" in output


def test_query_cli_shows_index_hint_when_namespace_is_empty(tmp_path: Path) -> None:
    # qdrant + sparse with nothing indexed: dense path never runs, no model downloads,
    # absent collection -> zero hits -> hint (exit 0 still)
    cfg_path = _write_config(tmp_path, "qdrant", "sparse")

    result = CliRunner().invoke(
        app, ["query", "--config", str(cfg_path), QUESTION], env={"COLUMNS": "160"}
    )

    assert result.exit_code == 0, result.output
    assert "ragbench index --config" in result.output
