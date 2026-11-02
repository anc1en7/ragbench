"""Unit tests for pipeline assembly, retrieval dispatch, timings, indexing and the config
loader (Task 2.9, controller rulings 2-7 and 9).

Everything here is offline: fake store/embedder/reranker/chunker only, no network and no
docker. Covered: mode dispatch (dense/sparse/hybrid with fuse spy), the exact six timing
keys in every mode with 0.0 for unexecuted stages, top_n truncation before rerank,
namespace == config_id at the store, generate() stub message, build_pipeline override
seams and defaults, index_documents ordering/stats (including the ControlBM25 dense-embed
skip), and load_pipeline_config round-trip plus bad-file errors.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import yaml

from ragbench.chunking import FixedChunker, SemanticChunker
from ragbench.config import PipelineConfig, load_pipeline_config
from ragbench.embedding.st_embedder import SentenceTransformerEmbedder
from ragbench.pipeline import Pipeline, build_pipeline, index_documents
from ragbench.rerank import RerankNone
from ragbench.rerank.cross_encoder import CrossEncoderReranker
from ragbench.stores import ControlBM25Store, QdrantStore
from ragbench.types import Chunk, Document, Hit

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG_PATH = ROOT / "configs" / "pipeline.example.yaml"

EXPECTED_TIMING_KEYS = {"embed_query", "dense", "sparse", "fusion", "rerank", "total"}


def _chunk(i: int, text: str | None = None) -> Chunk:
    body = text if text is not None else f"chunk {i} body"
    return Chunk(chunk_id=f"c{i:03d}", doc_id=f"d{i:03d}", start=0, end=len(body), text=body)


def _hit(i: int, score: float, source: str) -> Hit:
    return Hit(chunk=_chunk(i), score=score, source=source)


class FakeStore:
    """Recording store: returns canned hits and logs every call (namespace included)."""

    def __init__(
        self,
        dense: list[Hit] | None = None,
        sparse: list[Hit] | None = None,
    ) -> None:
        self.dense_hits: list[Hit] = dense or []
        self.sparse_hits: list[Hit] = sparse or []
        self.events: list[str] = []
        self.dense_calls: list[tuple[str, int]] = []
        self.sparse_calls: list[tuple[str, str, int]] = []
        self.reset_calls: list[tuple[str, int]] = []
        self.add_calls: list[tuple[str, list[Chunk], np.ndarray]] = []

    def reset(self, namespace: str, dim: int) -> None:
        self.events.append("reset")
        self.reset_calls.append((namespace, dim))

    def add(self, namespace: str, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        self.events.append("add")
        self.add_calls.append((namespace, list(chunks), vectors))

    def dense_search(self, namespace: str, qvec: np.ndarray, k: int) -> list[Hit]:
        self.events.append("dense")
        self.dense_calls.append((namespace, k))
        return list(self.dense_hits)

    def sparse_search(self, namespace: str, query: str, k: int) -> list[Hit]:
        self.events.append("sparse")
        self.sparse_calls.append((namespace, query, k))
        return list(self.sparse_hits)

    def size_bytes(self, namespace: str) -> int:
        return 0


class FakeEmbedder:
    """Recording embedder: zero vectors, logs query texts and document batches."""

    name = "fake-embedder"
    dim = 4

    def __init__(self) -> None:
        self.query_texts: list[str] = []
        self.document_batches: list[list[str]] = []

    def embed_documents(self, texts: list[str] | tuple[str, ...] | np.ndarray) -> np.ndarray:
        batch = [str(text) for text in texts]
        self.document_batches.append(batch)
        return np.zeros((len(batch), self.dim), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        self.query_texts.append(text)
        return np.zeros(self.dim, dtype=np.float32)


class FakeChunker:
    """One chunk per document: the whole text."""

    def chunk(self, doc: Document) -> list[Chunk]:
        return [
            Chunk(
                chunk_id=f"{doc.doc_id}#0",
                doc_id=doc.doc_id,
                start=0,
                end=len(doc.text),
                text=doc.text,
            )
        ]


class RecordingReranker:
    """Records (query, hits passed in, k) and returns the first k hits unchanged."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[Hit], int]] = []

    def rerank(self, query: str, hits: list[Hit] | tuple[Hit, ...], k: int) -> list[Hit]:
        self.calls.append((query, list(hits), k))
        return list(hits)[: max(k, 0)]


class FuseSpy:
    def __init__(self) -> None:
        self.calls: list[list[list[Hit]]] = []


def _spy_fuse(monkeypatch: pytest.MonkeyPatch) -> FuseSpy:
    from ragbench.retrieval import rrf

    spy = FuseSpy()

    def wrapper(lists: list[list[Hit]], k: int = 60) -> list[Hit]:
        spy.calls.append(lists)
        return rrf.fuse(lists, k)

    monkeypatch.setattr("ragbench.pipeline.fuse", wrapper)
    return spy


def _cfg(**overrides: object) -> PipelineConfig:
    data: dict[str, object] = {
        "corpus": "unit_corpus",
        "chunker": {"type": "fixed", "size_tokens": 256, "overlap_tokens": 32},
        "embedder": {"name": "BAAI/bge-small-en-v1.5"},
        "backend": "pgvector",
        "retrieval": {"mode": "dense", "k_dense": 5, "k_sparse": 5, "rrf_k": 60},
        "rerank": {"type": "none", "top_n": 50},
        "final_k": 3,
        "generation": {"enabled": False},
        "seed": 42,
    }
    data.update(overrides)
    return PipelineConfig.model_validate(data)


def _pipeline(
    cfg: PipelineConfig,
    store: FakeStore,
    *,
    embedder: FakeEmbedder | None = None,
    reranker: RecordingReranker | None = None,
) -> Pipeline:
    return build_pipeline(
        cfg,
        store=store,
        embedder=embedder if embedder is not None else FakeEmbedder(),
        reranker=reranker if reranker is not None else RecordingReranker(),
        chunker=FakeChunker(),
    )


# --- ruling 3: mode dispatch -------------------------------------------------------------


def test_dense_mode_runs_dense_only_and_zeroes_sparse_and_fusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg()  # mode: dense
    store = FakeStore(dense=[_hit(1, 0.9, "dense"), _hit(2, 0.8, "dense")])
    spy = _spy_fuse(monkeypatch)
    embedder = FakeEmbedder()
    pipeline = _pipeline(cfg, store, embedder=embedder)

    hits, timings = pipeline.retrieve("what is a query parameter?")

    assert store.dense_calls == [(cfg.config_id, 5)]
    assert store.sparse_calls == []
    assert spy.calls == []  # no fusion in single-mode dispatch
    assert embedder.query_texts == ["what is a query parameter?"]
    assert timings["sparse"] == 0.0
    assert timings["fusion"] == 0.0
    assert [hit.source for hit in hits] == ["dense", "dense"]
    assert set(timings) == EXPECTED_TIMING_KEYS
    assert timings["total"] > 0.0


def test_sparse_mode_skips_embedding_and_fusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(retrieval={"mode": "sparse", "k_dense": 5, "k_sparse": 7, "rrf_k": 60})
    store = FakeStore(sparse=[_hit(3, 2.5, "sparse")])
    spy = _spy_fuse(monkeypatch)
    embedder = FakeEmbedder()
    pipeline = _pipeline(cfg, store, embedder=embedder)

    hits, timings = pipeline.retrieve("xylotomic")

    assert store.sparse_calls == [(cfg.config_id, "xylotomic", 7)]
    assert store.dense_calls == []
    assert embedder.query_texts == []  # sparse never embeds the query
    assert spy.calls == []
    assert timings["embed_query"] == 0.0
    assert timings["dense"] == 0.0
    assert timings["fusion"] == 0.0
    assert [hit.source for hit in hits] == ["sparse"]
    assert set(timings) == EXPECTED_TIMING_KEYS
    assert timings["total"] > 0.0


def test_hybrid_mode_fuses_both_lists_and_labels_hits_fused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(retrieval={"mode": "hybrid", "k_dense": 3, "k_sparse": 4, "rrf_k": 60})
    dense_hits = [_hit(1, 0.9, "dense"), _hit(2, 0.8, "dense")]
    sparse_hits = [_hit(2, 5.0, "sparse"), _hit(3, 3.0, "sparse")]
    store = FakeStore(dense=dense_hits, sparse=sparse_hits)
    spy = _spy_fuse(monkeypatch)
    pipeline = _pipeline(cfg, store)

    hits, timings = pipeline.retrieve("how do I configure this?")

    assert spy.calls == [[dense_hits, sparse_hits]]  # exactly one fuse of both lists
    assert timings["fusion"] > 0.0
    assert store.dense_calls == [(cfg.config_id, 3)]
    assert store.sparse_calls == [(cfg.config_id, "how do I configure this?", 4)]
    assert {hit.source for hit in hits} == {"fused"}
    # c002 appears in both lists (dense rank 2 + sparse rank 1) so it must lead the
    # fused ranking: 1/(60+2) + 1/(60+1) > 1/(60+1) (c001, dense only) > 1/(60+2) (c003)
    assert [hit.chunk.chunk_id for hit in hits] == ["c002", "c001", "c003"]
    assert set(timings) == EXPECTED_TIMING_KEYS
    assert timings["total"] > 0.0


@pytest.mark.parametrize(
    ("mode", "zero_stages"),
    [
        ("dense", ("sparse", "fusion")),
        ("sparse", ("embed_query", "dense", "fusion")),
    ],
)
def test_timings_have_exactly_six_keys_with_zeros_for_unexecuted_stages(
    mode: str,
    zero_stages: tuple[str, ...],
) -> None:
    cfg = _cfg(retrieval={"mode": mode, "k_dense": 5, "k_sparse": 5, "rrf_k": 60})
    store = FakeStore(dense=[_hit(1, 0.9, "dense")], sparse=[_hit(1, 4.0, "sparse")])
    pipeline = _pipeline(cfg, store)

    _hits, timings = pipeline.retrieve("q")

    assert set(timings) == EXPECTED_TIMING_KEYS  # exact key set, no extras
    assert timings["total"] > 0.0
    for stage in zero_stages:
        assert timings[stage] == 0.0


def test_timings_have_exactly_six_keys_in_hybrid_mode() -> None:
    cfg = _cfg(retrieval={"mode": "hybrid", "k_dense": 5, "k_sparse": 5, "rrf_k": 60})
    store = FakeStore(dense=[_hit(1, 0.9, "dense")], sparse=[_hit(2, 4.0, "sparse")])
    pipeline = _pipeline(cfg, store)

    _hits, timings = pipeline.retrieve("q")

    assert set(timings) == EXPECTED_TIMING_KEYS
    assert timings["fusion"] > 0.0
    assert timings["total"] > 0.0


# --- ruling 3: top_n truncation before rerank -------------------------------------------


def test_top_n_truncates_candidates_before_rerank_then_final_k() -> None:
    cfg = _cfg(
        retrieval={"mode": "dense", "k_dense": 10, "k_sparse": 5, "rrf_k": 60},
        rerank={"type": "none", "top_n": 4},
        final_k=2,
    )
    dense_hits = [_hit(i, 1.0 - 0.05 * i, "dense") for i in range(10)]
    store = FakeStore(dense=dense_hits)
    recorder = RecordingReranker()
    pipeline = _pipeline(cfg, store, reranker=recorder)

    hits, _timings = pipeline.retrieve("q")

    assert len(recorder.calls) == 1
    query, passed, k = recorder.calls[0]
    assert query == "q"
    assert len(passed) == 4  # 10 candidates truncated to top_n BEFORE rerank
    assert [hit.chunk.chunk_id for hit in passed] == ["c000", "c001", "c002", "c003"]
    assert k == 2  # rerank is asked for final_k
    assert len(hits) == 2
    assert [hit.chunk.chunk_id for hit in hits] == ["c000", "c001"]


# --- ruling 4: namespace == config_id ----------------------------------------------------


def test_namespace_passed_to_every_store_call_is_config_id() -> None:
    cfg = _cfg(retrieval={"mode": "hybrid", "k_dense": 3, "k_sparse": 4, "rrf_k": 60})
    store = FakeStore(dense=[_hit(1, 0.9, "dense")], sparse=[_hit(2, 4.0, "sparse")])
    pipeline = _pipeline(cfg, store)

    pipeline.retrieve("q")

    assert re.fullmatch(r"[0-9a-f]{12}", cfg.config_id)
    assert [namespace for namespace, _k in store.dense_calls] == [cfg.config_id]
    assert [ns for ns, _q, _k in store.sparse_calls] == [cfg.config_id]


# --- generate(): grounded answer from the top context_k hits (Phase 4) ------------------


class _FakeChat:
    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, str]], float]] = []

    def chat(self, messages, *, model=None, temperature=0.0, max_tokens=512):  # type: ignore[no-untyped-def]
        self.calls.append((messages, temperature))
        return "  Declare a function parameter.  "


def test_generate_uses_top_context_k_hits_and_config_temperature() -> None:
    from ragbench.generation import PROMPT_VERSION

    gen = _cfg().generation.model_copy(update={"context_k": 2, "temperature": 0.3})
    cfg = _cfg().model_copy(update={"generation": gen})
    chat = _FakeChat()
    pipeline = Pipeline(
        cfg,
        store=FakeStore(),
        embedder=FakeEmbedder(),
        reranker=RerankNone(),
        chunker=FakeChunker(),
        llm=chat,
    )
    hits = [_hit(i, 1.0 - i / 10, "dense") for i in range(4)]

    answer = pipeline.generate("what is a query parameter?", hits)

    assert answer == "Declare a function parameter."
    ((messages, temperature),) = chat.calls
    assert temperature == 0.3
    user = messages[1]["content"]
    assert "[1] chunk 0 body" in user and "[2] chunk 1 body" in user
    assert "chunk 2 body" not in user  # only context_k=2 passages
    assert "what is a query parameter?" in user
    assert PROMPT_VERSION == "gen-v1"


# --- ruling 1: build_pipeline seams and defaults -----------------------------------------


def test_build_pipeline_uses_injected_test_seams() -> None:
    cfg = _cfg()
    store = FakeStore()
    embedder = FakeEmbedder()
    reranker = RecordingReranker()
    chunker = FakeChunker()

    pipeline = build_pipeline(
        cfg, store=store, embedder=embedder, reranker=reranker, chunker=chunker
    )

    assert pipeline.cfg is cfg
    assert pipeline.store is store
    assert pipeline.embedder is embedder
    assert pipeline.reranker is reranker
    assert pipeline.chunker is chunker


def test_build_pipeline_defaults_build_real_components_offline() -> None:
    cfg = _cfg(backend="qdrant", retrieval={"mode": "dense"})

    pipeline = build_pipeline(cfg)

    assert isinstance(pipeline.store, QdrantStore)
    assert isinstance(pipeline.embedder, SentenceTransformerEmbedder)
    assert isinstance(pipeline.reranker, RerankNone)
    assert isinstance(pipeline.chunker, FixedChunker)


def test_build_pipeline_cross_encoder_backend_uses_cross_encoder_reranker() -> None:
    cfg = _cfg(rerank={"type": "cross_encoder", "model": "BAAI/bge-reranker-base"})

    pipeline = build_pipeline(cfg)

    assert isinstance(pipeline.reranker, CrossEncoderReranker)


def test_build_pipeline_passes_built_embedder_to_semantic_chunker() -> None:
    cfg = _cfg(chunker={"type": "semantic"})

    pipeline = build_pipeline(cfg)

    assert isinstance(pipeline.chunker, SemanticChunker)


# --- ruling 6: index_documents -----------------------------------------------------------


def test_index_documents_embeds_once_resets_then_adds_and_reports_stats() -> None:
    cfg = _cfg()
    store = FakeStore()
    embedder = FakeEmbedder()
    pipeline = _pipeline(cfg, store, embedder=embedder)
    documents = [
        Document(doc_id=f"doc{i}.md", title=f"Doc {i}", text=f"body number {i}", sections=[])
        for i in range(3)
    ]

    stats = index_documents(documents, pipeline)

    assert stats["docs"] == 3
    assert stats["chunks"] == 3  # FakeChunker: one chunk per document
    assert stats["embed_seconds"] > 0.0
    assert stats["index_seconds"] > 0.0
    assert store.events == ["reset", "add"]  # reset ONCE first, then a single add
    assert store.reset_calls[0][0] == cfg.config_id
    namespace, chunks, vectors = store.add_calls[0]
    assert namespace == cfg.config_id
    assert len(chunks) == 3
    assert vectors.shape == (3, 4)
    assert embedder.document_batches == [
        ["body number 0", "body number 1", "body number 2"]  # ONE batch pass
    ]


def test_index_documents_skips_dense_embedding_for_control_bm25() -> None:
    cfg = _cfg(backend="control_bm25", retrieval={"mode": "sparse", "k_sparse": 5})
    embedder = FakeEmbedder()
    store = ControlBM25Store()  # real in-process backend, offline
    pipeline = _pipeline(cfg, store, embedder=embedder)
    documents = [
        Document(
            doc_id="planted.md",
            title="Planted",
            text="the rare xylotomic term appears only in this planted document",
            sections=[],
        )
    ]

    stats = index_documents(documents, pipeline)

    assert embedder.document_batches == []  # no dense vectors for the BM25 control
    assert stats["embed_seconds"] == 0.0
    assert stats["docs"] == 1
    assert stats["chunks"] == 1
    # the index is immediately searchable through the normal retrieve path
    hits, timings = pipeline.retrieve("xylotomic")
    assert [hit.chunk.doc_id for hit in hits] == ["planted.md"]
    assert hits[0].source == "sparse"
    assert timings["embed_query"] == 0.0
    assert set(timings) == EXPECTED_TIMING_KEYS


# --- ruling 7: load_pipeline_config ------------------------------------------------------


def test_load_pipeline_config_round_trips_the_example_config() -> None:
    cfg = load_pipeline_config(EXAMPLE_CONFIG_PATH)

    assert isinstance(cfg, PipelineConfig)
    expected = PipelineConfig.model_validate(
        yaml.safe_load(EXAMPLE_CONFIG_PATH.read_text(encoding="utf-8"))
    )
    assert cfg.config_id == expected.config_id
    assert cfg.corpus == "fastapi_docs"
    assert cfg.retrieval.mode == "hybrid"
    assert cfg.backend == "pgvector"


def test_load_pipeline_config_accepts_str_path() -> None:
    cfg = load_pipeline_config(str(EXAMPLE_CONFIG_PATH))
    assert cfg.config_id == load_pipeline_config(EXAMPLE_CONFIG_PATH).config_id


def test_load_pipeline_config_invalid_content_names_the_file(tmp_path: Path) -> None:
    bad = tmp_path / "broken.yaml"
    bad.write_text(
        "corpus: x\nretrieval:\n  mode: banana\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="broken.yaml"):
        load_pipeline_config(bad)


def test_load_pipeline_config_missing_file_names_the_file(tmp_path: Path) -> None:
    missing = tmp_path / "absent.yaml"
    with pytest.raises(FileNotFoundError, match="absent.yaml"):
        load_pipeline_config(missing)
