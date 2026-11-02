"""Pipeline assembly: config-bound retrieval with per-stage timings (Task 2.9).

``build_pipeline`` wires a validated :class:`~ragbench.config.PipelineConfig` into a
:class:`Pipeline` (embedder, store by backend, reranker by rerank type, chunker by
chunker cfg); its keyword arguments are TEST SEAMS so unit tests inject fakes without
network or docker. ``Pipeline.retrieve`` runs the mode flow (ruling 3): embed query →
dense and/or sparse search → RRF fusion (hybrid only) → ``top_n`` truncation →
rerank → ``final_k``, and returns the hits together with ``timings_ms`` whose keys are
ALWAYS exactly ``embed_query, dense, sparse, fusion, rerank, total`` (ruling 2) —
stages not executed for the current mode report ``0.0`` and ``total`` is the wall time
of the whole ``retrieve`` call (its own ``perf_counter`` pair, so it may be smaller than
the sum of the parts). The namespace for every store call is ``cfg.config_id``
(ruling 4): one index per configuration, reused across runs.

``index_documents`` (ruling 6) chunks every document, embeds all chunk texts in ONE
batch, then ``reset`` + ``add`` once per namespace, returning real perf_counter stats.
Dense embedding is SKIPPED for :class:`ControlBM25Store` (ruling 10): the BM25 control
never stores vectors (its ``add`` ignores them), so ``embed_seconds`` is ``0.0`` there
and the vectors passed are a shape-valid empty-width placeholder.

``generate`` answers from the top ``generation.context_k`` hits via an OpenAI-compatible
chat client (Phase 4; injectable ``llm`` seam for tests).
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from typing import cast

import numpy as np

from ragbench.chunking import get_chunker
from ragbench.config import PipelineConfig
from ragbench.embedding.st_embedder import SentenceTransformerEmbedder
from ragbench.generation.prompt import ChatClient
from ragbench.rerank import CrossEncoderReranker, RerankNone
from ragbench.rerank.cross_encoder import DEFAULT_RERANK_MODEL
from ragbench.retrieval import fuse
from ragbench.stores import ControlBM25Store, PgvectorStore, QdrantStore
from ragbench.types import Chunker, Document, Embedder, Hit, Reranker, Store

TIMING_KEYS: tuple[str, ...] = (
    "embed_query",
    "dense",
    "sparse",
    "fusion",
    "rerank",
    "total",
)
"""The exact ``timings_ms`` key set reported by every ``retrieve`` call (ruling 2)."""


class Pipeline:
    """Config-bound retrieval pipeline: embed → search → fuse → rerank, timed per stage."""

    def __init__(
        self,
        cfg: PipelineConfig,
        *,
        store: Store,
        embedder: Embedder,
        reranker: Reranker,
        chunker: Chunker,
        llm: ChatClient | None = None,
    ) -> None:
        self.llm = llm
        self.cfg = cfg
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.chunker = chunker

    def retrieve(self, question: str) -> tuple[list[Hit], dict[str, float]]:
        """Rank chunks for ``question``; returns final hits plus ``timings_ms``.

        Mode flow (ruling 3): ``dense`` embeds the query then dense-searches;
        ``sparse`` searches with the raw question only (``embed_query`` stays
        ``0.0``); ``hybrid`` runs both lists through ``fuse`` (real ``fusion``
        timing). Candidates are truncated to ``rerank.top_n`` (min with available
        length) before the reranker is asked for ``final_k`` hits. ``Hit.source`` is
        whatever the stage assigned: ``dense``/``sparse`` single-mode labels survive
        the ``RerankNone`` passthrough, fusion yields ``fused``, and
        ``CrossEncoderReranker`` overwrites with ``reranked``.
        """
        timings: dict[str, float] = dict.fromkeys(TIMING_KEYS, 0.0)
        total_start = time.perf_counter()
        namespace = self.cfg.config_id
        mode = self.cfg.retrieval.mode

        dense_hits: list[Hit] = []
        sparse_hits: list[Hit] = []
        if mode in ("dense", "hybrid"):
            start = time.perf_counter()
            query_vector = self.embedder.embed_query(question)
            timings["embed_query"] = (time.perf_counter() - start) * 1000.0
            start = time.perf_counter()
            dense_hits = self.store.dense_search(
                namespace, query_vector, self.cfg.retrieval.k_dense
            )
            timings["dense"] = (time.perf_counter() - start) * 1000.0
        if mode in ("sparse", "hybrid"):
            start = time.perf_counter()
            sparse_hits = self.store.sparse_search(namespace, question, self.cfg.retrieval.k_sparse)
            timings["sparse"] = (time.perf_counter() - start) * 1000.0

        if mode == "hybrid":
            start = time.perf_counter()
            candidates = fuse([dense_hits, sparse_hits], self.cfg.retrieval.rrf_k)
            timings["fusion"] = (time.perf_counter() - start) * 1000.0
        elif mode == "dense":
            candidates = dense_hits
        else:
            candidates = sparse_hits

        truncated = candidates[: min(self.cfg.rerank.top_n, len(candidates))]
        start = time.perf_counter()
        hits = self.reranker.rerank(question, truncated, self.cfg.final_k)
        timings["rerank"] = (time.perf_counter() - start) * 1000.0
        timings["total"] = (time.perf_counter() - total_start) * 1000.0
        return hits, timings

    def generate(self, question: str, contexts: Sequence[Hit]) -> str:
        """Answer ``question`` from the top ``generation.context_k`` hits (Phase 4).

        Uses the injected ``llm`` chat client, or lazily builds an
        :class:`~ragbench.llm_client.OpenAICompatClient` from the ``LLM_*`` env vars.
        Prompt: :mod:`ragbench.generation.prompt` (versioned by ``PROMPT_VERSION``).
        """
        from ragbench.generation import generate_answer

        if self.llm is None:
            from ragbench.llm_client import OpenAICompatClient, settings_from_env

            self.llm = OpenAICompatClient(settings_from_env())
        texts = [hit.chunk.text for hit in contexts[: self.cfg.generation.context_k]]
        return generate_answer(
            self.llm, question, texts, temperature=self.cfg.generation.temperature
        )


def build_pipeline(
    cfg: PipelineConfig,
    *,
    store: Store | None = None,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    chunker: Chunker | None = None,
) -> Pipeline:
    """Assemble a :class:`Pipeline` from ``cfg``.

    Every keyword argument is a TEST SEAM: unit tests inject fakes here instead of
    touching real backends. Defaults (ruling 1): embedder
    ``SentenceTransformerEmbedder(cfg.embedder.name)`` (constructed offline — the model
    loads lazily on first embed), store chosen by ``cfg.backend``, reranker by
    ``cfg.rerank.type``, chunker via ``get_chunker(cfg.chunker, embedder)`` so a
    ``semantic`` chunker receives the built embedder.
    """
    # SentenceTransformerEmbedder.dim is a lazy read-only property while the Embedder
    # protocol declares a settable dim (the protocol shape is locked by Task 2.1's
    # test_types), so mypy needs a cast here; runtime conformance is complete.
    built_embedder = (
        embedder
        if embedder is not None
        else cast(Embedder, SentenceTransformerEmbedder(cfg.embedder.name))
    )
    built_store = store
    if built_store is None:
        if cfg.backend == "pgvector":
            built_store = PgvectorStore()
        elif cfg.backend == "qdrant":
            built_store = QdrantStore()
        else:
            built_store = ControlBM25Store()
    built_reranker = reranker
    if built_reranker is None:
        if cfg.rerank.type == "none":
            built_reranker = RerankNone()
        else:
            built_reranker = CrossEncoderReranker(cfg.rerank.model or DEFAULT_RERANK_MODEL)
    built_chunker = chunker if chunker is not None else get_chunker(cfg.chunker, built_embedder)
    return Pipeline(
        cfg,
        store=built_store,
        embedder=built_embedder,
        reranker=built_reranker,
        chunker=built_chunker,
    )


def index_documents(documents: Iterable[Document], pipeline: Pipeline) -> dict[str, int | float]:
    """Chunk and index ``documents`` into ``pipeline``'s config-id namespace (ruling 6).

    Order: chunk every document → embed ALL chunk texts in ONE batch pass →
    ``store.reset(namespace, dim)`` once → a single ``store.add``. Dense embedding is
    skipped entirely for :class:`ControlBM25Store` (documented optimization, ruling 10):
    the BM25 control ignores vectors, so no model is touched and ``embed_seconds`` is
    ``0.0`` (chunk counts are still reported). Returns stats with real perf_counter
    timings: ``{"docs", "chunks", "embed_seconds", "index_seconds"}``.
    """
    docs = list(documents)
    namespace = pipeline.cfg.config_id
    chunks = [chunk for doc in docs for chunk in pipeline.chunker.chunk(doc)]

    if isinstance(pipeline.store, ControlBM25Store):
        # BM25 stores no vectors: skip the dense embed (ruling 10). The empty-width
        # placeholder stays shape-valid (2-D, one row per chunk) for store.add checks,
        # and embed_seconds is exactly 0.0 because no embedding work ran.
        vectors = np.zeros((len(chunks), 0), dtype=np.float32)
        dim = 0
        embed_seconds = 0.0
    else:
        embed_start = time.perf_counter()
        vectors = pipeline.embedder.embed_documents([chunk.text for chunk in chunks])
        dim = int(pipeline.embedder.dim)
        embed_seconds = time.perf_counter() - embed_start

    index_start = time.perf_counter()
    pipeline.store.reset(namespace, dim)
    pipeline.store.add(namespace, chunks, vectors)
    index_seconds = time.perf_counter() - index_start

    return {
        "docs": len(docs),
        "chunks": len(chunks),
        "embed_seconds": embed_seconds,
        "index_seconds": index_seconds,
    }
