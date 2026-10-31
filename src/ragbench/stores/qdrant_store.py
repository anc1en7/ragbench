"""Qdrant Store backend: HNSW cosine dense search + fastembed BM25 sparse search (Task 2.6).

Controller rulings (2-8): one Qdrant collection per namespace (``chunks_<sanitized>``,
reusing pgvector's exact table-name sanitization), because HNSW needs a fixed-dim vector
per embedder while dims differ; dense vectors are ``float32`` rows with ``Cosine``
distance, HNSW ``m=16`` / ``ef_construct=64`` and search ``hnsw_ef=100`` (identical to
pgvector). Sparse search uses fastembed's ``Qdrant/bm25`` encoder with Qdrant's
``modifier="idf"`` sparse index (the Bm25 model declares ``requires_idf=True``);
scores are backend-native — cosine similarity for dense, IDF-weighted sparse relevance
for sparse — and ``Chunk`` metadata round-trips as ``{}``.

Sparse vectors materialize lazily: ``add`` stores only the dense vector plus payload
(ruling 5) and never loads the sparse model; the first ``sparse_search`` creates the
encoder (ruling 2 — its model download lands in the repo ``.hf_cache/``) and backfills
sparse vectors for points that lack them, because Qdrant sparse search only scores
points with a stored sparse vector (verified against server v1.20.0).

``size_bytes`` is a documented approximation (ruling 7): ``get_collection`` exposes no
disk-size field, so it reports ``points_count * dim * 4`` (float32 dense vectors) plus
the summed UTF-8 byte length of stored ``text`` payloads.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from ragbench.embedding.cache import hf_cache_root
from ragbench.stores.pgvector_store import _table_name
from ragbench.types import Chunk, Hit

if TYPE_CHECKING:
    from fastembed import SparseTextEmbedding

DEFAULT_QDRANT_URL = "http://localhost:6333"
"""docker-compose default: resolution order is explicit arg → ``QDRANT_URL`` env → this."""

DEFAULT_SPARSE_MODEL = "Qdrant/bm25"
"""BM25-style sparse encoder from fastembed 0.9.1's supported list (ruling 1)."""

_SPARSE_VECTOR_NAME = "sparse"
_DENSE_VECTOR_KEY = ""  # dict-form key of the single unnamed dense vector
_HNSW_EF_SEARCH = 100


def _collection_name(namespace: str) -> str:
    """Collection-per-namespace name (ruling 3): ``chunks_<sanitized>``.

    Identical to pgvector's table-name sanitization (imported, not copied): lowercase,
    every ``[^a-z0-9_]`` becomes ``_``; a segment that is empty or starts with a digit
    gets an ``n_`` prefix so the identifier can never start with a digit.
    """
    return _table_name(namespace)


def _point_id(chunk_id: str) -> int:
    """Stable unsigned point id (ruling 5): first 8 bytes of sha256(chunk_id), big-endian."""
    digest = hashlib.sha256(chunk_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _hit_from_point(point: models.ScoredPoint, source: str) -> Hit:
    """Reconstruct a :class:`Hit` from a search result (``metadata={}`` parity ruling 4)."""
    payload: dict[str, Any] = point.payload or {}
    return Hit(
        chunk=Chunk(
            chunk_id=str(payload["chunk_id"]),
            doc_id=str(payload["doc_id"]),
            start=int(payload["start"]),
            end=int(payload["end"]),
            text=str(payload["text"]),
            metadata={},
        ),
        score=float(point.score),
        source=source,
    )


class QdrantStore:
    """Qdrant implementation of the :class:`~ragbench.types.Store` protocol.

    One lazily-connected :class:`QdrantClient` per instance. ``close()`` releases the
    client and is not part of the ``Store`` protocol.
    """

    def __init__(self, url: str | None = None, *, sparse_model: str | None = None) -> None:
        self._url: str = url or os.environ.get("QDRANT_URL") or DEFAULT_QDRANT_URL
        self._sparse_model: str = sparse_model or DEFAULT_SPARSE_MODEL
        self._client: QdrantClient | None = None
        self._encoder: SparseTextEmbedding | None = None
        self._sparse_indexed: set[str] = set()

    def _connect(self) -> QdrantClient:
        if self._client is not None:
            return self._client
        try:
            client = QdrantClient(url=self._url)
            client.get_collections()  # ping so an unreachable server fails fast
        except (ResponseHandlingException, UnexpectedResponse) as exc:
            raise RuntimeError(
                f"could not connect to Qdrant (QDRANT_URL={self._url!r}); "
                "check QDRANT_URL or run `make up`"
            ) from exc
        self._client = client
        return client

    def close(self) -> None:
        """Close the underlying client (idempotent; not part of ``Store``)."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def reset(self, namespace: str, dim: int) -> None:
        """Drop and recreate the namespace collection (idempotent).

        Dense vectors: single unnamed ``VectorParams(size=dim, distance=Cosine)`` with
        collection-level HNSW ``m=16, ef_construct=64`` (ruling 3; the top-level
        ``hnsw_config`` is what ``get_collection`` reports — verified). The sparse
        vector config (ruling 8) uses ``modifier="idf"`` per the Qdrant/bm25 model card.
        """
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        name = _collection_name(namespace)
        client = self._connect()
        if client.collection_exists(name):
            client.delete_collection(name)
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=64),
            sparse_vectors_config={
                _SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
        self._sparse_indexed.discard(name)

    def add(self, namespace: str, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        """Bulk-upsert chunks with float32 dense vectors (validates count and dim).

        Point id is the first 8 bytes of ``sha256(chunk_id)`` as an unsigned int,
        collision-checked within the batch (ruling 5). Payload is exactly
        ``{chunk_id, doc_id, start, end, text, namespace}``; no sparse vector is
        stored here — it is backfilled on the first ``sparse_search`` (ruling 2).
        """
        name = _collection_name(namespace)
        client = self._connect()
        if not client.collection_exists(name):
            raise ValueError(f"namespace collection {name!r} does not exist — call reset() first")
        info = client.get_collection(name)
        vectors_cfg = info.config.params.vectors
        if vectors_cfg is None or isinstance(vectors_cfg, dict):
            raise ValueError(
                f"collection {name!r} does not expose a single unnamed dense vector "
                "config (named or absent) — recreate it with reset()"
            )
        dim = int(vectors_cfg.size)
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2-D (n, dim); got shape {vectors.shape}")
        if vectors.shape[0] != len(chunks):
            raise ValueError(f"len(chunks)={len(chunks)} != vectors.shape[0]={vectors.shape[0]}")
        if vectors.shape[1] != dim:
            raise ValueError(
                f"vectors.shape[1]={vectors.shape[1]} != namespace dim {dim} "
                f"for namespace {namespace!r}"
            )
        if not chunks:
            return
        rows = np.asarray(vectors, dtype=np.float32)
        seen: dict[int, str] = {}
        points: list[models.PointStruct] = []
        for chunk, row in zip(chunks, rows, strict=True):
            point_id = _point_id(chunk.chunk_id)
            if point_id in seen:
                raise ValueError(
                    f"point id collision within batch for chunk_ids "
                    f"{seen[point_id]!r} and {chunk.chunk_id!r}"
                )
            seen[point_id] = chunk.chunk_id
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=row.tolist(),
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "start": chunk.start,
                        "end": chunk.end,
                        "text": chunk.text,
                        "namespace": namespace,
                    },
                )
            )
        client.upsert(collection_name=name, points=points, wait=True)
        # fresh points lack sparse vectors; the next sparse_search backfills them
        self._sparse_indexed.discard(name)

    def dense_search(self, namespace: str, qvec: np.ndarray, k: int) -> list[Hit]:
        """Top-k cosine neighbors: score is Qdrant's cosine similarity, descending.

        Every HNSW query runs with ``hnsw_ef=100`` (ruling 3). Absent namespaces
        return ``[]`` (parity ruling 6).
        """
        if k <= 0:
            return []
        name = _collection_name(namespace)
        client = self._connect()
        if not client.collection_exists(name):
            return []
        response = client.query_points(
            collection_name=name,
            query=np.asarray(qvec, dtype=np.float32).tolist(),
            search_params=models.SearchParams(hnsw_ef=_HNSW_EF_SEARCH),
            limit=k,
            with_payload=True,
        )
        return [_hit_from_point(point, "dense") for point in response.points]

    def sparse_search(self, namespace: str, query: str, k: int) -> list[Hit]:
        """Top-k BM25 sparse matches (fastembed ``Qdrant/bm25`` + Qdrant IDF scoring).

        The encoder is created lazily on the first call (ruling 2 — one-time model
        download into ``.hf_cache/``); the query is encoded with it and searched as a
        Qdrant sparse vector (ruling 8). Blank queries and queries whose terms are all
        stopwords/out-of-vocabulary yield ``[]`` (parity, never an error). Absent
        namespaces return ``[]`` before any model is loaded.
        """
        if k <= 0 or not query.strip():
            return []
        name = _collection_name(namespace)
        client = self._connect()
        if not client.collection_exists(name):
            return []
        encoder = self._sparse_encoder()
        embedding = next(iter(encoder.embed([query])))
        if len(embedding.indices) == 0:
            return []
        self._ensure_sparse_vectors(client, name, encoder)
        response = client.query_points(
            collection_name=name,
            query=models.SparseVector(
                indices=[int(index) for index in embedding.indices],
                values=[float(value) for value in embedding.values],
            ),
            using=_SPARSE_VECTOR_NAME,
            limit=k,
            with_payload=True,
        )
        return [_hit_from_point(point, "sparse") for point in response.points]

    def size_bytes(self, namespace: str) -> int:
        """Documented size approximation (ruling 7): ``points_count * dim * 4`` plus
        the summed UTF-8 byte length of stored ``text`` payloads (scrolled in batches).

        The ``points_count * dim * 4`` term counts float32 dense vectors; Qdrant's
        ``get_collection`` exposes no on-disk size field (verified against
        qdrant-client 1.20 / server 1.20.0), so sparse vectors, the HNSW graph, the
        payload structure and WAL are excluded. Returns 0 for absent namespaces
        (parity ruling 6).
        """
        name = _collection_name(namespace)
        client = self._connect()
        if not client.collection_exists(name):
            return 0
        info = client.get_collection(name)
        vectors_cfg = info.config.params.vectors
        if isinstance(vectors_cfg, dict):
            dim = sum(int(params.size) for params in vectors_cfg.values())
        elif vectors_cfg is None:
            dim = 0
        else:
            dim = int(vectors_cfg.size)
        total = int(info.points_count or 0) * dim * 4
        offset: int | str | uuid.UUID | None = None
        while True:
            batch, offset = client.scroll(
                collection_name=name,
                limit=256,
                offset=offset,
                with_payload=["text"],
                with_vectors=False,
            )
            for record in batch:
                text = (record.payload or {}).get("text")
                if isinstance(text, str):
                    total += len(text.encode("utf-8"))
            if offset is None:
                break
        return total

    def _sparse_encoder(self) -> SparseTextEmbedding:
        """Lazily create the fastembed sparse encoder (first ``sparse_search`` call)."""
        if self._encoder is None:
            # Deferred import: neither fastembed nor its ONNX runtime loads until a
            # sparse search actually needs the model (ruling 2).
            from fastembed import SparseTextEmbedding

            # Model cache: FASTEMBED_CACHE_PATH env if set, else repo-root .hf_cache
            # (or $HF_HOME) — same project-drive rule as the embedding caches.
            cache_dir = os.environ.get("FASTEMBED_CACHE_PATH") or str(hf_cache_root())
            self._encoder = SparseTextEmbedding(self._sparse_model, cache_dir=cache_dir)
        return self._encoder

    def _ensure_sparse_vectors(
        self,
        client: QdrantClient,
        collection: str,
        encoder: SparseTextEmbedding,
    ) -> None:
        """Backfill sparse vectors for points that lack them (see module docstring).

        Runs once per collection until the next ``add``/``reset`` (which drops the
        flag), so repeated queries cost only the flag check; points whose text encodes
        to zero terms stay dense-only (they must not match any sparse query anyway).
        """
        if collection in self._sparse_indexed:
            return
        dense_rows: list[Any] = []
        payloads: list[dict[str, Any]] = []
        texts: list[str] = []
        offset: int | str | uuid.UUID | None = None
        while True:
            batch, offset = client.scroll(
                collection_name=collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            for record in batch:
                vector = record.vector
                if isinstance(vector, dict):
                    if _SPARSE_VECTOR_NAME in vector:
                        continue
                    dense_row = vector.get(_DENSE_VECTOR_KEY)
                else:
                    dense_row = vector
                payload = record.payload or {}
                text = payload.get("text")
                if dense_row is None or not isinstance(text, str):
                    continue
                dense_rows.append((record.id, dense_row))
                payloads.append(dict(payload))
                texts.append(text)
            if offset is None:
                break
        if dense_rows:
            points: list[models.PointStruct] = []
            for (point_id, dense_row), payload, _text, embedding in zip(
                dense_rows, payloads, texts, encoder.embed(texts), strict=True
            ):
                if len(embedding.indices) == 0:
                    continue  # no indexable terms — stays dense-only, matches nothing
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector={
                            _DENSE_VECTOR_KEY: [float(value) for value in dense_row],
                            _SPARSE_VECTOR_NAME: models.SparseVector(
                                indices=[int(index) for index in embedding.indices],
                                values=[float(value) for value in embedding.values],
                            ),
                        },
                        payload=payload,
                    )
                )
            if points:
                client.upsert(collection_name=collection, points=points, wait=True)
        self._sparse_indexed.add(collection)
