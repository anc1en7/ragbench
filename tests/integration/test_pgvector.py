"""Integration tests for PgvectorStore (Task 2.5, controller ruling 9).

Requires the docker compose stack (``make up``): when Postgres is unreachable the
suite fails explicitly rather than skipping. All SQL is exercised through the
public ``Store`` surface — no internal poking.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import psycopg
import pytest

from ragbench.stores.pgvector_store import PgvectorStore
from ragbench.types import Chunk

pytestmark = pytest.mark.integration

DSN = "postgresql://ragbench:ragbench@localhost:5432/ragbench"
DIM = 8
NAMESPACE = "it_pgvector"
N_CHUNKS = 100


def _postgres_reachable() -> bool:
    try:
        with psycopg.connect(DSN, connect_timeout=3) as conn:
            conn.execute("SELECT 1")
    except psycopg.OperationalError:
        return False
    return True


@pytest.fixture(autouse=True)
def _require_postgres() -> None:
    if not _postgres_reachable():
        pytest.fail("Postgres unreachable — run `make up` first", pytrace=False)


@pytest.fixture()
def store() -> Iterator[PgvectorStore]:
    pg_store = PgvectorStore(dsn=DSN)
    try:
        yield pg_store
    finally:
        pg_store.close()


def _chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for i in range(N_CHUNKS):
        text = f"plain retrieval body number {i}"
        if i == 7:
            text = "the rare xylotomic term appears only in chunk seven"
        chunks.append(
            Chunk(chunk_id=f"c{i:03d}", doc_id=f"d{i:03d}", start=0, end=len(text), text=text)
        )
    return chunks


def _vectors() -> np.ndarray:
    """100 deterministic vectors whose cosine similarity to e0 is cos(0.015 * i).

    The similarities are strictly decreasing in ``i``, so the expected dense
    neighbor order for query ``e0`` is exactly ``0, 1, 2, ...``.
    """
    angles = np.arange(N_CHUNKS, dtype=np.float32) * np.float32(0.015)
    vectors = np.zeros((N_CHUNKS, DIM), dtype=np.float32)
    vectors[:, 0] = np.cos(angles)
    vectors[:, 1] = np.sin(angles)
    return vectors


def test_reset_creates_schema_and_is_idempotent(store: PgvectorStore) -> None:
    store.reset(NAMESPACE, DIM)
    store.reset(NAMESPACE, DIM)  # second reset drops + recreates without raising
    assert store.size_bytes(NAMESPACE) > 0
    assert store.size_bytes("never_reset") == 0


def test_add_then_dense_search_returns_planted_neighbor_order(store: PgvectorStore) -> None:
    store.reset(NAMESPACE, DIM)
    store.add(NAMESPACE, _chunks(), _vectors())

    query = np.zeros(DIM, dtype=np.float32)
    query[0] = 1.0  # planted vector == chunk 0's vector (angle 0)
    hits = store.dense_search(NAMESPACE, query, k=10)

    assert [hit.chunk.chunk_id for hit in hits] == [f"c{i:03d}" for i in range(10)]
    assert all(hit.source == "dense" for hit in hits)
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert hits[0].chunk.metadata == {}
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)


def test_sparse_search_finds_the_keyword_document(store: PgvectorStore) -> None:
    store.reset(NAMESPACE, DIM)
    store.add(NAMESPACE, _chunks(), _vectors())

    hits = store.sparse_search(NAMESPACE, "xylotomic", k=5)
    assert [hit.chunk.chunk_id for hit in hits] == ["c007"]
    assert hits[0].source == "sparse"
    assert hits[0].score > 0.0
    assert hits[0].chunk.metadata == {}

    # stopword-only / blank queries return [] rather than raising (ruling 7)
    assert store.sparse_search(NAMESPACE, "the and of", k=5) == []
    assert store.sparse_search(NAMESPACE, "   ", k=5) == []


def test_add_validates_chunk_count_and_vector_dim(store: PgvectorStore) -> None:
    store.reset(NAMESPACE, DIM)
    chunks = _chunks()[:2]
    vectors = _vectors()[:2]

    with pytest.raises(ValueError):
        store.add(NAMESPACE, chunks, vectors[:1])  # 2 chunks vs 1 vector row
    with pytest.raises(ValueError):
        store.add(NAMESPACE, chunks, np.zeros((2, DIM + 1), dtype=np.float32))  # wrong dim


def test_close_cleans_up_and_is_idempotent() -> None:
    pg_store = PgvectorStore(dsn=DSN)
    pg_store.reset(NAMESPACE, DIM)
    pg_store.close()
    pg_store.close()  # second close is a no-op
    assert pg_store.size_bytes(NAMESPACE) > 0  # lazy reconnect on next use
    pg_store.close()
