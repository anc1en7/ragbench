"""Integration tests for QdrantStore (Task 2.6, controller ruling 9).

Requires the docker compose stack (``make up``): when Qdrant is unreachable the
suite fails explicitly rather than skipping. All behavior is exercised through the
public ``Store`` surface — no internal poking.
"""

from __future__ import annotations

from collections.abc import Iterator
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import pytest

from ragbench.stores.qdrant_store import DEFAULT_QDRANT_URL, QdrantStore
from ragbench.types import Chunk

pytestmark = pytest.mark.integration

DIM = 8
NAMESPACE = "it_qdrant"
N_CHUNKS = 100


def _qdrant_reachable() -> bool:
    try:
        with urlopen(f"{DEFAULT_QDRANT_URL}/collections", timeout=3) as resp:
            return resp.status == 200
    except (URLError, TimeoutError, OSError):
        return False


@pytest.fixture(autouse=True)
def _require_qdrant() -> None:
    if not _qdrant_reachable():
        pytest.fail("Qdrant unreachable — run `make up` first", pytrace=False)


@pytest.fixture()
def store() -> Iterator[QdrantStore]:
    qdrant_store = QdrantStore(url=DEFAULT_QDRANT_URL)
    try:
        yield qdrant_store
    finally:
        qdrant_store.close()


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


def test_reset_creates_collection_and_is_idempotent(store: QdrantStore) -> None:
    store.reset(NAMESPACE, DIM)
    store.reset(NAMESPACE, DIM)  # second reset drops + recreates without raising
    store.add(NAMESPACE, _chunks(), _vectors())
    assert store.size_bytes(NAMESPACE) > 0
    assert store.size_bytes("never_reset") == 0


def test_add_then_dense_search_returns_planted_neighbor_order(store: QdrantStore) -> None:
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


def test_sparse_search_finds_the_keyword_document(store: QdrantStore) -> None:
    store.reset(NAMESPACE, DIM)
    store.add(NAMESPACE, _chunks(), _vectors())
    # sparse vectors are stored by add — this first search needs no pre-backfill step

    hits = store.sparse_search(NAMESPACE, "xylotomic", k=5)
    assert [hit.chunk.chunk_id for hit in hits] == ["c007"]
    assert hits[0].source == "sparse"
    assert hits[0].score > 0.0
    assert hits[0].chunk.metadata == {}

    # stopword-only / blank queries return [] rather than raising (parity ruling 8)
    assert store.sparse_search(NAMESPACE, "the and of", k=5) == []
    assert store.sparse_search(NAMESPACE, "   ", k=5) == []


def test_sparse_search_sees_later_adds_from_another_instance(store: QdrantStore) -> None:
    """Chunks added after another store instance has searched must still be found.

    Locks the encode-at-add fix: sparse vectors are built at add time, so a later
    add is immediately visible to every instance. The old lazy-backfill design
    silently missed such chunks once an instance had already backfilled.
    """
    store.reset(NAMESPACE, DIM)
    store.add(NAMESPACE, _chunks()[:1], _vectors()[:1])  # no "xylotomic" term

    other = QdrantStore(url=DEFAULT_QDRANT_URL)
    try:
        assert other.sparse_search(NAMESPACE, "xylotomic", k=5) == []
        late = [
            Chunk(
                chunk_id="c_late",
                doc_id="d_late",
                start=0,
                end=44,
                text="the rare xylotomic term appears only in chunk seven",
            )
        ]
        store.add(NAMESPACE, late, _vectors()[7:8])
        hits = other.sparse_search(NAMESPACE, "xylotomic", k=5)
    finally:
        other.close()
    assert [hit.chunk.chunk_id for hit in hits] == ["c_late"]


def test_add_validates_chunk_count_and_vector_dim(store: QdrantStore) -> None:
    store.reset(NAMESPACE, DIM)
    chunks = _chunks()[:2]
    vectors = _vectors()[:2]

    with pytest.raises(ValueError):
        store.add(NAMESPACE, chunks, vectors[:1])  # 2 chunks vs 1 vector row
    with pytest.raises(ValueError):
        store.add(NAMESPACE, chunks, np.zeros((2, DIM + 1), dtype=np.float32))  # wrong dim
    # missing namespace raises a clear error naming reset() (parity ruling 5)
    with pytest.raises(ValueError, match="reset"):
        store.add("never_reset", chunks, vectors)


def test_absent_namespace_returns_empty_and_zero(store: QdrantStore) -> None:
    query = np.zeros(DIM, dtype=np.float32)
    query[0] = 1.0
    assert store.dense_search("never_reset", query, k=5) == []
    assert store.sparse_search("never_reset", "xylotomic", k=5) == []
    assert store.size_bytes("never_reset") == 0


def test_close_cleans_up_and_is_idempotent() -> None:
    qdrant_store = QdrantStore(url=DEFAULT_QDRANT_URL)
    qdrant_store.reset(NAMESPACE, DIM)
    qdrant_store.add(NAMESPACE, _chunks()[:2], _vectors()[:2])
    qdrant_store.close()
    qdrant_store.close()  # second close is a no-op
    assert qdrant_store.size_bytes(NAMESPACE) > 0  # lazy reconnect on next use
    qdrant_store.close()
