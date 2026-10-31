"""Unit tests for the in-process control_bm25 Store backend (Task 2.7).

Pure unit tests: no Docker, no network. bm25s tokenization/indexing is fully
in-process — the English stopwords list ships as Python constants inside the
bm25s package (``bm25s.stopwords``), so nothing is downloaded at first use.
"""

import numpy as np
import pytest

from ragbench.stores.control_bm25 import ControlBM25Store
from ragbench.types import Chunk

NAMESPACE = "control"

TOY_DOCS = [
    ("c1", "the quick brown fox jumps over the lazy dog"),
    ("c2", "machine learning models require large climate datasets"),
    ("c3", "photosynthesis converts sunlight into chemical energy"),
    ("c4", "the ocean currents regulate global climate patterns"),
    ("c5", "quantum computers solve problems with qubits"),
]


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=f"{chunk_id}.md",
        start=0,
        end=len(text),
        text=text,
    )


def _vectors(n: int, dim: int = 4, fill: float = 0.0) -> np.ndarray:
    return np.full((n, dim), fill, dtype=np.float32)


def _indexed_store() -> ControlBM25Store:
    store = ControlBM25Store()
    store.reset(NAMESPACE, dim=8)
    chunks = [_chunk(chunk_id, text) for chunk_id, text in TOY_DOCS]
    store.add(NAMESPACE, chunks, _vectors(len(chunks)))
    return store


def test_term_containing_doc_ranks_first():
    store = _indexed_store()
    hits = store.sparse_search(NAMESPACE, "photosynthesis", k=5)
    assert hits
    assert hits[0].chunk.chunk_id == "c3"
    assert hits[0].source == "sparse"
    assert isinstance(hits[0].score, float)


def test_multi_match_results_ordered_by_descending_score():
    store = _indexed_store()
    hits = store.sparse_search(NAMESPACE, "climate", k=5)
    assert [hit.chunk.chunk_id for hit in hits] == ["c4", "c2"]
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)
    assert len(hits) <= 5


def test_repeated_search_returns_identical_order_and_scores():
    store = _indexed_store()
    first = [
        (hit.chunk.chunk_id, hit.score) for hit in store.sparse_search(NAMESPACE, "climate", k=5)
    ]
    second = [
        (hit.chunk.chunk_id, hit.score) for hit in store.sparse_search(NAMESPACE, "climate", k=5)
    ]
    assert first
    assert first == second


def test_rebuild_and_requery_returns_identical_results():
    store = _indexed_store()
    rebuilt = _indexed_store()
    first = [
        (hit.chunk.chunk_id, hit.score) for hit in store.sparse_search(NAMESPACE, "climate", k=5)
    ]
    again = [
        (hit.chunk.chunk_id, hit.score) for hit in rebuilt.sparse_search(NAMESPACE, "climate", k=5)
    ]
    assert first
    assert first == again


def test_query_matching_nothing_returns_empty():
    store = _indexed_store()
    assert store.sparse_search(NAMESPACE, "zzzznotinterm", k=5) == []


def test_add_ignores_vector_values():
    garbage = ControlBM25Store()
    garbage.reset(NAMESPACE, dim=8)
    chunks = [_chunk(chunk_id, text) for chunk_id, text in TOY_DOCS]
    garbage.add(NAMESPACE, chunks, _vectors(len(chunks), fill=42.0))
    baseline = _indexed_store()
    garbage_hits = [
        (hit.chunk.chunk_id, hit.score) for hit in garbage.sparse_search(NAMESPACE, "climate", k=5)
    ]
    baseline_hits = [
        (hit.chunk.chunk_id, hit.score) for hit in baseline.sparse_search(NAMESPACE, "climate", k=5)
    ]
    assert garbage_hits == baseline_hits


def test_add_validates_vector_shape_but_not_values():
    store = _indexed_store()
    with pytest.raises(ValueError, match=r"vectors must be 2-D"):
        store.add(NAMESPACE, [_chunk("c6", "more text")], np.zeros(4, dtype=np.float32))
    with pytest.raises(ValueError, match=r"len\(chunks\)=2 != vectors\.shape\[0\]=1"):
        store.add(
            NAMESPACE,
            [_chunk("c6", "more text"), _chunk("c7", "even more text")],
            _vectors(1),
        )


def test_add_missing_namespace_raises_naming_reset():
    store = ControlBM25Store()
    with pytest.raises(ValueError, match=r"does not exist — call reset\(\) first"):
        store.add("ghost", [_chunk("g1", "orphan text")], _vectors(1))


def test_reset_creates_and_clears_namespace_ignoring_dim():
    store = _indexed_store()
    assert store.sparse_search(NAMESPACE, "photosynthesis", k=5)
    # dim is accepted and ignored: BM25 has no vector dimension.
    store.reset(NAMESPACE, dim=0)
    assert store.sparse_search(NAMESPACE, "photosynthesis", k=5) == []
    assert store.size_bytes(NAMESPACE) == 0
    store.add(NAMESPACE, [_chunk("c3", TOY_DOCS[2][1])], _vectors(1))
    assert store.sparse_search(NAMESPACE, "photosynthesis", k=5)


def test_dense_search_raises_not_implemented_naming_control_bm25():
    store = ControlBM25Store()
    with pytest.raises(NotImplementedError) as excinfo:
        store.dense_search(NAMESPACE, np.zeros(4, dtype=np.float32), k=5)
    message = str(excinfo.value)
    assert "control_bm25" in message
    assert "sparse" in message


def test_empty_and_whitespace_queries_return_empty():
    store = _indexed_store()
    assert store.sparse_search(NAMESPACE, "", k=5) == []
    assert store.sparse_search(NAMESPACE, "   \t ", k=5) == []


def test_non_positive_k_returns_empty():
    store = _indexed_store()
    assert store.sparse_search(NAMESPACE, "climate", k=0) == []
    assert store.sparse_search(NAMESPACE, "climate", k=-3) == []


def test_absent_namespace_search_size_and_add_parity():
    store = ControlBM25Store()
    assert store.sparse_search("nope", "climate", k=5) == []
    assert store.size_bytes("nope") == 0
    with pytest.raises(ValueError, match=r"does not exist — call reset\(\) first"):
        store.add("nope", [_chunk("c1", "text")], _vectors(1))


def test_size_bytes_is_sum_of_utf8_text_bytes():
    store = ControlBM25Store()
    assert store.size_bytes(NAMESPACE) == 0
    store.reset(NAMESPACE, dim=8)
    assert store.size_bytes(NAMESPACE) == 0
    chunks = [_chunk("c1", "hello world"), _chunk("c2", "café")]
    store.add(NAMESPACE, chunks, _vectors(2))
    expected = sum(len(chunk.text.encode("utf-8")) for chunk in chunks)
    assert store.size_bytes(NAMESPACE) == expected
    assert store.size_bytes(NAMESPACE) > 0


def test_close_clears_all_namespace_state_and_is_idempotent():
    store = _indexed_store()
    store.reset("second", dim=4)
    store.add("second", [_chunk("s1", "photosynthesis again")], _vectors(1))
    store.close()
    assert store.size_bytes(NAMESPACE) == 0
    assert store.size_bytes("second") == 0
    assert store.sparse_search(NAMESPACE, "photosynthesis", k=5) == []
    assert store.sparse_search("second", "photosynthesis", k=5) == []
    store.close()  # idempotent
    with pytest.raises(ValueError, match=r"does not exist — call reset\(\) first"):
        store.add(NAMESPACE, [_chunk("c3", "text")], _vectors(1))


def test_equal_scores_break_ties_by_chunk_id_ascending():
    store = ControlBM25Store()
    store.reset(NAMESPACE, dim=8)
    chunks = [
        _chunk("zeta", "identical corpus text here"),
        _chunk("alpha", "identical corpus text here"),
    ]
    store.add(NAMESPACE, chunks, _vectors(2))
    hits = store.sparse_search(NAMESPACE, "identical corpus", k=2)
    assert [hit.chunk.chunk_id for hit in hits] == ["alpha", "zeta"]
    assert hits[0].score == hits[1].score
