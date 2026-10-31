"""In-process BM25 control backend: sparse-only ``control_bm25`` (Task 2.7).

Spec section 5.3 purpose: "a backend-agnostic in-process BM25 (``bm25s`` or
``rank_bm25``) as ``control_bm25``. The two backends' lexical scoring differs, so this
control separates 'backend effect' from 'lexical-scorer effect'." bm25s 0.3.16 (the
installed, plan-preferred library) is used; the config side already enforces
``backend: control_bm25`` with ``retrieval.mode: sparse`` only.

Controller rulings (2-7): the constructor takes no arguments and touches no
environment variables or connections; per-namespace state (chunks + their bm25s index)
lives in a dict, and the index is REBUILT from all stored texts on every non-empty
``add`` — ruling 2 sanctions rebuild-on-add, and it is the simplest fit for bm25s's
corpus-wide ``index()`` while staying deterministic. ``reset`` accepts and ignores
``dim`` (BM25 has no vector dimension); ``add`` shape-checks but IGNORES ``vectors``
(dense vectors are meaningless here — this lets the pipeline drive all backends
uniformly). ``dense_search`` raises :class:`NotImplementedError` (sparse-only).
``sparse_search`` ranks by raw bm25s score descending, ties broken by ``chunk_id``
ascending, ``Hit.source = "sparse"``, and returns ``[]`` for blank queries, ``k <= 0``,
absent namespaces and non-matching queries (parity with the pgvector/qdrant FTS paths).
``size_bytes`` is a documented approximation (sum of the UTF-8 byte length of stored
chunk texts; the in-process bm25s index overhead is excluded). ``close()`` clears all
namespace state and is idempotent.

Verified bm25s 0.3.16 facts (task-2.7 report): ``tokenize(..., return_ids=False)``
returns nested token lists (``[0]`` selects the single query's tokens) with
``lower=True`` and token pattern ``(?u)\\b\\w\\w+\\b`` (punctuation stripped, case
folded, single characters dropped) plus the bundled English stopwords — all in-process,
no network; ``get_scores(list[str])`` returns a ``(n_docs,)`` float32 ndarray;
``batch_get_scores`` does not exist in 0.3.16; an empty token list makes
``get_scores`` raise ``IndexError``, so empty-token queries are guarded to ``[]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import bm25s
import numpy as np

from ragbench.types import Chunk, Hit


@dataclass
class _NamespaceState:
    """Per-namespace chunks plus their bm25s index (``None`` until first non-empty add)."""

    chunks: list[Chunk] = field(default_factory=list)
    retriever: bm25s.BM25 | None = None


class ControlBM25Store:
    """In-process bm25s implementation of the :class:`~ragbench.types.Store` protocol.

    Sparse-only control backend (spec section 5.3): no env vars, no connections, no
    vector storage. ``close()`` is not part of the ``Store`` protocol.
    """

    def __init__(self) -> None:
        self._namespaces: dict[str, _NamespaceState] = {}

    def close(self) -> None:
        """Clear all namespace state (idempotent; not part of ``Store``)."""
        self._namespaces.clear()

    def reset(self, namespace: str, dim: int) -> None:
        """Create the namespace empty, discarding any previous state (idempotent).

        ``dim`` is accepted and deliberately ignored: BM25 scores tokens, so there is
        no vector dimension to store or validate.
        """
        self._namespaces[namespace] = _NamespaceState()

    def add(self, namespace: str, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        """Append chunks and rebuild the namespace's bm25s index from all stored texts.

        ``vectors`` is shape-checked like every other store (2-D, one row per chunk)
        but its values are ignored: dense vectors are meaningless for BM25, so the
        pipeline can drive this backend without embedding. ``dim`` columns are
        unconstrained because ``reset`` ignores the dimension.
        """
        state = self._namespaces.get(namespace)
        if state is None:
            raise ValueError(f"namespace {namespace!r} does not exist — call reset() first")
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2-D (n, dim); got shape {vectors.shape}")
        if vectors.shape[0] != len(chunks):
            raise ValueError(f"len(chunks)={len(chunks)} != vectors.shape[0]={vectors.shape[0]}")
        if not chunks:
            return
        state.chunks.extend(chunks)
        # Rebuild-on-add (ruling 2): bm25s fits corpus-wide statistics in index(), so a
        # fresh index over every stored text is the simplest deterministic choice.
        corpus_tokens = bm25s.tokenize(
            [chunk.text for chunk in state.chunks],
            show_progress=False,
        )
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens, show_progress=False)
        state.retriever = retriever

    def dense_search(self, namespace: str, qvec: np.ndarray, k: int) -> list[Hit]:
        """Always raises: ``control_bm25`` is a sparse-only backend (spec section 5.3).

        BM25 has no dense vector space, so there is nothing meaningful to search; the
        pipeline drives every backend through the ``Store`` protocol uniformly.
        """
        raise NotImplementedError(
            "control_bm25 is a sparse-only backend (spec section 5.3); "
            "dense_search is not supported"
        )

    def sparse_search(self, namespace: str, query: str, k: int) -> list[Hit]:
        """Top-k BM25 matches: raw bm25s ``get_scores`` descending, ties by ``chunk_id``.

        Blank queries, ``k <= 0``, absent or still-empty namespaces, and queries whose
        terms match no stored chunk all yield ``[]`` (parity with the pgvector/qdrant
        FTS paths, which likewise return only matching rows/points — never an error).
        """
        if k <= 0 or not query.strip():
            return []
        state = self._namespaces.get(namespace)
        if state is None or state.retriever is None:
            return []
        tokens = bm25s.tokenize(query, return_ids=False, show_progress=False)[0]
        if not tokens:
            return []
        scores = state.retriever.get_scores(tokens)
        ranked = sorted(
            (
                (float(score), chunk)
                for score, chunk in zip(scores, state.chunks, strict=True)
                if score > 0
            ),
            key=lambda pair: (-pair[0], pair[1].chunk_id),
        )
        return [Hit(chunk=chunk, score=score, source="sparse") for score, chunk in ranked[:k]]

    def size_bytes(self, namespace: str) -> int:
        """Documented in-process approximation: summed UTF-8 bytes of stored chunk texts.

        Excludes the bm25s index's own in-memory overhead (token ids, BM25 tables) and
        the per-namespace Python object graph — there is no on-disk footprint to
        measure. Absent namespaces report 0 (parity with the other stores).
        """
        state = self._namespaces.get(namespace)
        if state is None:
            return 0
        return sum(len(chunk.text.encode("utf-8")) for chunk in state.chunks)
