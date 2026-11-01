"""Reciprocal Rank Fusion over backend Hit lists (Task 2.8, controller ruling 1).

``fuse`` runs in APPLICATION code (spec section 4 design rule) so pgvector, Qdrant and
the BM25 control backend are compared with identical fusion logic — it only ever sees
``Hit`` lists and is therefore backend-agnostic. A chunk's identity across lists is its
``chunk_id``; its rank within each list is the 1-based position, and it contributes
``1 / (k + rank)`` for EVERY list it appears in (absent lists contribute nothing; a
chunk repeated inside one list contributes from its FIRST occurrence's rank only).
Scores are the raw RRF sums; the output is ALL fused candidates sorted by score DESC
with ties broken by ``chunk_id`` ASC, each with ``Hit.source = "fused"``.

``k`` is the RRF smoothing constant (default 60), NOT a result count — the caller
truncates the returned list to the desired top-n.
"""

from __future__ import annotations

from ragbench.types import Chunk, Hit


def fuse(lists: list[list[Hit]], k: int = 60) -> list[Hit]:
    """Fuse ranked ``lists`` with Reciprocal Rank Fusion; returns ALL candidates.

    Rank within each list is 1-based; a chunk contributes ``1/(k + rank)`` per list it
    appears in (identity = ``chunk_id``, first occurrence's rank only when duplicated
    inside one list). ``k`` is the RRF constant, not a top-n — the caller truncates.
    Sorted by score DESC, ties by ``chunk_id`` ASC; empty input yields ``[]``.
    """
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for hits in lists:
        seen: set[str] = set()
        for position, hit in enumerate(hits, start=1):
            chunk_id = hit.chunk.chunk_id
            if chunk_id in seen:
                continue  # first occurrence's rank only (ruling 1)
            seen.add(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + position)
            chunks.setdefault(chunk_id, hit.chunk)
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    return [Hit(chunk=chunks[cid], score=scores[cid], source="fused") for cid in ordered]
