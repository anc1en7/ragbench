"""Passthrough reranker: min-max normalizes scores, keeps order and source (Task 2.8, ruling 2).

Why ``source`` is NOT rewritten to ``"reranked"``: no reordering model ran — this is a
score-rescaling passthrough used as the baseline rerank mode, so the hits keep the
source their backend assigned (``"dense"``/``"sparse"``/``"fused"``). Normalization
uses the min and max over ALL given hits (before truncation): ``(s - min) / (max - min)``,
with every score becoming ``1.0`` when ``max == min`` (including a single hit). Input
order is preserved (stable — equal scores keep input order), output is truncated to the
first ``k``, and new ``Hit`` objects are constructed (inputs are never mutated).
"""

from __future__ import annotations

from collections.abc import Sequence

from ragbench.types import Hit


class RerankNone:
    """Min-max-normalizing passthrough implementing :class:`~ragbench.types.Reranker`."""

    def rerank(self, query: str, hits: Sequence[Hit], k: int) -> list[Hit]:
        """Normalize scores of ``hits`` to ``[0, 1]`` and return the first ``k``.

        ``query`` is ignored (no model runs). ``k <= 0`` or no hits yields ``[]``.
        """
        del query  # unused: passthrough never consults the query
        if k <= 0 or not hits:
            return []
        scores = [hit.score for hit in hits]
        low, high = min(scores), max(scores)
        if high == low:
            normalized = [1.0] * len(hits)
        else:
            normalized = [(score - low) / (high - low) for score in scores]
        rescaled = [
            Hit(chunk=hit.chunk, score=norm, source=hit.source)
            for hit, norm in zip(hits, normalized, strict=True)
        ]
        return rescaled[:k]
