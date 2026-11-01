"""Cross-encoder reranker with lazy model load and a test seam (Task 2.8, ruling 3).

Scores ``[(query, hit.chunk.text) for hit in hits]`` with a sentence-transformers
``CrossEncoder`` (default ``BAAI/bge-reranker-base``), sorts by score DESC with
``chunk_id`` ASC tie-break (stable on full ties), takes the top ``k`` and stamps
``Hit.source = "reranked"`` with the RAW model scores (no normalization). The pipeline
truncates its fused candidates to ``top_n`` before calling (spec 5.4) — this class's
``k`` IS ``final_k`` and returns at most ``k`` hits.

The constructor's ``model`` argument is a TEST SEAM: when provided it is used directly
(no loading — it must expose ``predict(pairs) -> list[float]``-compatible behavior);
when ``None`` the real ``CrossEncoder(model_name, cache_folder=...)`` is created lazily
on the first non-trivial ``rerank`` (``k <= 0``/empty inputs return ``[]`` WITHOUT
loading). The default ``cache_folder`` is ``hf_cache_root()`` — model downloads stay on
the project drive (same rule as the embedding caches).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ragbench.embedding.cache import hf_cache_root
from ragbench.types import Hit

DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"


def _load_cross_encoder(model_name: str, cache_folder: str) -> Any:
    """Create the real CrossEncoder (deferred import; download happens on first call)."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, cache_folder=cache_folder)


class CrossEncoderReranker:
    """Cross-encoder reranker implementing :class:`~ragbench.types.Reranker`."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANK_MODEL,
        *,
        cache_folder: str | Path | None = None,
        model: Any | None = None,
    ) -> None:
        self._model_name = model_name
        # Kept as given (str | Path | None) so an explicit str passes through verbatim.
        self._cache_folder = cache_folder
        self._model = model  # test seam: skip lazy loading when provided

    def rerank(self, query: str, hits: Sequence[Hit], k: int) -> list[Hit]:
        """Score every hit against ``query`` and return the top ``k`` (score DESC).

        Raw model scores, ``chunk_id`` ASC tie-break (stable on full ties),
        ``Hit.source = "reranked"``, new ``Hit`` objects. ``k <= 0`` or no hits
        returns ``[]`` without touching the model.
        """
        if k <= 0 or not hits:
            return []
        if self._model is None:
            folder = (
                str(self._cache_folder) if self._cache_folder is not None else str(hf_cache_root())
            )
            self._model = _load_cross_encoder(self._model_name, folder)
        pairs = [(query, hit.chunk.text) for hit in hits]
        scores = [float(score) for score in self._model.predict(pairs)]
        ranked = sorted(
            zip(hits, scores, strict=True),
            key=lambda pair: (-pair[1], pair[0].chunk.chunk_id),
        )
        return [Hit(chunk=hit.chunk, score=score, source="reranked") for hit, score in ranked[:k]]
