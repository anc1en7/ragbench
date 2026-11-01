"""Rerankers implementing the :class:`~ragbench.types.Reranker` protocol (Task 2.8).

``RerankNone`` is a min-max-normalizing passthrough (no model); ``CrossEncoderReranker``
scores ``(query, text)`` pairs with a sentence-transformers CrossEncoder, lazily
downloaded into the project-drive HF cache on first use.
"""

from __future__ import annotations

from ragbench.rerank.cross_encoder import CrossEncoderReranker
from ragbench.rerank.none import RerankNone

__all__ = ["CrossEncoderReranker", "RerankNone"]
