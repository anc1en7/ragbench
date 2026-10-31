"""Embedding layer: sentence-transformers embedders with a keyed disk cache (Task 2.4).

``SentenceTransformerEmbedder`` implements the :class:`~ragbench.types.Embedder`
protocol (lazy model load, ``float32`` L2-normalized vectors, prefix rules from the
verified model cards); :class:`~ragbench.embedding.cache.EmbeddingCache` stores vectors
under ``.hf_cache/embeddings/{sha}.npy`` keyed by
``(model_name, model_revision, sha256(text))``.
"""

from __future__ import annotations

from ragbench.embedding.base import document_prefix, query_prefix
from ragbench.embedding.cache import EmbeddingCache, encode_with_cache, hf_cache_root
from ragbench.embedding.st_embedder import SentenceTransformerEmbedder

__all__ = [
    "EmbeddingCache",
    "SentenceTransformerEmbedder",
    "document_prefix",
    "encode_with_cache",
    "hf_cache_root",
    "query_prefix",
]
