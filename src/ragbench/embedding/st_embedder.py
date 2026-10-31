"""SentenceTransformer embedder with lazy model load and a keyed disk cache (Task 2.4, rulings 3-6).

Implements the :class:`~ragbench.types.Embedder` protocol. The model loads on the first
embed call (never at construction), downloads land in ``$HF_HOME`` or repo-root
``.hf_cache`` (``cache_dir`` overrides both model and vector-cache locations), and
vectors are cached per ``(model_name, model_revision, sha256(raw_text))`` — prefixes are
applied only inside the model call, so cache hits skip prefixing and compute.

Contract (ruling 3): ``embed_documents`` returns ``(len(texts), dim)`` and
``embed_query`` returns ``(dim,)``, both ``float32`` and L2-normalized
(``normalize_embeddings=True`` for documents and queries — all three supported models
are cosine models). ``dim`` is a lazy property: available after the first embed call
(model reports ``get_sentence_embedding_dimension()``).

``model_revision`` (ruling 6) is the requested revision until the first embed call, when
it is best-effort replaced by the resolved HF snapshot sha via
``huggingface_hub.model_info`` (falls back to the requested revision, else
``"unresolved"``, when the hub is unreachable); offline unit tests never call the hub
because they only construct the object.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from huggingface_hub import model_info
from sentence_transformers import SentenceTransformer

from ragbench.embedding.base import document_prefix, query_prefix
from ragbench.embedding.cache import EmbeddingCache, encode_with_cache, hf_cache_root

_UNRESOLVED = "unresolved"


class SentenceTransformerEmbedder:
    """Lazily-loaded SentenceTransformer implementing the ``Embedder`` protocol."""

    def __init__(
        self,
        name: str,
        revision: str | None = None,
        *,
        cache_dir: Path | None = None,
    ) -> None:
        self.name = name
        self.model_revision: str = _UNRESOLVED if revision is None else revision
        self._requested_revision = revision
        self._cache_dir = cache_dir
        self._cache = EmbeddingCache(cache_dir / "embeddings" if cache_dir is not None else None)
        self._model: SentenceTransformer | None = None
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        """Embedding dimension; available only after the first embed call (lazy load)."""
        if self._dim is None:
            raise RuntimeError(
                f"dim for {self.name!r} is unavailable before the first embed call "
                "(the SentenceTransformer model loads lazily)"
            )
        return self._dim

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """``(len(texts), dim)`` float32 L2-normalized document embeddings."""
        model = self._ensure_loaded()
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        return encode_with_cache(
            list(texts),
            cache=self._cache,
            model_name=self.name,
            model_revision=self.model_revision,
            prefix=document_prefix(self.name),
            compute=lambda batch: model.encode(batch, normalize_embeddings=True),
        )

    def embed_query(self, text: str) -> np.ndarray:
        """``(dim,)`` float32 L2-normalized query embedding."""
        model = self._ensure_loaded()
        vectors = encode_with_cache(
            [text],
            cache=self._cache,
            model_name=self.name,
            model_revision=self.model_revision,
            prefix=query_prefix(self.name),
            compute=lambda batch: model.encode(batch, normalize_embeddings=True),
        )
        return np.asarray(vectors[0], dtype=np.float32)

    def _ensure_loaded(self) -> SentenceTransformer:
        """Load the model on first use, resolving and recording the HF snapshot sha."""
        if self._model is None:
            resolved = self._resolve_revision()
            if resolved is not None:
                self.model_revision = resolved
            load_revision = resolved if resolved is not None else self._requested_revision
            self._model = SentenceTransformer(
                self.name,
                revision=load_revision,
                cache_folder=str(self._hf_cache_dir()),
            )
            dimension = self._model.get_sentence_embedding_dimension()
            if dimension is None:
                raise RuntimeError(f"model {self.name!r} did not report an embedding dimension")
            self._dim = int(dimension)
        return self._model

    def _resolve_revision(self) -> str | None:
        """Best-effort HF snapshot sha for the requested revision; ``None`` when offline."""
        try:
            info = model_info(self.name, revision=self._requested_revision)
        except Exception:
            return None
        return info.sha

    def _hf_cache_dir(self) -> Path:
        """Explicit ``cache_dir`` override, else ``$HF_HOME``/repo-root ``.hf_cache``."""
        if self._cache_dir is not None:
            return self._cache_dir
        return hf_cache_root()
