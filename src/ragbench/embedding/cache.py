"""Keyed on-disk cache for embedding vectors (Task 2.4, rulings 2, 4, 5).

Cache key = ``(model_name, model_revision, sha256(text))``; the file name is the sha256
of the canonical key encoding ``f"{model_name}|{model_revision}|{text_sha}"``. Files live
at ``<root>/{sha}.npy`` where ``<root>`` defaults to repo-root ``.hf_cache/embeddings``
(``$HF_HOME/embeddings`` when ``HF_HOME`` is set — same project-drive rule as model
downloads). The stored vector is the embedding of the PREFIXED text, while the key uses
the raw input text you were given (ruling 4): a cache hit therefore skips prefixing and
model compute entirely. Because the prefix is not part of the key, the same raw text
embedded once as a query and once as a document shares one entry (first writer wins).
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_HF_CACHE = _REPO_ROOT / ".hf_cache"


def hf_cache_root() -> Path:
    """Model cache root: ``$HF_HOME`` when set, else repo-root ``.hf_cache``."""
    env = os.environ.get("HF_HOME")
    return Path(env) if env else _DEFAULT_HF_CACHE


def cache_key_hex(model_name: str, model_revision: str, text: str) -> str:
    """sha256 hex over the canonical encoding of ``(model_name, model_revision, sha256(text))``."""
    text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    canonical = f"{model_name}|{model_revision}|{text_sha}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """``.npy`` vector store keyed by :func:`cache_key_hex`."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = root if root is not None else hf_cache_root() / "embeddings"

    @property
    def root(self) -> Path:
        """Directory holding the ``{sha}.npy`` files."""
        return self._root

    def path_for(self, model_name: str, model_revision: str, text: str) -> Path:
        """Absolute path of the cache file for the given key components."""
        return self._root / f"{cache_key_hex(model_name, model_revision, text)}.npy"

    def get(self, model_name: str, model_revision: str, text: str) -> np.ndarray | None:
        """Cached vector for the key, or ``None`` on a miss; always ``float32``."""
        path = self.path_for(model_name, model_revision, text)
        if not path.is_file():
            return None
        return np.asarray(np.load(path), dtype=np.float32)

    def set(self, model_name: str, model_revision: str, text: str, vector: np.ndarray) -> None:
        """Persist ``vector`` (coerced to ``float32``) under the key, creating the root."""
        path = self.path_for(model_name, model_revision, text)
        self._root.mkdir(parents=True, exist_ok=True)
        np.save(path, np.asarray(vector, dtype=np.float32))


def encode_with_cache(
    texts: Sequence[str],
    *,
    cache: EmbeddingCache,
    model_name: str,
    model_revision: str,
    prefix: str,
    compute: Callable[[list[str]], np.ndarray],
) -> np.ndarray:
    """Embed ``texts`` through the cache, calling ``compute`` only for cache misses.

    Keys are the raw texts; ``prefix`` is prepended only inside the ``compute`` call, so
    hits skip everything. Misses are computed in one batch and stored individually.
    Returns a ``(len(texts), dim)`` ``float32`` array with rows in input order.
    """
    if not texts:
        raise ValueError("encode_with_cache requires at least one text")
    raw = list(texts)
    resolved: dict[int, np.ndarray] = {}
    missing: list[int] = []
    for index, text in enumerate(raw):
        hit = cache.get(model_name, model_revision, text)
        if hit is None:
            missing.append(index)
        else:
            resolved[index] = hit
    if missing:
        computed = compute([prefix + raw[index] for index in missing])
        for position, index in enumerate(missing):
            vector = np.asarray(computed[position], dtype=np.float32)
            resolved[index] = vector
            cache.set(model_name, model_revision, raw[index], vector)
    return np.stack([resolved[index] for index in range(len(raw))]).astype(np.float32, copy=False)
