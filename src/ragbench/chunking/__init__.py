"""Chunker registry and the canonical chunk-id helper (spec section 5.2, Task 2.3).

``get_chunker`` builds the chunker selected by a validated :class:`~ragbench.config.ChunkerCfg`.
Registry contract (ruling 1): every chunker derives ``chunk.chunk_id`` from
``make_chunk_id(doc_id, start, end, cfg.model_dump(mode="json"))`` — the FULL, default-filled
pydantic dump of the ChunkerCfg that built the chunker (no fields removed). Later tasks that
index into stores must serialize the config with the same ``model_dump(mode="json")`` call so
chunk ids stay stable between pipeline runs and store namespaces.

``make_chunk_id`` encoding (ruling 1): sha256 hex digest (64 chars) over the UTF-8 bytes of
``f"{doc_id}|{start}|{end}|{config}"``, where ``config`` is
``json.dumps(dict(chunker_config), sort_keys=True, separators=(",", ":"))``. Equal inputs give
equal ids; the id changes when any of the four inputs changes. Chunk ``metadata`` is never
hashed (ruling 3).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from ragbench.chunking.fixed import FixedChunker
from ragbench.chunking.recursive_md import RecursiveMarkdownChunker
from ragbench.chunking.semantic import SemanticChunker
from ragbench.chunking.tokenizer import count_tokens, get_tokenizer, token_char_offsets
from ragbench.config import ChunkerCfg
from ragbench.types import Chunker, Embedder

__all__ = [
    "FixedChunker",
    "RecursiveMarkdownChunker",
    "SemanticChunker",
    "count_tokens",
    "get_chunker",
    "get_tokenizer",
    "make_chunk_id",
    "token_char_offsets",
]


def make_chunk_id(doc_id: str, start: int, end: int, chunker_config: Mapping[str, Any]) -> str:
    """Stable sha256 hex id over ``(doc_id, start, end, chunker_config)``.

    See the module docstring for the exact canonical encoding. ``chunker_config`` is the
    full ``ChunkerCfg.model_dump(mode="json")`` of the config that built the producing
    chunker.
    """
    canonical = json.dumps(dict(chunker_config), sort_keys=True, separators=(",", ":"))
    payload = f"{doc_id}|{start}|{end}|{canonical}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_chunker(cfg: ChunkerCfg, embedder: Embedder | None = None) -> Chunker:
    """Build the chunker selected by ``cfg`` (ruling 4).

    ``semantic`` requires a non-``None`` ``embedder`` and raises ``ValueError`` naming
    ``embedder`` when it is missing; ``fixed`` and ``recursive_md`` ignore ``embedder``.
    """
    if cfg.type == "fixed":
        return FixedChunker(cfg)
    if cfg.type == "recursive_md":
        return RecursiveMarkdownChunker(cfg)
    if embedder is None:
        raise ValueError(
            "chunker type 'semantic' requires an embedder; pass a non-None embedder to get_chunker"
        )
    return SemanticChunker(cfg, embedder)
