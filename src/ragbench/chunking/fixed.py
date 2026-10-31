"""Fixed token-window chunker (spec section 5.2, ruling 7).

Windows of ``size_tokens`` cl100k tokens advance by ``size_tokens - overlap_tokens``
tokens and are mapped back to exact char offsets, so every ``chunk.text`` is an exact
``doc.text[start:end]`` slice. With ``overlap_tokens == 0`` the windows tile the document
contiguously; the last window may be shorter. Empty (token-less) documents yield no chunks
and no chunk is ever zero-length.
"""

from __future__ import annotations

from typing import Any

from ragbench import (
    chunking as _chunking,  # make_chunk_id resolved at call time (import order safe)
)
from ragbench.chunking.tokenizer import token_char_offsets
from ragbench.config import ChunkerCfg
from ragbench.types import Chunk, Document


class FixedChunker:
    """Sliding cl100k token windows over ``Document.text``."""

    def __init__(self, cfg: ChunkerCfg) -> None:
        if cfg.type != "fixed":
            raise ValueError(f"FixedChunker requires chunker type 'fixed', got '{cfg.type}'")
        if cfg.size_tokens is None or cfg.overlap_tokens is None:
            raise ValueError("FixedChunker requires size_tokens and overlap_tokens")
        if cfg.overlap_tokens >= cfg.size_tokens:
            raise ValueError(
                f"overlap_tokens ({cfg.overlap_tokens}) must be smaller than "
                f"size_tokens ({cfg.size_tokens})"
            )
        self._size = cfg.size_tokens
        self._overlap = cfg.overlap_tokens
        self._config_dump: dict[str, Any] = cfg.model_dump(mode="json")

    def chunk(self, doc: Document) -> list[Chunk]:
        offsets = token_char_offsets(doc.text)
        n_tokens = len(offsets) - 1
        if n_tokens == 0:
            return []
        step = self._size - self._overlap
        chunks: list[Chunk] = []
        first = 0
        while first < n_tokens:
            last = min(first + self._size, n_tokens)
            start, end = offsets[first], offsets[last]
            if start < end:
                chunks.append(
                    Chunk(
                        chunk_id=_chunking.make_chunk_id(doc.doc_id, start, end, self._config_dump),
                        doc_id=doc.doc_id,
                        start=start,
                        end=end,
                        text=doc.text[start:end],
                    )
                )
            if last == n_tokens:
                break
            first += step
        return chunks
