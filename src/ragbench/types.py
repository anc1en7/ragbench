"""Core type contracts (spec section 4, implemented exactly as specified)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Document:
    doc_id: str  # stable, e.g. relative path
    title: str
    text: str  # cleaned markdown
    sections: list[Section]


@dataclass(frozen=True)
class Section:
    section_id: str  # f"{doc_id}#{slug}"
    heading_path: str
    start: int  # char offsets in Document.text
    end: int


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # hash of (doc_id, start, end, chunker_config)
    doc_id: str
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    source: str  # "dense" | "sparse" | "fused" | "reranked"


class Chunker(Protocol):
    def chunk(self, doc: Document) -> list[Chunk]: ...


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class Store(Protocol):
    def reset(self, namespace: str, dim: int) -> None: ...

    def add(self, namespace: str, chunks: Sequence[Chunk], vectors: np.ndarray) -> None: ...

    def dense_search(self, namespace: str, qvec: np.ndarray, k: int) -> list[Hit]: ...

    def sparse_search(self, namespace: str, query: str, k: int) -> list[Hit]: ...

    def size_bytes(self, namespace: str) -> int: ...


class Reranker(Protocol):
    def rerank(self, query: str, hits: Sequence[Hit], k: int) -> list[Hit]: ...
