"""Spec section 4 contract tests for ragbench.types (tests-only; types.py was frozen in Phase 1).

The chunk_id tests are the spec-by-test for make_chunk_id() (Task 2.3, src/ragbench/chunking):
production code must satisfy the same equal-inputs => equal-id and differing-config =>
differing-id rule that the local _chunk_id() helper below encodes.
"""

import hashlib
import re
from dataclasses import FrozenInstanceError, fields
from typing import Protocol

import pytest

from ragbench.types import Chunk, Chunker, Document, Embedder, Hit, Reranker, Section, Store


def _chunk_id(doc_id: str, start: int, end: int, chunker_config: dict[str, int]) -> str:
    """Stable sha256 over (doc_id, start, end, chunker_config).

    Encoding: sha256 hex digest of ``f"{doc_id}|{start}|{end}|{items}"`` where ``items`` is
    the sorted ``key=value`` pairs of ``chunker_config`` joined by ``,``, so the digest does
    not depend on dict insertion order.
    """
    items = ",".join(f"{key}={value}" for key, value in sorted(chunker_config.items()))
    payload = f"{doc_id}|{start}|{end}|{items}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _make_chunk(doc_id: str, start: int, end: int, chunker_config: dict[str, int]) -> Chunk:
    return Chunk(
        chunk_id=_chunk_id(doc_id, start, end, chunker_config),
        doc_id=doc_id,
        start=start,
        end=end,
        text="chunk body",
    )


def _assert_frozen(instance, **replacements: object) -> None:
    for name, value in replacements.items():
        with pytest.raises(FrozenInstanceError):
            setattr(instance, name, value)


def test_dataclass_fields_match_spec_section_4():
    expected = {
        Document: ["doc_id", "title", "text", "sections"],
        Section: ["section_id", "heading_path", "start", "end"],
        # metadata is the sanctioned addition (Task 2.3 controller ruling 3): last field,
        # default_factory=dict, never hashed into chunk_id.
        Chunk: ["chunk_id", "doc_id", "start", "end", "text", "metadata"],
        Hit: ["chunk", "score", "source"],
    }
    for cls, names in expected.items():
        assert [field.name for field in fields(cls)] == names, cls.__name__


def test_document_rejects_mutation():
    doc = Document(doc_id="docs/a.md", title="Alpha", text="hello", sections=[])
    _assert_frozen(doc, doc_id="docs/b.md", title="Beta", text="world", sections=[])


def test_section_rejects_mutation():
    section = Section(section_id="docs/a.md#alpha", heading_path="Alpha", start=0, end=5)
    _assert_frozen(section, section_id="docs/b.md#alpha", heading_path="Beta", start=1, end=6)


def test_chunk_rejects_mutation():
    chunk = _make_chunk("docs/a.md", 0, 5, {"window": 512})
    _assert_frozen(
        chunk,
        chunk_id="f" * 64,
        doc_id="docs/b.md",
        start=1,
        end=6,
        text="changed",
    )


def test_hit_rejects_mutation():
    other = _make_chunk("docs/b.md", 10, 20, {"window": 512})
    hit = Hit(chunk=_make_chunk("docs/a.md", 0, 5, {"window": 512}), score=0.5, source="dense")
    _assert_frozen(hit, chunk=other, score=0.9, source="fused")


def test_chunk_id_stable_across_constructions_with_equal_inputs():
    config = {"window": 512, "overlap": 64}
    first = _make_chunk("docs/a.md", 0, 100, config)
    # Same inputs; config built in a different insertion order must not change the id.
    second = _make_chunk("docs/a.md", 0, 100, {"overlap": 64, "window": 512})
    assert first.chunk_id == second.chunk_id
    assert re.fullmatch(r"[0-9a-f]{64}", first.chunk_id)


def test_chunk_id_differs_when_chunker_config_differs():
    base = _make_chunk("docs/a.md", 0, 100, {"window": 512, "overlap": 64})
    changed = _make_chunk("docs/a.md", 0, 100, {"window": 256, "overlap": 64})
    assert base.chunk_id != changed.chunk_id


def test_chunk_id_differs_when_doc_or_span_differs():
    base = _make_chunk("docs/a.md", 0, 100, {"window": 512})
    other_doc = _make_chunk("docs/b.md", 0, 100, {"window": 512})
    other_span = _make_chunk("docs/a.md", 0, 64, {"window": 512})
    assert base.chunk_id != other_doc.chunk_id
    assert base.chunk_id != other_span.chunk_id


def test_protocols_are_importable_protocol_classes():
    for proto in (Chunker, Embedder, Store, Reranker):
        assert issubclass(proto, Protocol)


def test_protocol_surface_matches_spec_section_4():
    assert hasattr(Chunker, "chunk")
    assert hasattr(Embedder, "embed_documents")
    assert hasattr(Embedder, "embed_query")
    assert set(getattr(Embedder, "__annotations__", {})) == {"name", "dim"}
    for method in ("reset", "add", "dense_search", "sparse_search", "size_bytes"):
        assert hasattr(Store, method), method
    assert hasattr(Reranker, "rerank")
