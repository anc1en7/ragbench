"""Chunking contract tests: shared invariants over all three chunkers, make_chunk_id rule,
registry behavior, and chunker-specific rules (Task 2.3).

Fixtures are small synthetic Documents constructed here (never the real corpus). The semantic
chunker is driven by a deterministic fake embedder so unit tests never touch the network.
"""

import hashlib
import math
import re

import numpy as np
import pytest

from ragbench.chunking import get_chunker, make_chunk_id
from ragbench.chunking.fixed import FixedChunker
from ragbench.chunking.recursive_md import RecursiveMarkdownChunker
from ragbench.chunking.semantic import SemanticChunker
from ragbench.chunking.tokenizer import get_tokenizer
from ragbench.config import ChunkerCfg
from ragbench.corpus.sections import parse_sections
from ragbench.types import Chunk, Chunker, Document

# --------------------------------------------------------------------------- fixtures

FIXED_CFG = ChunkerCfg(type="fixed", size_tokens=16, overlap_tokens=4)
RECURSIVE_CFG = ChunkerCfg(type="recursive_md", size_tokens=16, overlap_tokens=4)
SEMANTIC_CFG = ChunkerCfg(type="semantic", min_tokens=8, max_tokens=16, threshold=0.5)

_PLAIN_TEXT = (
    "Cats are wonderful animals. Cats love to play outside. "
    "Dogs are loyal friends. Dogs enjoy long walks every day. "
    "Birds sing beautifully in the morning. Birds migrate south for winter."
)

_MD_TEXT = (
    "# Guide\n"
    "\n"
    "Intro paragraph about the guide and its contents overall.\n"
    "\n"
    "## Alpha\n"
    "\n"
    "Alpha talks about cats and their habits. Cats sleep a lot every day.\n"
    "\n"
    "### A1\n"
    "\n"
    "A1 goes deeper into dogs. Dogs enjoy long walks every day.\n"
    "\n"
    "## Beta\n"
    "\n"
    "Beta covers birds and migration patterns. Birds migrate south for winter.\n"
)

# One section whose small paragraphs force multiple chunks, so overlap is exercised.
_OVERLAP_MD_TEXT = (
    "# Doc\n"
    "\n"
    "First paragraph cats.\n"
    "\n"
    "Second paragraph dogs.\n"
    "\n"
    "Third paragraph birds.\n"
    "\n"
    "Fourth paragraph cats again.\n"
    "\n"
    "Fifth paragraph dogs again.\n"
)


def _make_doc(text: str, doc_id: str = "docs/a.md", title: str = "Doc") -> Document:
    _title, sections = parse_sections(doc_id, text)
    return Document(doc_id=doc_id, title=title, text=text, sections=sections)


PLAIN_DOC = _make_doc(_PLAIN_TEXT)
MD_DOC = _make_doc(_MD_TEXT)
OVERLAP_DOC = _make_doc(_OVERLAP_MD_TEXT)


def _count_tokens(text: str) -> int:
    return len(get_tokenizer().encode_ordinary(text))


class FakeEmbedder:
    """Deterministic offline embedder: topic direction per keyword, no network."""

    name: str = "fake"
    dim: int = 2

    def __init__(self) -> None:
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        self.document_calls.append(list(texts))
        rows = [self._vec(t) for t in texts]
        return np.array(rows, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        self.query_calls.append(text)
        return self._vec(text)

    @staticmethod
    def _vec(text: str) -> np.ndarray:
        lowered = text.lower()
        if "cats" in lowered:
            return np.array([1.0, 0.0], dtype=np.float32)
        if "dogs" in lowered:
            return np.array([0.0, 1.0], dtype=np.float32)
        if "birds" in lowered:
            return np.array([1.0, 0.2], dtype=np.float32)
        return np.array([1.0, 1.0], dtype=np.float32)


@pytest.fixture(params=["fixed", "recursive_md", "semantic"])
def chunker_case(request: pytest.FixtureRequest) -> tuple[Chunker, ChunkerCfg]:
    if request.param == "fixed":
        return FixedChunker(FIXED_CFG), FIXED_CFG
    if request.param == "recursive_md":
        return RecursiveMarkdownChunker(RECURSIVE_CFG), RECURSIVE_CFG
    return SemanticChunker(SEMANTIC_CFG, FakeEmbedder()), SEMANTIC_CFG


def _all_docs() -> list[Document]:
    return [PLAIN_DOC, MD_DOC]


# ------------------------------------------------------------------- shared invariants


def test_chunks_have_exact_offsets_and_text_slice(chunker_case: tuple[Chunker, ChunkerCfg]) -> None:
    chunker, _cfg = chunker_case
    for doc in _all_docs():
        chunks = chunker.chunk(doc)
        assert chunks, doc.doc_id
        for chunk in chunks:
            assert 0 <= chunk.start < chunk.end <= len(doc.text), chunk
            assert chunk.text == doc.text[chunk.start : chunk.end], chunk


def test_chunks_propagate_doc_id_and_have_chunk_ids(
    chunker_case: tuple[Chunker, ChunkerCfg],
) -> None:
    chunker, _cfg = chunker_case
    for doc in _all_docs():
        for chunk in chunker.chunk(doc):
            assert chunk.doc_id == doc.doc_id
            assert re.fullmatch(r"[0-9a-f]{64}", chunk.chunk_id)


def test_chunk_ids_unique_within_a_document(chunker_case: tuple[Chunker, ChunkerCfg]) -> None:
    chunker, _cfg = chunker_case
    for doc in _all_docs():
        ids = [chunk.chunk_id for chunk in chunker.chunk(doc)]
        assert len(ids) == len(set(ids))


def test_chunking_is_deterministic(chunker_case: tuple[Chunker, ChunkerCfg]) -> None:
    chunker, _cfg = chunker_case
    for doc in _all_docs():
        assert chunker.chunk(doc) == chunker.chunk(doc)


def test_empty_document_yields_no_chunks(chunker_case: tuple[Chunker, ChunkerCfg]) -> None:
    chunker, _cfg = chunker_case
    empty = _make_doc("", doc_id="docs/empty.md")
    assert chunker.chunk(empty) == []


def test_whitespace_only_document_yields_no_zero_length_chunks(
    chunker_case: tuple[Chunker, ChunkerCfg],
) -> None:
    chunker, _cfg = chunker_case
    doc = _make_doc("   \n\n  \n", doc_id="docs/ws.md")
    for chunk in chunker.chunk(doc):
        assert 0 <= chunk.start < chunk.end <= len(doc.text)
        assert chunk.text == doc.text[chunk.start : chunk.end]


def test_chunk_ids_follow_make_chunk_id_rule(chunker_case: tuple[Chunker, ChunkerCfg]) -> None:
    chunker, cfg = chunker_case
    for doc in _all_docs():
        for chunk in chunker.chunk(doc):
            assert chunk.chunk_id == make_chunk_id(
                doc.doc_id, chunk.start, chunk.end, cfg.model_dump(mode="json")
            )


# ------------------------------------------------------------------------- make_chunk_id


def test_make_chunk_id_is_stable_and_hex64() -> None:
    config = {"type": "fixed", "size_tokens": 16, "overlap_tokens": 4}
    first = make_chunk_id("docs/a.md", 0, 100, config)
    reordered = {"overlap_tokens": 4, "size_tokens": 16, "type": "fixed"}
    second = make_chunk_id("docs/a.md", 0, 100, reordered)
    assert first == second
    assert re.fullmatch(r"[0-9a-f]{64}", first)


def test_make_chunk_id_is_config_sensitive() -> None:
    base = make_chunk_id("docs/a.md", 0, 100, {"window": 512, "overlap": 64})
    changed = make_chunk_id("docs/a.md", 0, 100, {"window": 256, "overlap": 64})
    assert base != changed


def test_make_chunk_id_is_span_and_doc_sensitive() -> None:
    config = {"window": 512}
    base = make_chunk_id("docs/a.md", 0, 100, config)
    assert base != make_chunk_id("docs/b.md", 0, 100, config)
    assert base != make_chunk_id("docs/a.md", 0, 64, config)
    assert base != make_chunk_id("docs/a.md", 1, 100, config)


def test_make_chunk_id_matches_reference_encoding() -> None:
    config = {"b": 2, "a": 1}
    payload = "docs/a.md|0|100|" + '{"a":1,"b":2}'
    assert (
        make_chunk_id("docs/a.md", 0, 100, config)
        == hashlib.sha256(payload.encode("utf-8")).hexdigest()
    )


# ------------------------------------------------------------------------ registry


def test_get_chunker_returns_matching_chunker_per_type() -> None:
    assert isinstance(get_chunker(FIXED_CFG), FixedChunker)
    assert isinstance(get_chunker(RECURSIVE_CFG), RecursiveMarkdownChunker)
    fake = FakeEmbedder()
    assert isinstance(get_chunker(SEMANTIC_CFG, embedder=fake), SemanticChunker)


def test_get_chunker_semantic_without_embedder_raises_naming_embedder() -> None:
    with pytest.raises(ValueError, match="embedder"):
        get_chunker(SEMANTIC_CFG)


def test_get_chunker_fixed_and_recursive_ignore_embedder() -> None:
    fake = FakeEmbedder()
    assert isinstance(get_chunker(FIXED_CFG, embedder=fake), FixedChunker)
    assert isinstance(get_chunker(RECURSIVE_CFG, embedder=fake), RecursiveMarkdownChunker)


# --------------------------------------------------------------------------- fixed


def test_fixed_windows_tile_contiguously_without_overlap() -> None:
    cfg = ChunkerCfg(type="fixed", size_tokens=16, overlap_tokens=0)
    chunker = FixedChunker(cfg)
    chunks = chunker.chunk(PLAIN_DOC)
    assert chunks[0].start == 0
    assert chunks[-1].end == len(PLAIN_DOC.text)
    for prev, cur in zip(chunks, chunks[1:], strict=False):
        assert prev.end == cur.start


def test_fixed_window_count_matches_token_arithmetic() -> None:
    cfg = ChunkerCfg(type="fixed", size_tokens=16, overlap_tokens=0)
    n_tokens = _count_tokens(PLAIN_DOC.text)
    chunks = FixedChunker(cfg).chunk(PLAIN_DOC)
    assert len(chunks) == math.ceil(n_tokens / 16)


def test_fixed_overlapping_windows_step_and_bounds() -> None:
    cfg = ChunkerCfg(type="fixed", size_tokens=16, overlap_tokens=4)
    n_tokens = _count_tokens(PLAIN_DOC.text)
    chunks = FixedChunker(cfg).chunk(PLAIN_DOC)
    step = 16 - 4
    expected = 1 if n_tokens <= 16 else math.ceil((n_tokens - 16) / step) + 1
    assert len(chunks) == expected
    assert chunks[0].start == 0
    assert chunks[-1].end == len(PLAIN_DOC.text)
    for prev, cur in zip(chunks, chunks[1:], strict=False):
        assert cur.start < prev.end  # windows overlap
    for chunk in chunks:
        assert _count_tokens(chunk.text) <= 16


def test_fixed_rejects_overlap_not_smaller_than_size() -> None:
    cfg = ChunkerCfg(type="fixed", size_tokens=16, overlap_tokens=16)
    with pytest.raises(ValueError, match="overlap"):
        FixedChunker(cfg)


# ----------------------------------------------------------------------- recursive_md


def test_recursive_md_splits_only_at_heading_or_paragraph_boundaries() -> None:
    chunker = RecursiveMarkdownChunker(RECURSIVE_CFG)
    for chunk in chunker.chunk(MD_DOC):
        pos = chunk.start
        if pos == 0:
            continue
        line_start = MD_DOC.text.rfind("\n", 0, pos) + 1
        at_heading = re.match(r"#{1,6}\s", MD_DOC.text[line_start:]) is not None
        at_paragraph = re.search(r"\n\s*\n\s*$", MD_DOC.text[:pos]) is not None
        assert at_heading or at_paragraph, chunk


def test_recursive_md_respects_size_and_isolates_oversize_units() -> None:
    cfg = ChunkerCfg(type="recursive_md", size_tokens=16, overlap_tokens=0)
    text = (
        "# Doc\n"
        "\n"
        "Short intro line here.\n"
        "\n"
        "This single paragraph is deliberately much longer than the sixteen token "
        "budget so that it must be emitted alone as one indivisible oversize unit.\n"
        "\n"
        "Tail line.\n"
    )
    doc = _make_doc(text)
    chunks = RecursiveMarkdownChunker(cfg).chunk(doc)
    oversize = [c for c in chunks if _count_tokens(c.text) > 16]
    assert len(oversize) == 1
    assert oversize[0].text.strip().startswith("This single paragraph")
    assert oversize[0].text.strip().endswith("indivisible oversize unit.")
    for chunk in chunks:
        if chunk is not oversize[0]:
            assert _count_tokens(chunk.text) <= 16


def test_recursive_md_overlap_is_paragraph_level_and_bounded() -> None:
    cfg = ChunkerCfg(type="recursive_md", size_tokens=16, overlap_tokens=4)
    chunker = RecursiveMarkdownChunker(cfg)
    chunks = chunker.chunk(OVERLAP_DOC)
    assert len(chunks) >= 2
    overlapped = 0
    for prev, cur in zip(chunks, chunks[1:], strict=False):
        if cur.start >= prev.end:
            continue
        overlapped += 1
        overlap_text = OVERLAP_DOC.text[cur.start : prev.end]
        assert _count_tokens(overlap_text) <= 4
        assert re.search(r"\n\s*\n\s*$", OVERLAP_DOC.text[: cur.start])
    assert overlapped >= 1


def test_recursive_md_metadata_carries_heading_path() -> None:
    chunker = RecursiveMarkdownChunker(RECURSIVE_CFG)
    chunks = chunker.chunk(MD_DOC)
    paths = {chunk.metadata["heading_path"] for chunk in chunks}
    assert "Guide" in paths
    assert "Guide > Alpha" in paths
    assert "Guide > Alpha > A1" in paths
    assert "Guide > Beta" in paths


def test_recursive_md_rejects_overlap_not_smaller_than_size() -> None:
    cfg = ChunkerCfg(type="recursive_md", size_tokens=16, overlap_tokens=20)
    with pytest.raises(ValueError, match="overlap"):
        RecursiveMarkdownChunker(cfg)


def test_recursive_md_chunks_text_before_the_first_heading() -> None:
    doc = _make_doc("Intro before any heading.\n\n# Title\n\nBody paragraph.\n")
    chunks = RecursiveMarkdownChunker(RECURSIVE_CFG).chunk(doc)
    assert chunks
    assert chunks[0].start == 0
    assert chunks[0].metadata["heading_path"] == ""
    assert "Title" in doc.text[chunks[-1].start : chunks[-1].end]
    assert any(c.metadata["heading_path"] == "Title" for c in chunks)


def test_recursive_md_size_holds_when_paragraphs_pack_in_one_flush() -> None:
    cfg = ChunkerCfg(type="recursive_md", size_tokens=16, overlap_tokens=0)
    text = "\n\n".join(
        [
            "Alpha 1 talks about dogs and cats",
            "Alpha 2 talks about dogs and cats",
            "Alpha 3 talks about dogs and cats",
            "Alpha 4 talks about dogs and cats",
            "Alpha 5 talks about dogs and cats",
            "Alpha 6 talks about dogs and cats",
            "Beta is tiny",
            "Beta is tiny again",
        ]
    )
    doc = _make_doc(text)
    chunks = RecursiveMarkdownChunker(cfg).chunk(doc)
    assert chunks
    for chunk in chunks:
        assert _count_tokens(chunk.text) <= 16, chunk
    assert any("\n\n" in chunk.text for chunk in chunks)


def test_recursive_md_overlap_stays_within_budget_when_paragraphs_are_dense() -> None:
    cfg = ChunkerCfg(type="recursive_md", size_tokens=16, overlap_tokens=6)
    text = "\n\n".join(
        [
            "cats and more",
            "birds fly far",
            "dogs run fast",
            "cats sleep now",
            "birds sing loud",
            "dogs bark here",
            "cats play out",
            "birds glide by",
        ]
    )
    doc = _make_doc(text)
    chunks = RecursiveMarkdownChunker(cfg).chunk(doc)
    assert len(chunks) >= 2
    overlapped = 0
    for prev, cur in zip(chunks, chunks[1:], strict=False):
        if cur.start >= prev.end:
            continue
        overlapped += 1
        overlap_text = doc.text[cur.start : prev.end]
        assert _count_tokens(overlap_text) <= 6, overlap_text
    assert overlapped >= 1
    for chunk in chunks:
        assert _count_tokens(chunk.text) <= 16, chunk


# ------------------------------------------------------------------------- semantic


def test_semantic_chunks_respect_min_and_max_bounds() -> None:
    chunker = SemanticChunker(SEMANTIC_CFG, FakeEmbedder())
    for doc in (PLAIN_DOC, MD_DOC):
        chunks = chunker.chunk(doc)
        assert chunks
        for chunk in chunks:
            assert _count_tokens(chunk.text) <= 16, chunk
        for chunk in chunks[:-1]:
            assert _count_tokens(chunk.text) >= 8, chunk


def test_semantic_breaks_at_low_similarity_topic_boundary() -> None:
    chunker = SemanticChunker(SEMANTIC_CFG, FakeEmbedder())
    chunks = chunker.chunk(PLAIN_DOC)
    texts = [chunk.text for chunk in chunks]
    assert len(texts) == 3
    assert "Cats" in texts[0] and "Dogs" not in texts[0] and "Birds" not in texts[0]
    assert "Dogs" in texts[1] and "Cats" not in texts[1] and "Birds" not in texts[1]
    assert "Birds" in texts[2] and "Cats" not in texts[2] and "Dogs" not in texts[2]


def test_semantic_force_splits_sentences_exceeding_max_tokens() -> None:
    long_sentence = "Dogs " * 40 + "end here."
    assert _count_tokens(long_sentence) > 16
    text = f"Cats are wonderful animals. {long_sentence} Birds sing in the morning."
    doc = _make_doc(text)
    chunker = SemanticChunker(SEMANTIC_CFG, FakeEmbedder())
    chunks = chunker.chunk(doc)
    for chunk in chunks:
        assert _count_tokens(chunk.text) <= 16
    joined = "".join(chunk.text for chunk in chunks)
    assert "Dogs" in joined
    windows = [c for c in chunks if c.text.strip().startswith("Dogs")]
    assert len(windows) >= 2


def test_semantic_uses_embed_documents_not_embed_query() -> None:
    fake = FakeEmbedder()
    chunker = SemanticChunker(SEMANTIC_CFG, fake)
    chunker.chunk(PLAIN_DOC)
    assert fake.document_calls and fake.document_calls[0]
    assert fake.query_calls == []


def test_semantic_leaves_metadata_empty() -> None:
    chunker = SemanticChunker(SEMANTIC_CFG, FakeEmbedder())
    for chunk in chunker.chunk(PLAIN_DOC):
        assert chunk.metadata == {}


def test_fixed_leaves_metadata_empty() -> None:
    chunker = FixedChunker(FIXED_CFG)
    for doc in _all_docs():
        for chunk in chunker.chunk(doc):
            assert chunk.metadata == {}


# ------------------------------------------------------------------- types sanction


def test_chunk_metadata_defaults_to_empty_dict() -> None:
    chunk = Chunk(chunk_id="f" * 64, doc_id="d", start=0, end=1, text="x")
    assert chunk.metadata == {}
