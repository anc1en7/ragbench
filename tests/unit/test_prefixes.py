"""Prefix/instruction contract tests for the embedder layer (Task 2.4, rulings 1 and 7).

The expected strings below are transcribed from the three Hugging Face model cards
verified on 2026-10-31 (URLs recorded in ``docs/METHODOLOGY.md``). They are tested as
pure functions — no model loads, no network.
"""

from ragbench.embedding.base import document_prefix, query_prefix

E5 = "intfloat/e5-small-v2"
BGE = "BAAI/bge-small-en-v1.5"
MINILM = "sentence-transformers/all-MiniLM-L6-v2"


def test_e5_query_prefix() -> None:
    assert query_prefix(E5) == "query: "


def test_e5_document_prefix() -> None:
    assert document_prefix(E5) == "passage: "


def test_bge_query_instruction_exact_with_trailing_space() -> None:
    assert query_prefix(BGE) == "Represent this sentence for searching relevant passages: "


def test_bge_document_unprefixed() -> None:
    assert document_prefix(BGE) == ""


def test_minilm_has_no_prefix_on_either_side() -> None:
    assert query_prefix(MINILM) == ""
    assert document_prefix(MINILM) == ""


def test_unknown_model_is_unprefixed_on_both_sides() -> None:
    assert query_prefix("some/unknown-model") == ""
    assert document_prefix("some/unknown-model") == ""
