"""Unit tests for the rerankers (Task 2.8, controller rulings 2-4) — offline, no network.

``RerankNone`` is a min-max-normalizing passthrough; ``CrossEncoderReranker`` is tested
only through its ``model`` test seam (scripted stub) — no real CrossEncoder is ever
instantiated and no model download can happen (the lazy loader itself is monkeypatched
to a boom function to prove it is NOT called).
"""

import pytest

import ragbench.rerank.cross_encoder as cross_encoder_module
from ragbench.embedding.cache import hf_cache_root
from ragbench.rerank import CrossEncoderReranker, RerankNone
from ragbench.types import Chunk, Hit


def _hit(chunk_id: str, score: float, source: str, text: str | None = None) -> Hit:
    return Hit(
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id="docs/a.md",
            start=0,
            end=10,
            text=text if text is not None else f"text of {chunk_id}",
        ),
        score=score,
        source=source,
    )


class _StubModel:
    """Scripted stand-in for ``sentence_transformers.CrossEncoder`` (records calls)."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.predict_calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.predict_calls.append(list(pairs))
        return self._scores[: len(pairs)]


def _boom(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("model must not be loaded")


# --------------------------------------------------------------------------- RerankNone


def test_none_min_max_normalization_over_given_hits():
    hits = [_hit("a", 0.2, "dense"), _hit("b", 0.4, "dense"), _hit("c", 0.8, "dense")]
    result = RerankNone().rerank("query", hits, k=3)
    # min=0.2, max=0.8 (over ALL given hits): (0.2-0.2)/0.6=0.0;
    # (0.4-0.2)/0.6=1/3; (0.8-0.2)/0.6=1.0
    assert [h.score for h in result] == pytest.approx([0.0, 1 / 3, 1.0])


def test_none_truncates_to_first_k_after_normalizing_over_all_given_hits():
    hits = [_hit("a", 0.2, "dense"), _hit("b", 0.4, "dense"), _hit("c", 0.8, "dense")]
    result = RerankNone().rerank("query", hits, k=2)
    # min/max come from the GIVEN hits (0.2, 0.8), then the first k=2 are kept.
    assert [h.chunk.chunk_id for h in result] == ["a", "b"]
    assert [h.score for h in result] == pytest.approx([0.0, 1 / 3])


def test_none_all_equal_scores_become_one():
    hits = [_hit("a", 0.5, "dense"), _hit("b", 0.5, "sparse")]
    result = RerankNone().rerank("query", hits, k=5)
    # max == min -> every normalized score is 1.0.
    assert [h.score for h in result] == [1.0, 1.0]


def test_none_single_hit_score_becomes_one():
    result = RerankNone().rerank("query", [_hit("a", 0.3, "dense")], k=1)
    assert result[0].score == 1.0


def test_none_preserves_input_order_even_when_not_sorted_by_score():
    hits = [_hit("a", 0.9, "dense"), _hit("b", 0.1, "dense"), _hit("c", 0.5, "dense")]
    result = RerankNone().rerank("query", hits, k=3)
    # Passthrough keeps input order; it never re-sorts by score.
    assert [h.chunk.chunk_id for h in result] == ["a", "b", "c"]


def test_none_keeps_source_unchanged_from_input():
    hits = [_hit("a", 0.1, "dense"), _hit("b", 0.9, "sparse")]
    result = RerankNone().rerank("query", hits, k=2)
    # No reordering model ran -> source is NOT rewritten to "reranked".
    assert [h.source for h in result] == ["dense", "sparse"]


def test_none_returns_new_hits_without_mutating_inputs():
    hits = [_hit("a", 0.2, "dense"), _hit("b", 0.8, "dense")]
    result = RerankNone().rerank("query", hits, k=2)
    assert all(new is not old for new, old in zip(result, hits, strict=True))
    # Inputs untouched: original scores/sources preserved.
    assert [h.score for h in hits] == [0.2, 0.8]
    assert [h.source for h in hits] == ["dense", "dense"]


def test_none_non_positive_k_or_empty_hits_return_empty():
    hits = [_hit("a", 0.2, "dense")]
    assert RerankNone().rerank("query", hits, k=0) == []
    assert RerankNone().rerank("query", hits, k=-1) == []
    assert RerankNone().rerank("query", [], k=3) == []


# ----------------------------------------------------------- CrossEncoderReranker


def test_cross_encoder_sorts_by_score_descending_with_raw_scores():
    # Scripted scores align with input order: a=0.2, b=0.9, c=0.5.
    stub = _StubModel([0.2, 0.9, 0.5])
    hits = [_hit("a", 0.0, "fused"), _hit("b", 0.0, "fused"), _hit("c", 0.0, "fused")]
    result = CrossEncoderReranker(model=stub).rerank("query text", hits, k=3)
    assert [h.chunk.chunk_id for h in result] == ["b", "c", "a"]
    # Raw model scores, no normalization.
    assert [h.score for h in result] == pytest.approx([0.9, 0.5, 0.2])
    assert all(h.source == "reranked" for h in result)


def test_cross_encoder_tie_breaks_by_chunk_id_ascending():
    stub = _StubModel([0.7, 0.7, 0.7])  # full score tie across inputs c, a, b
    hits = [_hit("c", 0.0, "fused"), _hit("a", 0.0, "fused"), _hit("b", 0.0, "fused")]
    result = CrossEncoderReranker(model=stub).rerank("query", hits, k=3)
    assert [h.chunk.chunk_id for h in result] == ["a", "b", "c"]


def test_cross_encoder_full_ties_keep_input_order():
    # Equal score AND equal chunk_id -> stable sort preserves input order.
    stub = _StubModel([0.5, 0.5])
    first = _hit("dup", 0.0, "fused", text="first body")
    second = _hit("dup", 0.0, "fused", text="second body")
    result = CrossEncoderReranker(model=stub).rerank("query", [first, second], k=2)
    assert [h.chunk.text for h in result] == ["first body", "second body"]


def test_cross_encoder_scores_all_input_hits_then_truncates_to_k():
    # Pipeline shape: top_n=5 fused candidates in, final_k=2 out.
    stub = _StubModel([0.1, 0.4, 0.9, 0.7, 0.5])
    hits = [_hit(cid, 0.0, "fused") for cid in "abcde"]
    result = CrossEncoderReranker(model=stub).rerank("query", hits, k=2)
    assert [h.chunk.chunk_id for h in result] == ["c", "d"]
    # Predict ran ONCE over ALL 5 candidates; truncation happens after scoring.
    assert len(stub.predict_calls) == 1
    assert len(stub.predict_calls[0]) == 5


def test_cross_encoder_predict_receives_query_text_pairs_in_input_order():
    stub = _StubModel([0.3, 0.6])
    hits = [
        _hit("a", 0.0, "fused", text="alpha body"),
        _hit("b", 0.0, "fused", text="beta body"),
    ]
    CrossEncoderReranker(model=stub).rerank("my query", hits, k=2)
    assert stub.predict_calls == [[("my query", "alpha body"), ("my query", "beta body")]]


def test_cross_encoder_returns_new_hits_without_mutating_inputs():
    stub = _StubModel([0.9])
    hit = _hit("a", 0.1, "fused")
    result = CrossEncoderReranker(model=stub).rerank("query", [hit], k=1)
    assert result[0] is not hit
    assert hit.score == 0.1
    assert hit.source == "fused"


def test_cross_encoder_non_positive_k_returns_empty_without_predict_or_load(
    monkeypatch: pytest.MonkeyPatch,
):
    # model=None exercises the LAZY path: any load attempt hits _boom and fails.
    monkeypatch.setattr(cross_encoder_module, "_load_cross_encoder", _boom)
    hits = [_hit("a", 0.1, "fused")]
    assert CrossEncoderReranker().rerank("query", hits, k=0) == []
    assert CrossEncoderReranker().rerank("query", hits, k=-3) == []
    # Stub seam pins the no-predict half (predict would be recorded if called).
    stub = _StubModel([])
    assert CrossEncoderReranker(model=stub).rerank("query", hits, k=0) == []
    assert CrossEncoderReranker(model=stub).rerank("query", hits, k=-3) == []
    assert stub.predict_calls == []


def test_cross_encoder_empty_hits_return_empty_without_predict_or_load(
    monkeypatch: pytest.MonkeyPatch,
):
    # model=None exercises the LAZY path: any load attempt hits _boom and fails.
    monkeypatch.setattr(cross_encoder_module, "_load_cross_encoder", _boom)
    assert CrossEncoderReranker().rerank("query", [], k=5) == []
    # Stub seam pins the no-predict half (predict would be recorded if called).
    stub = _StubModel([])
    assert CrossEncoderReranker(model=stub).rerank("query", [], k=5) == []
    assert stub.predict_calls == []


def test_cross_encoder_model_seam_skips_lazy_loading(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cross_encoder_module, "_load_cross_encoder", _boom)
    stub = _StubModel([0.4])
    result = CrossEncoderReranker(model=stub).rerank("query", [_hit("a", 0.0, "fused")], k=1)
    assert len(result) == 1
    assert len(stub.predict_calls) == 1


def test_cross_encoder_lazy_loads_once_with_default_name_and_hf_cache(
    monkeypatch: pytest.MonkeyPatch,
):
    stub = _StubModel([0.8, 0.2])
    loads: list[tuple[str, str]] = []

    def fake_load(model_name: str, cache_folder: str) -> object:
        loads.append((model_name, cache_folder))
        return stub

    monkeypatch.setattr(cross_encoder_module, "_load_cross_encoder", fake_load)
    reranker = CrossEncoderReranker()  # model=None -> lazy on first rerank
    hits = [_hit("a", 0.0, "fused"), _hit("b", 0.0, "fused")]
    reranker.rerank("query", hits, k=1)
    reranker.rerank("query", hits, k=1)
    # Loaded exactly once, with the default model name and the project-drive cache root.
    assert loads == [("BAAI/bge-reranker-base", str(hf_cache_root()))]
    assert len(stub.predict_calls) == 2


def test_cross_encoder_passes_explicit_model_name_and_cache_folder(
    monkeypatch: pytest.MonkeyPatch,
):
    stub = _StubModel([0.9])
    loads: list[tuple[str, str]] = []

    def fake_load(model_name: str, cache_folder: str) -> object:
        loads.append((model_name, cache_folder))
        return stub

    monkeypatch.setattr(cross_encoder_module, "_load_cross_encoder", fake_load)
    reranker = CrossEncoderReranker("my/model", cache_folder="D:/custom/cache")
    reranker.rerank("query", [_hit("a", 0.0, "fused")], k=1)
    assert loads == [("my/model", "D:/custom/cache")]
