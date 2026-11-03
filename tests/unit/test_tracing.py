"""Unit tests for OpenTelemetry stage tracing (Task 5.2)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from ragbench.config import PipelineConfig
from ragbench.pipeline import build_pipeline, index_documents
from ragbench.tracing import setup as tracing
from ragbench.types import Document

CFG = PipelineConfig.model_validate(
    {
        "corpus": "mini",
        "chunker": {"type": "fixed", "size_tokens": 64, "overlap_tokens": 0},
        "embedder": {"name": "BAAI/bge-small-en-v1.5"},
        "backend": "control_bm25",
        "retrieval": {"mode": "sparse"},
        "rerank": {"type": "none", "top_n": 10},
        "generation": {"enabled": False},
        "final_k": 3,
    }
)
DOCS = [
    Document(doc_id=f"d{i}.md", title=f"D{i}", text=f"document {i} about topic {i}", sections=[])
    for i in range(5)
]


@pytest.fixture(autouse=True)
def _reset() -> Iterator[None]:
    tracing._reset_for_tests()
    yield
    tracing._reset_for_tests()


def test_disabled_configures_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRACING", raising=False)
    exporter = InMemorySpanExporter()
    assert tracing.configure_tracing(exporter=exporter) is False
    assert tracing._state["tracer"] is None
    assert tracing.stage_span("dense") is tracing._NOOP
    with tracing.request_span("query") as trace_id:
        assert len(trace_id) == 32
    pipeline = build_pipeline(CFG)
    index_documents(DOCS, pipeline)
    pipeline.retrieve("topic 3")
    assert exporter.get_finished_spans() == ()


def test_enabled_emits_stage_spans_under_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRACING", "1")
    exporter = InMemorySpanExporter()
    assert tracing.configure_tracing(exporter=exporter) is True
    pipeline = build_pipeline(CFG)
    index_documents(DOCS, pipeline)
    with tracing.request_span("query") as trace_id:
        pipeline.retrieve("topic 3")
    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert {"ragbench.query", "ragbench.sparse", "ragbench.rerank"} <= set(spans)
    assert format(spans["ragbench.sparse"].context.trace_id, "032x") == trace_id
    assert "ragbench.dense" not in spans  # sparse mode never runs the dense stage
