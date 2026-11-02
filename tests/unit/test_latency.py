"""Unit tests for the latency profiler and indexing resource metrics (Task 3.3) — offline."""

from __future__ import annotations

import time

import numpy as np
import pytest

from ragbench.chunking.tokenizer import count_tokens
from ragbench.eval.latency import (
    ResourceSampler,
    measure_indexing,
    percentile_summary,
    profile_queries,
    tokens_per_query,
)

STAGES = ("embed_query", "dense", "total")


class ScriptedRetriever:
    """Returns pass-dependent timings: pass 0 is a huge warm-up outlier."""

    def __init__(self, queries: list[str]) -> None:
        self.queries = queries
        self.calls: list[str] = []

    def __call__(self, question: str) -> tuple[str, dict[str, float]]:
        call_no = len(self.calls)
        self.calls.append(question)
        pass_no = call_no // len(self.queries)
        q_idx = self.queries.index(question)
        if pass_no == 0:
            timings = dict.fromkeys(STAGES, 10_000.0)
        else:
            base = float(10 * pass_no + q_idx)  # pass 1: 10,11,12 ; pass 2: 20,21,22
            timings = {"embed_query": base, "dense": base * 2, "total": base * 3}
        return f"result:{question}:{pass_no}", timings


def test_percentile_summary_linear_interpolation():
    values = [10.0, 11.0, 12.0, 20.0, 21.0, 22.0]
    summary = percentile_summary(values)
    assert summary["p50"] == float(np.percentile(values, 50))
    assert summary["p95"] == float(np.percentile(values, 95))
    assert summary["p50"] == 16.0  # midpoint of 12 and 20
    with pytest.raises(ValueError, match="empty"):
        percentile_summary([])


def test_profile_discards_warmup_pass_and_summarizes_each_stage():
    queries = ["q0", "q1", "q2"]
    retriever = ScriptedRetriever(queries)

    results, latency = profile_queries(retriever, queries, passes=3)

    # every query run once per pass, in order, single request at a time
    assert retriever.calls == queries * 3
    # results are those of the LAST pass
    assert results == ["result:q0:2", "result:q1:2", "result:q2:2"]
    # pass 0 (10_000 ms) is discarded: stats come from passes 1-2 only
    measured = [10.0, 11.0, 12.0, 20.0, 21.0, 22.0]
    assert set(latency) == set(STAGES)
    assert latency["embed_query"]["p50"] == float(np.percentile(measured, 50))
    assert latency["embed_query"]["p95"] == float(np.percentile(measured, 95))
    assert latency["dense"]["p95"] == float(np.percentile([2 * v for v in measured], 95))
    assert latency["total"]["p50"] == float(np.percentile([3 * v for v in measured], 50))
    assert all(stage["p95"] < 10_000.0 for stage in latency.values())


def test_profile_requires_three_passes_and_queries():
    with pytest.raises(ValueError, match="passes"):
        profile_queries(ScriptedRetriever(["q"]), ["q"], passes=2)
    with pytest.raises(ValueError, match="queries"):
        profile_queries(ScriptedRetriever(["q"]), [], passes=3)


def test_resource_sampler_records_wall_time_and_peak_rss():
    with ResourceSampler(interval_s=0.01) as sampler:
        blob = b"\x01" * (64 * 2**20)  # touched pages, so RSS really grows
        time.sleep(0.05)
    assert len(blob) == 64 * 2**20  # keep alive until after the block
    assert sampler.wall_seconds >= 0.05
    assert sampler.start_rss_bytes > 0
    assert sampler.peak_rss_bytes >= sampler.start_rss_bytes + 32 * 2**20


def test_measure_indexing_combines_stats_size_and_throughput():
    def index_fn() -> dict[str, int | float]:
        time.sleep(0.01)
        return {"docs": 3, "chunks": 40, "embed_seconds": 2.0, "index_seconds": 0.5}

    stats = measure_indexing(index_fn, size_fn=lambda: 12_345)

    assert stats["docs"] == 3
    assert stats["chunks"] == 40
    assert stats["embed_seconds"] == 2.0
    assert stats["index_seconds"] == 0.5
    assert stats["chunks_per_sec"] == 20.0  # 40 chunks / 2.0 s embedding
    assert stats["size_bytes"] == 12_345
    assert stats["wall_seconds"] >= 0.01
    assert stats["peak_rss_bytes"] > 0


def test_measure_indexing_no_embedding_reports_null_throughput():
    stats = measure_indexing(
        lambda: {"docs": 1, "chunks": 5, "embed_seconds": 0.0, "index_seconds": 0.1},
        size_fn=lambda: 0,
    )
    assert stats["chunks_per_sec"] is None


def test_tokens_per_query_counts_prompt_plus_contexts():
    prompt = "Answer the question using the context."
    contexts = ["FastAPI declares query parameters.", "Path parameters are typed."]
    expected = count_tokens(prompt) + sum(count_tokens(c) for c in contexts)
    assert tokens_per_query(prompt, contexts) == expected
    assert tokens_per_query(prompt, []) == count_tokens(prompt)
