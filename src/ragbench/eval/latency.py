"""Latency profiling and indexing resource measurement (Task 3.3, spec section 5.7).

Query latency protocol: warm, single request at a time, every query in order, repeated
for ``passes >= 3`` passes; pass 0 is a warm-up and is DISCARDED, so p50/p95 per stage
come from passes ``1..passes-1`` (numpy linear-interpolated percentiles). Retrieval is
deterministic, so the ranked results returned are those of the last pass.

Indexing: wall time, embedding throughput (``chunks / embed_seconds``; ``None`` when no
embedding ran, e.g. the BM25 control), store ``size_bytes``, and peak process RSS sampled
by a background thread (psutil). Embedding throughput reflects the embedding-cache state
at index time — a warm cache makes it look faster than a cold model pass.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from types import TracebackType
from typing import Any

import numpy as np
import psutil

from ragbench.chunking.tokenizer import count_tokens

MIN_PASSES: int = 3


def percentile_summary(values: Sequence[float]) -> dict[str, float]:
    """``{"p50", "p95"}`` of ``values`` (numpy linear interpolation)."""
    if len(values) == 0:
        raise ValueError("cannot summarize an empty sequence of timings")
    arr = np.asarray(values, dtype=np.float64)
    return {"p50": float(np.percentile(arr, 50)), "p95": float(np.percentile(arr, 95))}


def profile_queries[R](
    retrieve: Callable[[str], tuple[R, Mapping[str, float]]],
    queries: Sequence[str],
    passes: int = MIN_PASSES,
) -> tuple[list[R], dict[str, dict[str, float]]]:
    """Run ``retrieve`` over ``queries`` for ``passes`` passes; drop pass 0; summarize.

    Returns ``(last_pass_results, {stage: {"p50", "p95"}})`` where stages are the keys
    of the timings mappings ``retrieve`` returns.
    """
    if passes < MIN_PASSES:
        raise ValueError(f"passes must be >= {MIN_PASSES} (pass 0 is warm-up), got {passes}")
    if not queries:
        raise ValueError("queries must not be empty")
    samples: dict[str, list[float]] = {}
    results: list[R] = []
    for pass_no in range(passes):
        results = []
        for question in queries:
            result, timings = retrieve(question)
            results.append(result)
            if pass_no == 0:
                continue
            for stage, ms in timings.items():
                samples.setdefault(stage, []).append(float(ms))
    return results, {stage: percentile_summary(values) for stage, values in samples.items()}


class ResourceSampler:
    """Context manager measuring wall time and peak RSS of the current process."""

    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self.wall_seconds: float = 0.0
        self.start_rss_bytes: int = 0
        self.peak_rss_bytes: int = 0
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start: float = 0.0

    def _sample(self) -> None:
        rss = int(self._process.memory_info().rss)
        self.peak_rss_bytes = max(self.peak_rss_bytes, rss)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def __enter__(self) -> ResourceSampler:
        self.start_rss_bytes = int(self._process.memory_info().rss)
        self.peak_rss_bytes = self.start_rss_bytes
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._start = time.perf_counter()
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._sample()  # final sample: memory still held at block exit is always seen
        self.wall_seconds = time.perf_counter() - self._start
        self._stop.set()
        if self._thread is not None:
            self._thread.join()


def measure_indexing(
    index_fn: Callable[[], Mapping[str, int | float]],
    *,
    size_fn: Callable[[], int],
) -> dict[str, Any]:
    """Run ``index_fn`` (e.g. ``index_documents``) under a :class:`ResourceSampler`.

    Returns the index stats plus ``wall_seconds``, ``peak_rss_bytes``, ``chunks_per_sec``
    (``None`` when ``embed_seconds`` is 0) and ``size_bytes`` from ``size_fn``.
    """
    with ResourceSampler() as sampler:
        stats = dict(index_fn())
    embed_seconds = float(stats.get("embed_seconds", 0.0))
    chunks = int(stats.get("chunks", 0))
    return {
        **stats,
        "wall_seconds": sampler.wall_seconds,
        "peak_rss_bytes": sampler.peak_rss_bytes,
        "chunks_per_sec": chunks / embed_seconds if embed_seconds > 0 else None,
        "size_bytes": int(size_fn()),
    }


def tokens_per_query(prompt: str, contexts: Iterable[str]) -> int:
    """cl100k token count of the generation prompt plus its retrieved contexts."""
    return count_tokens(prompt) + sum(count_tokens(text) for text in contexts)
