"""Cold embedding-throughput benchmark (resolves the Phase 3 cache-dependence caveat).

Run JSONs record ``chunks_per_sec`` from indexing, which depends on the document
embedding cache. This benchmark instead calls the model directly (cache bypassed) on a
fixed text set, after a small warm-up batch so model load is excluded, and records one
comparable chunks/sec figure per embedder in ``results/embedder_throughput.json``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

WARMUP_TEXTS: int = 8


def measure_throughput(
    encode: Callable[[list[str]], Any], texts: Sequence[str], warmup: int = WARMUP_TEXTS
) -> dict[str, float]:
    """Time ``encode`` over ``texts`` after a warm-up call on the first ``warmup`` texts."""
    if not texts:
        raise ValueError("texts must not be empty")
    batch = list(texts)
    encode(batch[:warmup])
    start = time.perf_counter()
    encode(batch)
    seconds = time.perf_counter() - start
    return {"chunks": float(len(batch)), "seconds": seconds, "chunks_per_sec": len(batch) / seconds}


def write_throughput(path: Path, entries: dict[str, dict[str, Any]], text_source: str) -> Path:
    """Merge ``entries`` (keyed by embedder name) into the throughput JSON file."""
    data: dict[str, Any] = {}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("embedders", {}).update(entries)
    data["text_source"] = text_source
    data["updated_at"] = datetime.now(UTC).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path
