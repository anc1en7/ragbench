"""FastAPI query service over benchmarked pipelines (Task 5.1).

Endpoints: ``GET /health`` (service + pgvector/Qdrant connectivity), ``GET /configs``
(configs from ``configs/`` and ``results/runs/``), ``POST /query`` (retrieve, optionally
generate; returns contexts, per-stage ``timings_ms`` and a ``trace_id``) and
``GET /eval/summary`` (``results/summary.csv`` as JSON). OpenAPI docs are at ``/docs``.

Run: ``uvicorn ragbench.api.app:app`` (or ``docker compose up api``).
"""

from __future__ import annotations

import csv
import os
from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI, HTTPException

from ragbench.api.pipeline_cache import PipelineBuilder, PipelineCache, discover_configs
from ragbench.api.schemas import (
    BackendStatus,
    ConfigInfo,
    ContextOut,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)
from ragbench.config import PipelineConfig
from ragbench.tracing import configure_tracing, request_span

REPO_ROOT = Path(__file__).resolve().parents[3]
HealthCheck = Callable[[], BackendStatus]


def check_pgvector() -> BackendStatus:
    import psycopg

    from ragbench.stores.pgvector_store import DEFAULT_PG_DSN

    dsn = os.environ.get("PG_DSN") or DEFAULT_PG_DSN
    try:
        with psycopg.connect(dsn, connect_timeout=2) as conn:
            conn.execute("SELECT 1")
    except Exception as exc:  # any connection failure is reported, not raised
        return BackendStatus(ok=False, detail=f"{type(exc).__name__}: {exc}"[:200])
    return BackendStatus(ok=True, detail="connected")


def check_qdrant() -> BackendStatus:
    import httpx

    from ragbench.stores.qdrant_store import DEFAULT_QDRANT_URL

    url = (os.environ.get("QDRANT_URL") or DEFAULT_QDRANT_URL).rstrip("/")
    try:
        response = httpx.get(f"{url}/readyz", timeout=2.0)
        response.raise_for_status()
    except Exception as exc:  # any connection failure is reported, not raised
        return BackendStatus(ok=False, detail=f"{type(exc).__name__}: {exc}"[:200])
    return BackendStatus(ok=True, detail="ready")


def _describe(config_id: str, cfg: PipelineConfig, source: str) -> ConfigInfo:
    size = cfg.chunker.size_tokens
    chunker = f"{cfg.chunker.type}-{size}" if size else cfg.chunker.type
    rerank = (
        cfg.rerank.type if cfg.rerank.type == "none" else f"{cfg.rerank.type}@{cfg.rerank.top_n}"
    )
    return ConfigInfo(
        config_id=config_id,
        source=source,
        corpus=cfg.corpus,
        backend=cfg.backend,
        chunker=chunker,
        embedder=cfg.embedder.name,
        retrieval_mode=cfg.retrieval.mode,
        rerank=rerank,
    )


def create_app(
    *,
    configs_dir: Path | None = None,
    results_dir: Path | None = None,
    builder: PipelineBuilder | None = None,
    health_checks: dict[str, HealthCheck] | None = None,
    cache_size: int | None = None,
) -> FastAPI:
    """Build the app. Every keyword argument is a test seam with a production default."""
    configs_path = configs_dir or Path(
        os.environ.get("RAGBENCH_CONFIGS_DIR", REPO_ROOT / "configs")
    )
    results_path = results_dir or Path(
        os.environ.get("RAGBENCH_RESULTS_DIR", REPO_ROOT / "results")
    )
    checks = (
        health_checks
        if health_checks is not None
        else {"pgvector": check_pgvector, "qdrant": check_qdrant}
    )
    size = cache_size or int(os.environ.get("API_PIPELINE_CACHE", "4"))
    cache = PipelineCache(builder, size) if builder is not None else PipelineCache(max_size=size)
    configure_tracing()

    app = FastAPI(
        title="RAGBench API",
        version="0.1.0",
        description="Query benchmarked RAG pipeline configs and read evaluation results.",
    )
    app.state.cache = cache

    def configs() -> dict[str, tuple[PipelineConfig, str]]:
        return discover_configs(configs_path, results_path / "runs")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        backends = {name: check() for name, check in checks.items()}
        healthy = all(b.ok for b in backends.values())
        return HealthResponse(
            status="ok" if healthy else "degraded", service="ragbench-api", backends=backends
        )

    @app.get("/configs", response_model=list[ConfigInfo])
    def list_configs() -> list[ConfigInfo]:
        return [_describe(cid, cfg, src) for cid, (cfg, src) in sorted(configs().items())]

    @app.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
        known = configs()
        if request.config_id not in known:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"unknown config_id {request.config_id!r}; GET /configs lists the valid ids"
                ),
            )
        cfg = known[request.config_id][0]
        with request_span("query", config_id=cfg.config_id) as trace_id:
            try:
                pipeline = cache.get(cfg)
                hits, timings = pipeline.retrieve(request.question)
            except (RuntimeError, ValueError, OSError) as exc:
                raise HTTPException(
                    status_code=503, detail=f"retrieval failed for {cfg.config_id}: {exc}"
                ) from exc
            answer: str | None = None
            if request.generate:
                try:
                    answer = pipeline.generate(request.question, hits)
                except Exception as exc:  # upstream LLM errors become a clear 502
                    raise HTTPException(
                        status_code=502, detail=f"generation failed: {exc}"
                    ) from exc
        contexts = [
            ContextOut(
                rank=rank,
                score=float(hit.score),
                source=hit.source,
                doc_id=hit.chunk.doc_id,
                chunk_id=hit.chunk.chunk_id,
                start=hit.chunk.start,
                end=hit.chunk.end,
                text=hit.chunk.text,
            )
            for rank, hit in enumerate(hits, start=1)
        ]
        return QueryResponse(
            config_id=cfg.config_id,
            question=request.question,
            contexts=contexts,
            answer=answer,
            timings_ms=timings,
            trace_id=trace_id,
        )

    @app.get("/eval/summary")
    def eval_summary() -> list[dict[str, str]]:
        path = results_path / "summary.csv"
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"no results yet: {path.name} not found")
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    return app


app = create_app()
