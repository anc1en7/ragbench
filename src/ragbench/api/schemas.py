"""Request and response models for the RAGBench query API (Task 5.1)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MAX_QUESTION_CHARS = 2000


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        ...,
        min_length=1,
        max_length=MAX_QUESTION_CHARS,
        description=f"Natural-language question, 1-{MAX_QUESTION_CHARS} characters.",
    )
    config_id: str = Field(..., description="12-hex config id; GET /configs lists valid ids.")
    generate: bool = Field(False, description="Also generate an answer with the LLM endpoint.")


class ContextOut(BaseModel):
    rank: int
    score: float
    source: str
    doc_id: str
    chunk_id: str
    start: int
    end: int
    text: str


class QueryResponse(BaseModel):
    config_id: str
    question: str
    contexts: list[ContextOut]
    answer: str | None
    timings_ms: dict[str, float]
    trace_id: str


class BackendStatus(BaseModel):
    ok: bool
    detail: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    service: str
    backends: dict[str, BackendStatus]


class ConfigInfo(BaseModel):
    config_id: str
    source: str
    corpus: str
    backend: str
    chunker: str
    embedder: str
    retrieval_mode: str
    rerank: str
