"""Pydantic pipeline configuration models (spec section 5.1).

A pipeline is fully described by YAML validated into these models. They are the single
source of truth for pipeline configuration: later stages (chunking, stores, reranking,
sweeping, the API) read their settings from here, and ``PipelineConfig.config_id`` is
the short hash identifying a configuration across runs and results.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

_DEFAULT_SEMANTIC_THRESHOLD: float = 0.5


class ChunkerCfg(BaseModel):
    """Chunking strategy (spec section 5.2).

    Per-type parameters:

    - ``fixed`` and ``recursive_md`` are fixed-window splitters and require
      ``size_tokens``/``overlap_tokens``. They ignore ``min_tokens``/``max_tokens``
      (present but inert, so dumps round-trip).
    - ``semantic`` does not require the window parameters (they may be present but are
      unused); it uses the ``min_tokens``/``max_tokens`` bounds instead, which default
      to 256/512 — the chunk-size envelope of spec section 5.2 — and ``threshold``, the
      configurable cosine-similarity breakpoint (spec section 5.2), which defaults to
      0.5, a neutral midpoint for the normalized sentence-embedding cosine similarity
      that semantic breakpoints are computed from. Set a bound to ``null`` to leave that
      side unbounded.
    - ``threshold`` set on ``fixed``/``recursive_md`` is rejected because it would be
      silently ignored; non-semantic dumps carry ``threshold: null``, so dumps still
      round-trip through ``model_validate``.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["fixed", "recursive_md", "semantic"]
    size_tokens: int | None = None
    overlap_tokens: int | None = None
    min_tokens: int | None = 256
    max_tokens: int | None = 512
    threshold: float | None = None

    @model_validator(mode="after")
    def _window_params_required_for_fixed_and_recursive(self) -> ChunkerCfg:
        if self.type in ("fixed", "recursive_md") and (
            self.size_tokens is None or self.overlap_tokens is None
        ):
            raise ValueError(
                f"chunker type '{self.type}' requires size_tokens and overlap_tokens; "
                "only 'semantic' may omit fixed-window parameters"
            )
        return self

    @model_validator(mode="after")
    def _threshold_rules(self) -> ChunkerCfg:
        if self.type == "semantic":
            if self.threshold is None:
                self.threshold = _DEFAULT_SEMANTIC_THRESHOLD
        elif self.threshold is not None:
            raise ValueError(
                "threshold applies only to chunker type 'semantic'; remove it from "
                f"type '{self.type}' or switch type"
            )
        return self

    @model_validator(mode="after")
    def _bounds_must_be_ordered(self) -> ChunkerCfg:
        if (
            self.min_tokens is not None
            and self.max_tokens is not None
            and self.min_tokens >= self.max_tokens
        ):
            raise ValueError(
                f"min_tokens ({self.min_tokens}) must be less than max_tokens ({self.max_tokens})"
            )
        return self


class EmbedderCfg(BaseModel):
    """Embedding model (spec section 5.5)."""

    model_config = ConfigDict(extra="forbid")

    name: str


class RetrievalCfg(BaseModel):
    """Retrieval stage (spec section 5.1)."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["dense", "sparse", "hybrid"]
    k_dense: int = 50
    k_sparse: int = 50
    rrf_k: int = 60


class RerankCfg(BaseModel):
    """Reranking stage (spec section 5.4); ``model`` is only required for ``cross_encoder``."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["none", "cross_encoder"]
    model: str | None = None
    top_n: int = 50

    @model_validator(mode="after")
    def _cross_encoder_needs_model(self) -> RerankCfg:
        if self.type == "cross_encoder" and self.model is None:
            raise ValueError("rerank.model is required when rerank.type is 'cross_encoder'")
        return self


class GenerationCfg(BaseModel):
    """Answer-generation stage (spec section 5.1)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    context_k: int = 5
    temperature: float = 0.0


class PipelineConfig(BaseModel):
    """Complete pipeline configuration (spec section 5.1)."""

    model_config = ConfigDict(extra="forbid")

    corpus: str
    chunker: ChunkerCfg
    embedder: EmbedderCfg
    backend: Literal["pgvector", "qdrant", "control_bm25"]
    retrieval: RetrievalCfg
    rerank: RerankCfg
    generation: GenerationCfg
    final_k: int = 10
    seed: int = 42

    @model_validator(mode="after")
    def _control_bm25_is_sparse_only(self) -> PipelineConfig:
        if self.backend == "control_bm25" and self.retrieval.mode in ("dense", "hybrid"):
            raise ValueError(
                "backend 'control_bm25' supports only sparse retrieval (spec section 5.3: "
                "dense_search raises NotImplementedError); set retrieval.mode to 'sparse', "
                f"got '{self.retrieval.mode}'"
            )
        return self

    @property
    def config_id(self) -> str:
        """First 12 hex chars of the sha256 over the canonical JSON of this config.

        Canonical JSON is the model dump WITH defaults filled, serialized as
        ``json.dumps(model_dump(mode="json"), sort_keys=True, separators=(",", ":"))``:
        keys sorted, whitespace removed. The id is therefore stable across dict/keyword
        ordering and changes exactly when any field value (explicit or default-filled)
        changes.
        """
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
