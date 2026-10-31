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


class ChunkerCfg(BaseModel):
    """Chunking strategy (spec section 5.2).

    ``threshold`` is the configurable cosine-similarity breakpoint used only by the
    ``semantic`` chunker; it defaults to 0.5, a neutral midpoint for the normalized
    sentence-embedding cosine similarity that semantic breakpoints are computed from.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["fixed", "recursive_md", "semantic"]
    size_tokens: int
    overlap_tokens: int
    threshold: float = 0.5


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
