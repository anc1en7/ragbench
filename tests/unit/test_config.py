"""Spec section 5.1 config contract tests for ragbench.config (Task 2.2).

Covers the controller rulings: the verbatim spec section 5.1 example config, the
retrieval.mode and backend literals (backend includes control_bm25), the
control_bm25 cross-field rule, the default-filled fields, the config_id definition,
the grid skeleton, and extra="forbid" rejection of unknown keys anywhere.
"""

import re
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from ragbench.config import (
    ChunkerCfg,
    EmbedderCfg,
    GenerationCfg,
    PipelineConfig,
    RerankCfg,
    RetrievalCfg,
)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG_PATH = ROOT / "configs" / "pipeline.example.yaml"
GRID_PATH = ROOT / "configs" / "grid.yaml"


def _read_yaml(path: Path) -> dict[str, object]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _load(path: Path) -> PipelineConfig:
    return PipelineConfig.model_validate(_read_yaml(path))


def _minimal_data() -> dict[str, object]:
    """Config data that relies on every default listed in the controller rulings."""
    return {
        "corpus": "fastapi_docs",
        "chunker": {"type": "recursive_md", "size_tokens": 512, "overlap_tokens": 64},
        "embedder": {"name": "BAAI/bge-small-en-v1.5"},
        "backend": "pgvector",
        "retrieval": {"mode": "hybrid"},
        "rerank": {"type": "none"},
        "generation": {},
    }


def test_example_config_loads_and_matches_spec_section_5_1():
    cfg = _load(EXAMPLE_CONFIG_PATH)
    assert cfg.corpus == "fastapi_docs"
    assert cfg.chunker.type == "recursive_md"
    assert cfg.chunker.size_tokens == 512
    assert cfg.chunker.overlap_tokens == 64
    assert cfg.embedder.name == "BAAI/bge-small-en-v1.5"
    assert cfg.backend == "pgvector"
    assert cfg.retrieval.mode == "hybrid"
    assert cfg.retrieval.k_dense == 50
    assert cfg.retrieval.k_sparse == 50
    assert cfg.retrieval.rrf_k == 60
    assert cfg.rerank.type == "cross_encoder"
    assert cfg.rerank.model == "BAAI/bge-reranker-base"
    assert cfg.rerank.top_n == 50
    assert cfg.final_k == 10
    assert cfg.generation.enabled is True
    assert cfg.generation.context_k == 5
    assert cfg.generation.temperature == 0.0
    assert cfg.seed == 42


def test_grid_skeleton_parses_as_valid_yaml_pointing_at_spec_section_5_8():
    text = GRID_PATH.read_text(encoding="utf-8")
    assert "5.8" in text
    parsed = yaml.safe_load(text)
    assert parsed is None or isinstance(parsed, dict)


def test_rejects_unknown_retrieval_mode():
    data = _minimal_data() | {"retrieval": {"mode": "bm25"}}
    with pytest.raises(ValidationError) as exc_info:
        PipelineConfig.model_validate(data)
    error = exc_info.value.errors()[0]
    assert error["type"] == "literal_error"
    assert error["loc"] == ("retrieval", "mode")


def test_backend_literal_includes_control_bm25_and_rejects_others():
    for backend in ("pgvector", "qdrant", "control_bm25"):
        mode = "sparse" if backend == "control_bm25" else "hybrid"
        data = _minimal_data() | {"backend": backend, "retrieval": {"mode": mode}}
        cfg = PipelineConfig.model_validate(data)
        assert cfg.backend == backend
    with pytest.raises(ValidationError) as exc_info:
        PipelineConfig.model_validate(_minimal_data() | {"backend": "elasticsearch"})
    assert exc_info.value.errors()[0]["type"] == "literal_error"


@pytest.mark.parametrize("mode", ["dense", "hybrid"])
def test_control_bm25_backend_rejects_non_sparse_modes(mode: str):
    data = _minimal_data() | {"backend": "control_bm25", "retrieval": {"mode": mode}}
    with pytest.raises(ValidationError, match="control_bm25"):
        PipelineConfig.model_validate(data)


def test_control_bm25_backend_accepts_sparse_mode():
    data = _minimal_data() | {"backend": "control_bm25", "retrieval": {"mode": "sparse"}}
    cfg = PipelineConfig.model_validate(data)
    assert cfg.backend == "control_bm25"
    assert cfg.retrieval.mode == "sparse"


def test_defaults_are_filled_when_fields_are_omitted():
    cfg = PipelineConfig.model_validate(_minimal_data())
    assert cfg.seed == 42
    assert cfg.final_k == 10
    assert cfg.retrieval.k_dense == 50
    assert cfg.retrieval.k_sparse == 50
    assert cfg.retrieval.rrf_k == 60
    assert cfg.rerank.top_n == 50
    assert cfg.generation.enabled is True
    assert cfg.generation.context_k == 5
    assert cfg.generation.temperature == 0.0


def test_rejects_unknown_top_level_key():
    with pytest.raises(ValidationError) as exc_info:
        PipelineConfig.model_validate(_minimal_data() | {"bogus_key": 1})
    assert exc_info.value.errors()[0]["type"] == "extra_forbidden"


@pytest.mark.parametrize("section", ["chunker", "embedder", "retrieval", "rerank", "generation"])
def test_rejects_unknown_key_in_any_section(section: str):
    data = _minimal_data()
    section_data = data[section]
    assert isinstance(section_data, dict)
    data[section] = {**section_data, "bogus_key": 1}
    with pytest.raises(ValidationError) as exc_info:
        PipelineConfig.model_validate(data)
    assert exc_info.value.errors()[0]["type"] == "extra_forbidden"


def test_embedder_rejects_unspecified_fields():
    with pytest.raises(ValidationError) as exc_info:
        EmbedderCfg(name="BAAI/bge-small-en-v1.5", dim=384)
    assert exc_info.value.errors()[0]["type"] == "extra_forbidden"


@pytest.mark.parametrize(
    ("chunker_type", "kwargs"),
    [
        ("fixed", {"size_tokens": 256, "overlap_tokens": 32}),
        ("recursive_md", {"size_tokens": 512, "overlap_tokens": 64}),
        ("semantic", {}),
    ],
)
def test_chunker_accepts_spec_section_5_2_types(chunker_type: str, kwargs: dict[str, int]):
    cfg = ChunkerCfg(type=chunker_type, **kwargs)
    assert cfg.type == chunker_type


def test_chunker_rejects_unknown_type():
    with pytest.raises(ValidationError, match="sliding_window"):
        ChunkerCfg(type="sliding_window", size_tokens=512, overlap_tokens=64)


@pytest.mark.parametrize("chunker_type", ["fixed", "recursive_md"])
def test_chunker_requires_window_params_for_fixed_types(chunker_type: str):
    with pytest.raises(ValidationError, match="requires size_tokens and overlap_tokens"):
        ChunkerCfg(type=chunker_type)


def test_semantic_chunker_needs_no_window_params_and_has_documented_bounds():
    cfg = ChunkerCfg(type="semantic")
    assert cfg.size_tokens is None
    assert cfg.overlap_tokens is None
    assert cfg.min_tokens == 256
    assert cfg.max_tokens == 512
    assert cfg.threshold == 0.5


def test_semantic_bounds_must_increase_when_both_given():
    ChunkerCfg(type="semantic", min_tokens=128, max_tokens=512)
    ChunkerCfg(type="semantic", min_tokens=512, max_tokens=None)
    with pytest.raises(ValidationError, match="min_tokens"):
        ChunkerCfg(type="semantic", min_tokens=512, max_tokens=256)
    with pytest.raises(ValidationError, match="min_tokens"):
        ChunkerCfg(type="semantic", min_tokens=256, max_tokens=256)


@pytest.mark.parametrize("chunker_type", ["fixed", "recursive_md"])
@pytest.mark.parametrize("threshold", [0.5, 0.7])
def test_threshold_rejected_when_set_on_non_semantic_chunker(chunker_type: str, threshold: float):
    with pytest.raises(ValidationError, match="threshold applies only to chunker type 'semantic'"):
        ChunkerCfg(type=chunker_type, size_tokens=256, overlap_tokens=32, threshold=threshold)


def test_chunker_semantic_threshold_default_is_configurable():
    assert ChunkerCfg(type="semantic").threshold == 0.5
    assert ChunkerCfg(type="semantic", threshold=0.7).threshold == 0.7


def test_rerank_none_accepts_missing_model():
    cfg = RerankCfg(type="none")
    assert cfg.model is None
    assert cfg.top_n == 50


def test_rerank_cross_encoder_requires_model():
    with pytest.raises(ValidationError, match="rerank.model is required"):
        RerankCfg(type="cross_encoder")
    assert RerankCfg(type="cross_encoder", model="BAAI/bge-reranker-base").model is not None


def test_config_id_is_twelve_lowercase_hex_chars():
    cfg = _load(EXAMPLE_CONFIG_PATH)
    assert re.fullmatch(r"[0-9a-f]{12}", cfg.config_id)


def test_config_id_stable_across_dict_ordering():
    forward = {
        "corpus": "fastapi_docs",
        "chunker": {"type": "recursive_md", "size_tokens": 512, "overlap_tokens": 64},
        "embedder": {"name": "BAAI/bge-small-en-v1.5"},
        "backend": "pgvector",
        "retrieval": {"mode": "hybrid", "k_dense": 50, "k_sparse": 50, "rrf_k": 60},
        "rerank": {"type": "cross_encoder", "model": "BAAI/bge-reranker-base", "top_n": 50},
        "final_k": 10,
        "generation": {"enabled": True, "context_k": 5, "temperature": 0.0},
        "seed": 42,
    }
    reordered = dict(reversed(list(forward.items())))
    reordered["retrieval"] = {"rrf_k": 60, "mode": "hybrid", "k_sparse": 50, "k_dense": 50}
    assert (
        PipelineConfig.model_validate(forward).config_id
        == PipelineConfig.model_validate(reordered).config_id
    )


def test_config_id_stable_across_keyword_order():
    chunker = ChunkerCfg(type="fixed", size_tokens=256, overlap_tokens=32)
    embedder = EmbedderCfg(name="BAAI/bge-small-en-v1.5")
    retrieval = RetrievalCfg(mode="sparse")
    rerank = RerankCfg(type="none")
    generation = GenerationCfg()
    first = PipelineConfig(
        corpus="fastapi_docs",
        chunker=chunker,
        embedder=embedder,
        backend="pgvector",
        retrieval=retrieval,
        rerank=rerank,
        generation=generation,
        seed=42,
        final_k=10,
    )
    second = PipelineConfig(
        final_k=10,
        seed=42,
        generation=generation,
        rerank=rerank,
        retrieval=retrieval,
        embedder=embedder,
        chunker=chunker,
        backend="pgvector",
        corpus="fastapi_docs",
    )
    assert first.config_id == second.config_id


def test_config_id_unchanged_when_defaults_are_spelled_out_explicitly():
    implicit = PipelineConfig.model_validate(_minimal_data())
    spelled_out = PipelineConfig.model_validate(
        _minimal_data()
        | {
            "chunker": {
                "type": "recursive_md",
                "size_tokens": 512,
                "overlap_tokens": 64,
                "min_tokens": 256,
                "max_tokens": 512,
            },
            "retrieval": {"mode": "hybrid", "k_dense": 50, "k_sparse": 50, "rrf_k": 60},
            "rerank": {"type": "none", "model": None, "top_n": 50},
            "generation": {"enabled": True, "context_k": 5, "temperature": 0.0},
            "final_k": 10,
            "seed": 42,
        }
    )
    assert implicit.config_id == spelled_out.config_id


def test_config_id_changes_when_any_value_changes():
    base = PipelineConfig.model_validate(_minimal_data()).config_id
    variants = [
        _minimal_data() | {"seed": 43},
        _minimal_data() | {"final_k": 9},
        _minimal_data() | {"corpus": "other_docs"},
        _minimal_data() | {"chunker": {"type": "fixed", "size_tokens": 256, "overlap_tokens": 32}},
        _minimal_data() | {"retrieval": {"mode": "hybrid", "k_dense": 30}},
        _minimal_data() | {"embedder": {"name": "sentence-transformers/all-MiniLM-L6-v2"}},
        _minimal_data() | {"retrieval": {"mode": "dense"}},
        _minimal_data() | {"retrieval": {"mode": "hybrid", "rrf_k": 30}},
        _minimal_data() | {"rerank": {"type": "none", "top_n": 20}},
        _minimal_data() | {"generation": {"temperature": 0.7}},
    ]
    ids = [PipelineConfig.model_validate(variant).config_id for variant in variants]
    assert base not in ids
    assert len(set(ids)) == len(ids)


def test_minimal_semantic_pipeline_config_validates_and_gets_config_id():
    implicit = PipelineConfig.model_validate(_minimal_data() | {"chunker": {"type": "semantic"}})
    explicit = PipelineConfig.model_validate(
        _minimal_data() | {"chunker": {"type": "semantic", "threshold": 0.5}}
    )
    assert implicit.chunker.threshold == 0.5
    assert re.fullmatch(r"[0-9a-f]{12}", implicit.config_id)
    assert implicit.config_id == explicit.config_id
    tuned = PipelineConfig.model_validate(
        _minimal_data() | {"chunker": {"type": "semantic", "threshold": 0.9}}
    )
    assert tuned.config_id != implicit.config_id


def test_example_config_round_trips_through_model_dump():
    cfg = _load(EXAMPLE_CONFIG_PATH)
    assert PipelineConfig.model_validate(cfg.model_dump()).config_id == cfg.config_id
