"""Unit tests for the FastAPI query service (Task 5.1).

Offline: the pipeline is the real in-process BM25 control over the mini fixture (no
docker, no model download), and backend health checks are injected stubs.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from ragbench.api.app import create_app
from ragbench.api.pipeline_cache import PipelineCache, build_and_index, discover_configs
from ragbench.api.schemas import MAX_QUESTION_CHARS, BackendStatus
from ragbench.config import PipelineConfig, load_pipeline_config

MINI_BM25 = {
    "corpus": "mini",
    "chunker": {"type": "recursive_md", "size_tokens": 256, "overlap_tokens": 32},
    "embedder": {"name": "BAAI/bge-small-en-v1.5"},
    "backend": "control_bm25",
    "retrieval": {"mode": "sparse", "k_dense": 50, "k_sparse": 50, "rrf_k": 60},
    "rerank": {"type": "none", "top_n": 50},
    "generation": {"enabled": False, "context_k": 5, "temperature": 0.0},
    "final_k": 5,
}


@pytest.fixture
def dirs(tmp_path: Path) -> tuple[Path, Path, str]:
    configs = tmp_path / "configs"
    results = tmp_path / "results"
    configs.mkdir()
    (results / "runs").mkdir(parents=True)
    path = configs / "mini_bm25.yaml"
    path.write_text(yaml.safe_dump(MINI_BM25), encoding="utf-8")
    (configs / "grid.yaml").write_text("stages: []\n", encoding="utf-8")  # not a pipeline
    with (results / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["config_id", "split", "ndcg@10"])
        writer.writerow(["abc123def456", "dev", "0.9"])
    return configs, results, load_pipeline_config(path).config_id


def _client(dirs: tuple[Path, Path, str], *, pg_ok: bool = True) -> TestClient:
    configs, results, _ = dirs
    app = create_app(
        configs_dir=configs,
        results_dir=results,
        health_checks={
            "pgvector": lambda: BackendStatus(ok=pg_ok, detail="stub"),
            "qdrant": lambda: BackendStatus(ok=True, detail="stub"),
        },
    )
    return TestClient(app)


def test_health_reports_backends(dirs: tuple[Path, Path, str]) -> None:
    body = _client(dirs).get("/health").json()
    assert body["status"] == "ok"
    assert set(body["backends"]) == {"pgvector", "qdrant"}
    degraded = _client(dirs, pg_ok=False).get("/health").json()
    assert degraded["status"] == "degraded"
    assert degraded["backends"]["pgvector"]["ok"] is False


def test_configs_lists_yaml_and_skips_non_pipeline_files(dirs: tuple[Path, Path, str]) -> None:
    _, _, config_id = dirs
    body = _client(dirs).get("/configs").json()
    assert [c["config_id"] for c in body] == [config_id]
    assert body[0]["backend"] == "control_bm25"
    assert body[0]["chunker"] == "recursive_md-256"


def test_query_returns_contexts_timings_and_trace_id(dirs: tuple[Path, Path, str]) -> None:
    _, _, config_id = dirs
    response = _client(dirs).post(
        "/query",
        json={"question": "How do I run FastAPI in a Docker container?", "config_id": config_id},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["config_id"] == config_id
    assert 1 <= len(body["contexts"]) <= 5
    assert [c["rank"] for c in body["contexts"]] == list(range(1, len(body["contexts"]) + 1))
    assert set(body["timings_ms"]) == {
        "embed_query",
        "dense",
        "sparse",
        "fusion",
        "rerank",
        "total",
    }
    assert len(body["trace_id"]) == 32
    assert body["answer"] is None


def test_query_too_long_is_422_with_clear_message(dirs: tuple[Path, Path, str]) -> None:
    _, _, config_id = dirs
    response = _client(dirs).post(
        "/query", json={"question": "x" * (MAX_QUESTION_CHARS + 1), "config_id": config_id}
    )
    assert response.status_code == 422
    assert f"at most {MAX_QUESTION_CHARS} characters" in response.text


def test_unknown_config_is_404_with_clear_message(dirs: tuple[Path, Path, str]) -> None:
    response = _client(dirs).post("/query", json={"question": "hi", "config_id": "nope"})
    assert response.status_code == 404
    assert "GET /configs" in response.json()["detail"]


def test_eval_summary_returns_rows(dirs: tuple[Path, Path, str]) -> None:
    body = _client(dirs).get("/eval/summary").json()
    assert body == [{"config_id": "abc123def456", "split": "dev", "ndcg@10": "0.9"}]


def test_eval_summary_404_without_results(tmp_path: Path) -> None:
    app = create_app(configs_dir=tmp_path, results_dir=tmp_path, health_checks={})
    assert TestClient(app).get("/eval/summary").status_code == 404


def test_openapi_docs_served(dirs: tuple[Path, Path, str]) -> None:
    client = _client(dirs)
    assert client.get("/docs").status_code == 200
    assert "/query" in client.get("/openapi.json").json()["paths"]


def test_discover_configs_reads_run_files(tmp_path: Path) -> None:
    cfg = PipelineConfig.model_validate(MINI_BM25)
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / f"{cfg.config_id}__dev.json").write_text(
        '{"config": ' + cfg.model_dump_json() + "}", encoding="utf-8"
    )
    (runs / "broken.json").write_text("{not json", encoding="utf-8")
    found = discover_configs(tmp_path / "missing", runs)
    assert list(found) == [cfg.config_id]


def test_pipeline_cache_is_lru() -> None:
    built: list[str] = []
    cfgs = [PipelineConfig.model_validate({**MINI_BM25, "final_k": k}) for k in (1, 2, 3)]

    def builder(cfg: PipelineConfig) -> object:
        built.append(cfg.config_id)
        return object()

    cache = PipelineCache(builder, max_size=2)  # type: ignore[arg-type]
    first = cache.get(cfgs[0])
    assert cache.get(cfgs[0]) is first
    cache.get(cfgs[1])
    cache.get(cfgs[2])  # evicts cfgs[0]
    assert cfgs[0].config_id not in cache
    assert len(cache) == 2
    assert built == [c.config_id for c in cfgs]


def test_build_and_index_indexes_empty_namespace() -> None:
    cfg = PipelineConfig.model_validate(MINI_BM25)
    pipeline = build_and_index(cfg)
    assert pipeline.store.size_bytes(cfg.config_id) > 0
