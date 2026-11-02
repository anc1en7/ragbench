"""Unit tests for the run-result schema and writers (Task 3.4) — offline, tmp dirs only."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ragbench.config import PipelineConfig
from ragbench.eval.results import (
    SUMMARY_COLUMNS,
    CorpusInfo,
    GoldsetInfo,
    IndexStats,
    MetricCI,
    QueryResult,
    RetrievedChunk,
    RunResult,
    StageLatency,
    corpus_info_from_lock,
    load_run,
    package_versions,
    run_path,
    upsert_summary,
    write_run,
)
from ragbench.eval.retrieval_metrics import METRIC_NAMES

CFG = PipelineConfig.model_validate(
    {
        "corpus": "mini",
        "chunker": {"type": "recursive_md", "size_tokens": 512, "overlap_tokens": 64},
        "embedder": {"name": "BAAI/bge-small-en-v1.5"},
        "backend": "pgvector",
        "retrieval": {"mode": "dense"},
        "rerank": {"type": "none"},
        "generation": {"enabled": False},
    }
)


def _result(split: str = "dev", recall5: float = 0.5, cfg: PipelineConfig = CFG) -> RunResult:
    metrics = {name: MetricCI(mean=0.5, ci_lo=0.25, ci_hi=0.75) for name in METRIC_NAMES}
    metrics["recall@5"] = MetricCI(mean=recall5, ci_lo=recall5 - 0.1, ci_hi=recall5 + 0.1)
    return RunResult(
        config_id=cfg.config_id,
        config=cfg,
        seed=42,
        split=split,
        corpus=CorpusInfo(
            name="mini",
            repo="https://github.com/fastapi/fastapi",
            tag="0.162.0",
            commit_sha="7a0a3f0b",
            sha256_of_files="abc",
        ),
        goldset=GoldsetInfo(n_questions=2, all_verified=True, provisional=False),
        package_versions={"numpy": "2.5.4"},
        hardware_ref="results/hardware.json",
        min_overlap_frac=0.2,
        bootstrap_resamples=1000,
        bootstrap_seed=42,
        metrics=metrics,
        latency_ms={"total": StageLatency(p50=12.0, p95=20.0)},
        latency_passes=3,
        index=IndexStats(
            docs=30,
            chunks=120,
            embed_seconds=4.0,
            index_seconds=1.0,
            wall_seconds=5.5,
            peak_rss_bytes=1_000_000,
            chunks_per_sec=30.0,
            size_bytes=2_000_000,
        ),
        ann_sanity_recall=1.0,
        per_query=[
            QueryResult(
                qid="q1",
                question_type="factual",
                relevance=[False, True],
                metrics=dict.fromkeys(METRIC_NAMES, 1.0),
                hits=[
                    RetrievedChunk(
                        chunk_id="c1", doc_id="a.md", start=0, end=10, score=0.9, source="dense"
                    )
                ],
            )
        ],
        created_at="2026-11-02T00:00:00+00:00",
    )


def test_run_result_round_trips_through_json_file(tmp_path: Path):
    result = _result()
    path = write_run(result, tmp_path)

    assert path == tmp_path / "runs" / f"{CFG.config_id}__dev.json"
    assert path == run_path(tmp_path, CFG.config_id, "dev")
    raw = json.loads(path.read_text(encoding="utf-8"))
    for key in (
        "config",
        "config_id",
        "seed",
        "corpus",
        "package_versions",
        "hardware_ref",
        "split",
        "metrics",
        "latency_ms",
        "index",
    ):
        assert key in raw
    loaded = load_run(path)
    assert loaded == result
    assert RunResult.model_validate_json(path.read_text(encoding="utf-8")) == result


def test_config_id_must_match_config():
    with pytest.raises(ValidationError, match="config_id"):
        RunResult.model_validate(_result().model_dump() | {"config_id": "deadbeef0000"})


def test_metrics_must_cover_every_metric_name():
    data = _result().model_dump()
    del data["metrics"]["ndcg@10"]
    with pytest.raises(ValidationError, match="ndcg@10"):
        RunResult.model_validate(data)


def test_split_is_dev_or_test_only():
    data = _result().model_dump() | {"split": "train"}
    with pytest.raises(ValidationError):
        RunResult.model_validate(data)


def test_ci_must_bracket_mean():
    with pytest.raises(ValidationError, match="ci_lo"):
        MetricCI(mean=0.1, ci_lo=0.2, ci_hi=0.3)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_summary_header_written_once_and_rerun_updates_in_place(tmp_path: Path):
    summary = upsert_summary(_result(recall5=0.5), tmp_path)
    upsert_summary(_result(recall5=0.7), tmp_path)  # same (config_id, split) -> replace

    text = summary.read_text(encoding="utf-8")
    assert text.splitlines()[0] == ",".join(SUMMARY_COLUMNS)
    assert text.count("config_id,") == 1
    rows = _rows(summary)
    assert len(rows) == 1
    assert rows[0]["config_id"] == CFG.config_id
    assert float(rows[0]["recall@5"]) == pytest.approx(0.7)
    assert float(rows[0]["recall@5_lo"]) == pytest.approx(0.6)


def test_summary_keeps_separate_rows_per_split_and_config(tmp_path: Path):
    other = CFG.model_copy(update={"backend": "qdrant"})
    upsert_summary(_result("dev"), tmp_path)
    upsert_summary(_result("test"), tmp_path)
    upsert_summary(_result("dev", cfg=other), tmp_path)
    upsert_summary(_result("dev"), tmp_path)

    rows = _rows(tmp_path / "summary.csv")
    keys = [(r["config_id"], r["split"]) for r in rows]
    assert keys == [
        (CFG.config_id, "dev"),
        (CFG.config_id, "test"),
        (other.config_id, "dev"),
    ]
    assert rows[2]["backend"] == "qdrant"
    assert rows[0]["mode"] == "dense"
    assert rows[0]["p95_total_ms"] == "20.0"


def test_corpus_info_from_lock(tmp_path: Path):
    lock = tmp_path / "corpus.lock"
    lock.write_text(
        json.dumps(
            {
                "repo": "https://github.com/fastapi/fastapi",
                "tag": "0.162.0",
                "commit_sha": "7a0a3f0b",
                "sha256_of_files": "abc",
                "license": "MIT",
            }
        ),
        encoding="utf-8",
    )
    info = corpus_info_from_lock(lock, name="fastapi_docs")
    assert info.commit_sha == "7a0a3f0b"
    assert info.tag == "0.162.0"
    assert info.name == "fastapi_docs"


def test_package_versions_reports_installed_pins():
    versions = package_versions()
    assert versions["numpy"]
    assert versions["pydantic"]
    assert "ragbench" in versions


def test_generation_block_round_trips_and_fills_summary_columns(tmp_path: Path):
    from ragbench.eval.results import GenerationMetrics, GenerationQuery, TokenStats

    gen = GenerationMetrics(
        generator_model="rag",
        judge_model="rag",
        prompt_version="gen-v1",
        ragas_version="0.4.3",
        ragas_embedder="BAAI/bge-small-en-v1.5",
        answer_relevancy_strictness=3,
        temperature=0.0,
        judge_temperature=0.0,
        context_k=5,
        subset_seed=42,
        n_questions=1,
        metrics={
            name: MetricCI(mean=0.8, ci_lo=0.7, ci_hi=0.9)
            for name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
        },
        n_valid={"faithfulness": 1},
        tokens_per_query=TokenStats(mean=900.0, p50=880.0, p95=1200.0),
        per_query=[
            GenerationQuery(
                qid="q1",
                answer="a",
                context_chunk_ids=["c1"],
                tokens=900,
                scores={"faithfulness": 0.8, "context_recall": None},
            )
        ],
        created_at="2026-11-02T00:00:00+00:00",
    )
    result = _result("test").model_copy(update={"generation": gen})
    path = write_run(result, tmp_path)
    assert load_run(path) == result  # None scores survive JSON (no NaN)

    upsert_summary(result, tmp_path)
    row = _rows(tmp_path / "summary.csv")[0]
    assert float(row["faithfulness"]) == pytest.approx(0.8)
    assert float(row["tokens_per_query"]) == pytest.approx(900.0)


def test_summary_generation_columns_blank_without_generation(tmp_path: Path):
    upsert_summary(_result(), tmp_path)
    row = _rows(tmp_path / "summary.csv")[0]
    assert row["faithfulness"] == "" and row["tokens_per_query"] == ""
