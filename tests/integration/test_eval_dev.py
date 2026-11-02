"""Integration test for ``ragbench eval`` on the mini fixture's dev split (Task 3.5).

Requires the docker compose stack (``make up``). Evaluates the baseline dense config
(``configs/baseline_dense.yaml`` with ``corpus: mini``) on both vector backends through
the CLI, then asserts metrics + CIs are present, latency covers every stage, the run JSON
is written and schema-valid, the ANN sanity recall is >= 0.98, and the summary row and
baseline file exist. Embeds with the real ``BAAI/bge-small-en-v1.5`` (cached in
``.hf_cache``); no reranker download.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import psycopg
import pytest
import yaml
from typer.testing import CliRunner

from ragbench.cli import app
from ragbench.config import load_pipeline_config
from ragbench.eval.results import load_run, run_path
from ragbench.eval.retrieval_metrics import METRIC_NAMES
from ragbench.pipeline import TIMING_KEYS
from ragbench.stores.pgvector_store import DEFAULT_PG_DSN
from ragbench.stores.qdrant_store import DEFAULT_QDRANT_URL

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]


def _reachable() -> list[str]:
    missing: list[str] = []
    try:
        with psycopg.connect(DEFAULT_PG_DSN, connect_timeout=3) as conn:
            conn.execute("SELECT 1")
    except psycopg.OperationalError:
        missing.append("Postgres")
    try:
        with urlopen(f"{DEFAULT_QDRANT_URL}/collections", timeout=3) as resp:
            if resp.status != 200:
                missing.append("Qdrant")
    except (URLError, TimeoutError, OSError):
        missing.append("Qdrant")
    return missing


@pytest.fixture(autouse=True)
def _require_backends() -> None:
    missing = _reachable()
    if missing:
        pytest.fail(f"{' and '.join(missing)} unreachable — run `make up` first", pytrace=False)


def _mini_config(tmp_path: Path, backend: str) -> Path:
    data = yaml.safe_load((ROOT / "configs" / "baseline_dense.yaml").read_text(encoding="utf-8"))
    data["corpus"] = "mini"
    data["backend"] = backend
    path = tmp_path / f"baseline_mini_{backend}.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.mark.parametrize("backend", ["pgvector", "qdrant"])
def test_eval_cli_baseline_dense_on_mini_dev(tmp_path: Path, backend: str) -> None:
    cfg_path = _mini_config(tmp_path, backend)
    cfg = load_pipeline_config(cfg_path)
    results_dir = tmp_path / "results"
    baseline = tmp_path / "eval" / "baseline.json"

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "--config",
            str(cfg_path),
            "--split",
            "dev",
            "--results-dir",
            str(results_dir),
            "--update-baseline",
            "--baseline",
            str(baseline),
        ],
        env={"COLUMNS": "160"},
    )
    assert result.exit_code == 0, result.output
    for name in METRIC_NAMES:
        assert name in result.output
    assert "p95" in result.output

    run = load_run(run_path(results_dir, cfg.config_id, "dev"))
    assert run.config == cfg and run.split == "dev"
    assert run.goldset.n_questions == 9 and not run.goldset.provisional
    assert set(run.metrics) == set(METRIC_NAMES)
    for metric in run.metrics.values():
        assert 0.0 <= metric.ci_lo <= metric.mean <= metric.ci_hi <= 1.0
    assert run.metrics["recall@10"].mean > 0.0
    assert set(run.latency_ms) == set(TIMING_KEYS)
    assert run.latency_ms["total"].p50 > 0.0
    assert run.latency_ms["dense"].p95 > 0.0
    assert run.index.chunks_per_sec is not None and run.index.size_bytes > 0
    assert run.ann_sanity_recall is not None and run.ann_sanity_recall >= 0.98

    with (results_dir / "summary.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [(r["config_id"], r["split"], r["backend"]) for r in rows] == [
        (cfg.config_id, "dev", backend)
    ]
    assert json.loads(baseline.read_text(encoding="utf-8"))["config_id"] == cfg.config_id
