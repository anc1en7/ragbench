"""Shared data loading for the replay dashboard (Task 5.4).

Replay mode: everything comes from committed files (``results/summary.csv``,
``results/runs/*.json``, ``data/goldset/verified.jsonl``). No vector database, model or
LLM is touched. Only light ragbench modules (pydantic schemas and bootstrap stats) are
imported, so the dashboard does not need torch.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # run without installing the package (deploys)
    sys.path.insert(0, str(ROOT / "src"))

from ragbench.eval.results import RunResult, load_run  # noqa: E402
from ragbench.sweep.select import ablations, describe  # noqa: E402

RESULTS = Path(os.environ.get("RAGBENCH_RESULTS_DIR", ROOT / "results"))
METRICS = ["ndcg@10", "mrr@10", "recall@1", "recall@3", "recall@5", "recall@10"]
GEN_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# Reference categorical palette (dataviz skill), first three slots: validated all-pairs
# for scatter. Fixed by mode so color follows the entity, never its rank.
MODE_COLORS = {"dense": "#2a78d6", "hybrid": "#eb6834", "sparse": "#1baf7a"}
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


@st.cache_data
def summary() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "summary.csv")
    df = df[df["corpus"] == "fastapi_docs"].copy()
    df["mode_label"] = df["mode"].where(df["backend"] != "control_bm25", "sparse")
    df["rerank_label"] = df.apply(
        lambda r: "none" if r["rerank"] == "none" else f"cross-encoder@{r['rerank_top_n']}",
        axis=1,
    )
    df["chunker_label"] = df.apply(
        lambda r: (
            r["chunker"] if r["chunker"] == "semantic" else f"{r['chunker']}-{int(r['chunk_size'])}"
        ),
        axis=1,
    )
    df["embedder_short"] = df["embedder"].str.split("/").str[-1]
    return df


@st.cache_data
def run_paths() -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for path in sorted((RESULTS / "runs").glob("*__*.json")):
        config_id, split = path.stem.split("__", 1)
        out[(config_id, split)] = str(path)
    return out


@st.cache_resource
def run(config_id: str, split: str) -> RunResult:
    return load_run(Path(run_paths()[(config_id, split)]))


@st.cache_resource
def runs(split: str) -> list[RunResult]:
    return [run(cid, s) for (cid, s) in run_paths() if s == split and _is_main(cid, s)]


def _is_main(config_id: str, split: str) -> bool:
    raw = json.loads(Path(run_paths()[(config_id, split)]).read_text(encoding="utf-8"))
    return bool(raw["config"]["corpus"] == "fastapi_docs")


def label(result: RunResult) -> str:
    return describe(result.config)


@st.cache_data
def ablation_table(split: str, metric: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ab in ablations(runs(split), metric, seed=42):
        rows.append(
            {
                "factor": ab.factor,
                "better": label(ab.a),
                "worse": label(ab.b),
                "delta": ab.delta,
                "lo": ab.lo,
                "hi": ab.hi,
                "significant": ab.significant,
                "latency_delta_ms": ab.latency_delta_ms,
            }
        )
    return pd.DataFrame(rows)


@st.cache_data
def gold_questions() -> dict[str, dict[str, Any]]:
    path = ROOT / "data" / "goldset" / "verified.jsonl"
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            q = json.loads(line)
            out[q["qid"]] = q
    return out


@st.cache_data
def documents() -> dict[str, str] | None:
    """Full document texts if the corpus has been ingested locally, else ``None``."""
    path = ROOT / "data" / "documents.jsonl"
    if not path.exists():
        return None
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            doc = json.loads(line)
            out[doc["doc_id"]] = doc["text"]
    return out


def ci_text(mean: float, lo: float, hi: float) -> str:
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"


def style_figure(fig: Any, height: int = 480) -> Any:
    """Recessive axes and grid, text in ink tokens, consistent surface."""
    fig.update_layout(
        height=height,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"color": INK, "size": 13},
        margin={"l": 60, "r": 20, "t": 30, "b": 50},
        hoverlabel={"bgcolor": "white", "font": {"color": INK}},
        legend={"orientation": "h", "y": 1.08, "x": 0, "font": {"color": INK_2}},
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont={"color": INK_2})
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont={"color": INK_2})
    return fig
