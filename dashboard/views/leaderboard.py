"""Leaderboard: every run, ranked, with the few columns that matter up front."""

from __future__ import annotations

import streamlit as st
import ui
from data import METRICS, table

ui.setup(
    "Leaderboard",
    "Ranked by nDCG@10. The test split holds the 5 dev-selected finalists (the headline "
    "numbers); the dev split holds all 33 swept pipelines (selection evidence only).",
)

split_name = st.segmented_control(
    "Split", ["Test finalists", "Dev sweep"], default="Test finalists", label_visibility="collapsed"
)
split = "dev" if split_name == "Dev sweep" else "test"
df = table(split)

f1, f2, f3, f4 = st.columns(4, gap="small")
filters = {
    "mode": f1.multiselect("Retrieval", sorted(df["mode"].unique()), placeholder="All"),
    "rerank": f2.multiselect("Rerank", sorted(df["rerank"].unique()), placeholder="All"),
    "embedder": f3.multiselect("Embedder", sorted(df["embedder"].unique()), placeholder="All"),
    "backend": f4.multiselect("Backend", sorted(df["backend"].unique()), placeholder="All"),
}
view = df
for column, chosen in filters.items():
    if chosen:
        view = view[view[column].isin(chosen)]

view = view.assign(ci=view.apply(lambda r: f"{r['ndcg@10_lo']:.3f}–{r['ndcg@10_hi']:.3f}", axis=1))
show_all = st.toggle("Show every metric", value=False)
columns = [
    "rank",
    "pipeline",
    "chunker",
    "embedder",
    "ndcg@10",
    "ci",
    "recall@5",
    "mrr@10",
    "p95_ms",
]
if show_all:
    columns += ["recall@1", "recall@3", "recall@10", "config_id"]

st.dataframe(
    view[columns],
    hide_index=True,
    width="stretch",
    height=min(35 * (len(view) + 1) + 3, 820),
    column_config={
        "rank": st.column_config.NumberColumn("#", width="small"),
        "pipeline": st.column_config.TextColumn("Pipeline", width="medium"),
        "chunker": "Chunks",
        "embedder": "Embedder",
        "ndcg@10": st.column_config.ProgressColumn(
            "nDCG@10", format="%.3f", min_value=0.0, max_value=1.0, width="medium"
        ),
        "ci": "95% CI",
        "p95_ms": st.column_config.NumberColumn("p95 ms", format="%.0f"),
        **{
            m: st.column_config.NumberColumn(
                m.replace("recall", "Recall").replace("mrr", "MRR"), format="%.3f"
            )
            for m in METRICS
            if m != "ndcg@10"
        },
        "config_id": "Config id",
    },
)
st.caption(
    f"{len(view)} of {len(df)} runs shown. Click a header to sort. "
    + (
        "pgvector sparse/hybrid rows are affected by the full-text-search issue described "
        "under Methodology (pgvector hybrid equals dense)."
        if split == "dev"
        else "All finalists share their chunker and embedder."
    )
)
