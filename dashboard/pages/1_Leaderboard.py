"""Leaderboard: sortable, filterable table of every run (Task 5.4)."""

from __future__ import annotations

import streamlit as st
from data import METRICS, summary

st.set_page_config(page_title="Leaderboard · RAGBench", layout="wide")
st.title("Leaderboard")

df = summary()
filters = st.columns(6)
split = filters[0].selectbox("Split", ["test", "dev"], help="test = dev-selected finalists")
view = df[df["split"] == split]
choices = {
    "chunker_label": filters[1].multiselect("Chunker", sorted(view["chunker_label"].unique())),
    "embedder_short": filters[2].multiselect("Embedder", sorted(view["embedder_short"].unique())),
    "mode_label": filters[3].multiselect("Retrieval", sorted(view["mode_label"].unique())),
    "rerank_label": filters[4].multiselect("Rerank", sorted(view["rerank_label"].unique())),
    "backend": filters[5].multiselect("Backend", sorted(view["backend"].unique())),
}
for column, selected in choices.items():
    if selected:
        view = view[view[column].isin(selected)]

table = view.assign(
    **{
        f"{m} (95% CI)": view.apply(
            lambda r, m=m: f"{r[m]:.3f} [{r[m + '_lo']:.3f}, {r[m + '_hi']:.3f}]", axis=1
        )
        for m in ("ndcg@10", "mrr@10", "recall@5")
    }
)
columns = [
    "config_id",
    "chunker_label",
    "embedder_short",
    "backend",
    "mode_label",
    "rerank_label",
    "ndcg@10 (95% CI)",
    "mrr@10 (95% CI)",
    "recall@5 (95% CI)",
    *METRICS,
    "p95_total_ms",
    "n_questions",
]
st.caption(
    f"{len(table)} runs. Click a column header to sort; the numeric metric columns sort "
    "correctly, the CI text columns are for reading."
)
st.dataframe(
    table.sort_values("ndcg@10", ascending=False)[columns],
    hide_index=True,
    width="stretch",
    column_config={
        "chunker_label": "chunker",
        "embedder_short": "embedder",
        "mode_label": "retrieval",
        "rerank_label": "rerank",
        "p95_total_ms": st.column_config.NumberColumn("p95 ms", format="%.1f"),
        **{m: st.column_config.NumberColumn(m, format="%.3f") for m in METRICS},
    },
)
if split == "dev":
    st.info(
        "Dev numbers were used to select finalists; treat them as selection evidence, not "
        "as headline results. pgvector `sparse`/`hybrid` rows are affected by the FTS "
        "AND-semantics issue (see Methodology)."
    )
