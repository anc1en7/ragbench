"""Ablations: single-factor effects with paired-bootstrap CIs on dev (Task 5.4)."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
from data import INK_2, MODE_COLORS, ablation_table, style_figure

st.set_page_config(page_title="Ablations · RAGBench", layout="wide")
st.title("Single-factor ablations")
st.caption(
    "Every pair of dev runs that differ in exactly one factor. Δ is better minus worse, "
    "with a 95% paired-bootstrap CI over the same 47 questions. A CI that excludes 0 is "
    "a detectable difference."
)

controls = st.columns([1, 1, 4])
metric = controls[0].selectbox("Metric", ["ndcg@10", "mrr@10", "recall@1", "recall@5"])
only_sig = controls[1].toggle("Only detectable differences", value=False)
table = ablation_table("dev", metric)
if table.empty:
    st.info("No single-factor pairs in the dev runs.")
    st.stop()

summary_rows = (
    table.groupby("factor")
    .agg(pairs=("delta", "size"), detectable=("significant", "sum"), max_delta=("delta", "max"))
    .reset_index()
)
st.dataframe(summary_rows, hide_index=True)

factors = sorted(table["factor"].unique())
most_detectable = summary_rows.sort_values("detectable", ascending=False)["factor"].iloc[0]
factor = st.radio("Factor", factors, index=factors.index(most_detectable), horizontal=True)
view = table[table["factor"] == factor].sort_values("delta")
if only_sig:
    view = view[view["significant"]]
if view.empty:
    st.info("No detectable differences for this factor.")
    st.stop()

labels = [f"{row.better}  vs  {row.worse}" for row in view.itertuples()]
fig = go.Figure(
    go.Scatter(
        x=view["delta"],
        y=labels,
        mode="markers",
        marker={"color": MODE_COLORS["dense"], "size": 10},
        error_x={
            "type": "data",
            "symmetric": False,
            "array": view["hi"] - view["delta"],
            "arrayminus": view["delta"] - view["lo"],
            "color": MODE_COLORS["dense"],
            "thickness": 2,
            "width": 0,
        },
        customdata=view[["lo", "hi", "latency_delta_ms"]],
        hovertemplate=(
            "%{y}<br>Δ %{x:+.3f} [%{customdata[0]:+.3f}, %{customdata[1]:+.3f}]"
            "<br>Δ p95 latency %{customdata[2]:+.1f} ms<extra></extra>"
        ),
    )
)
fig.add_vline(x=0, line={"color": INK_2, "width": 1})
fig.update_xaxes(title=f"Δ {metric} (better − worse)")
fig.update_yaxes(showticklabels=False, title=None)
st.plotly_chart(style_figure(fig, max(260, 34 * len(view) + 90)), theme=None)
st.caption("Hover a row for the configs; the table below lists the same pairs.")
st.dataframe(
    view.sort_values("delta", ascending=False).drop(columns="factor"),
    hide_index=True,
    width="stretch",
    column_config={
        "delta": st.column_config.NumberColumn("Δ", format="%+.3f"),
        "lo": st.column_config.NumberColumn("CI lo", format="%+.3f"),
        "hi": st.column_config.NumberColumn("CI hi", format="%+.3f"),
        "latency_delta_ms": st.column_config.NumberColumn("Δ p95 ms", format="%+.1f"),
    },
)
