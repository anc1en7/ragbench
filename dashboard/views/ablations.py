"""Ablations: what changing one factor does, with paired-bootstrap CIs."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
import ui
from data import ablation_table

FACTORS = {
    "rerank": "Reranker",
    "embedder": "Embedder",
    "chunker": "Chunker",
    "retrieval": "Retrieval mode",
    "backend": "Backend",
}

ui.setup(
    "Ablations",
    "Every pair of dev runs that differ in exactly one factor. Each row shows how much the "
    "better setting beat the worse one, with a 95% paired-bootstrap interval over the same "
    "47 questions. Only intervals that exclude zero are real differences.",
)

metric = (
    st.segmented_control(
        "Metric", ["ndcg@10", "mrr@10", "recall@1"], default="ndcg@10", label_visibility="collapsed"
    )
    or "ndcg@10"
)
abl = ablation_table("dev", metric)
if abl.empty:
    st.info("No single-factor pairs in the dev runs.")
    st.stop()

present = [f for f in FACTORS if f in set(abl["factor"])]
cols = st.columns(len(present), gap="small")
for col, factor in zip(cols, present, strict=True):
    part = abl[abl["factor"] == factor]
    with col:
        ui.tile(
            FACTORS[factor],
            f"{int(part['significant'].sum())} / {len(part)}",
            f"pairs differ · largest Δ {part['delta'].max():+.3f}",
        )

st.write("")
default = max(present, key=lambda f: int(abl[abl["factor"] == f]["significant"].sum()))
factor = (
    st.segmented_control(
        "Factor", present, default=default, format_func=FACTORS.get, label_visibility="collapsed"
    )
    or default
)
view = abl[abl["factor"] == factor].sort_values("delta")

fig = go.Figure()
for significant, name, color in (
    (False, "no detectable difference", ui.MUTED_MARK),
    (True, "CI excludes 0", ui.BLUE),
):
    part = view[view["significant"] == significant]
    if part.empty:
        continue
    fig.add_trace(
        go.Scatter(
            x=part["delta"],
            y=part["change"] + "  ·  " + part["held fixed"],
            mode="markers",
            name=name,
            marker={"color": color, "size": 10, "line": {"color": ui.SURFACE, "width": 2}},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": part["hi"] - part["delta"],
                "arrayminus": part["delta"] - part["lo"],
                "color": color,
                "thickness": 2,
                "width": 0,
            },
            customdata=part[["lo", "hi", "latency_delta_ms"]],
            hovertemplate=(
                "%{y}<br>Δ %{x:+.3f} [%{customdata[0]:+.3f}, %{customdata[1]:+.3f}]"
                "<br>Δ p95 latency %{customdata[2]:+.0f} ms<extra></extra>"
            ),
        )
    )
fig.add_vline(x=0, line={"color": ui.INK_3, "width": 1})
fig.update_xaxes(title=f"Δ {metric}: better setting minus worse setting")
fig.update_yaxes(
    categoryorder="array", categoryarray=list(view["change"] + "  ·  " + view["held fixed"])
)
st.plotly_chart(
    ui.style(fig, max(240, 30 * len(view) + 110)),
    theme=None,
    config={"displayModeBar": False},
)
st.caption("Row label: the change (better vs worse) · what was held fixed.")

with st.expander("Table view"):
    st.dataframe(
        view.sort_values("delta", ascending=False).drop(columns="factor"),
        hide_index=True,
        width="stretch",
        column_config={
            "change": "Change (better vs worse)",
            "held fixed": "Held fixed",
            "delta": st.column_config.NumberColumn("Δ", format="%+.3f"),
            "lo": st.column_config.NumberColumn("CI low", format="%+.3f"),
            "hi": st.column_config.NumberColumn("CI high", format="%+.3f"),
            "significant": "Detectable",
            "latency_delta_ms": st.column_config.NumberColumn("Δ p95 ms", format="%+.0f"),
        },
    )
