"""Pareto: retrieval quality vs p95 latency, frontier highlighted (Task 5.4)."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
from data import INK_2, METRICS, MODE_COLORS, SURFACE, style_figure, summary

st.set_page_config(page_title="Pareto · RAGBench", layout="wide")
st.title("Quality vs latency")

df = summary()
controls = st.columns([1, 1, 1, 3])
split = controls[0].selectbox("Split", ["dev", "test"])
metric = controls[1].selectbox("Metric", METRICS, index=METRICS.index("ndcg@10"))
log_x = controls[2].toggle("Log latency axis", value=True)
view = df[df["split"] == split].copy()

# Frontier: configs that no other config beats on both quality (higher) and latency (lower).
view = view.sort_values(["p95_total_ms", metric], ascending=[True, False])
frontier_ids: list[str] = []
best = -1.0
for _, row in view.iterrows():
    if row[metric] > best:
        frontier_ids.append(row["config_id"])
        best = row[metric]
frontier = view[view["config_id"].isin(frontier_ids)].sort_values("p95_total_ms")

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=frontier["p95_total_ms"],
        y=frontier[metric],
        mode="lines",
        line={"color": INK_2, "width": 2, "dash": "dot", "shape": "hv"},
        name="Pareto frontier",
        hoverinfo="skip",
    )
)
for mode, color in MODE_COLORS.items():
    part = view[view["mode_label"] == mode]
    if part.empty:
        continue
    on_frontier = part["config_id"].isin(frontier_ids)
    fig.add_trace(
        go.Scatter(
            x=part["p95_total_ms"],
            y=part[metric],
            mode="markers",
            name=mode,
            marker={
                "color": color,
                "opacity": 1.0,
                "size": [13 if f else 9 for f in on_frontier],
                "line": {"color": SURFACE, "width": 2},
            },
            customdata=part[
                ["config_id", "chunker_label", "embedder_short", "backend", "rerank_label"]
            ],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>%{customdata[1]} / %{customdata[2]} / "
                "%{customdata[3]}<br>rerank %{customdata[4]}<br>"
                f"{metric} %{{y:.3f}}<br>p95 %{{x:.1f}} ms<extra>{mode}</extra>"
            ),
        )
    )
fig.update_xaxes(title="p95 total latency (ms)", type="log" if log_x else "linear")
fig.update_yaxes(title=metric)
event = st.plotly_chart(
    style_figure(fig, 520), theme=None, on_select="rerun", selection_mode="points"
)

st.caption(
    "Larger points are on the Pareto frontier (dotted line). Colour is the retrieval mode; "
    "BM25-control runs count as sparse. Hover a point for its config; select points to list "
    "them below."
)
points = event.selection.points if event and event.selection else []
picked = [p["customdata"][0] for p in points if p.get("customdata")]
shown = view[view["config_id"].isin(picked)] if picked else frontier
st.subheader("Selected configs" if picked else "Frontier configs")
st.dataframe(
    shown[
        [
            "config_id",
            "chunker_label",
            "embedder_short",
            "backend",
            "mode_label",
            "rerank_label",
            metric,
            "p95_total_ms",
        ]
    ],
    hide_index=True,
    width="stretch",
)
