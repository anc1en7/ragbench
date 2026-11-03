"""Quality vs latency: every dev run, with the Pareto frontier labelled."""

from __future__ import annotations

import math

import plotly.graph_objects as go
import streamlit as st
import ui
from data import METRICS, table

ui.setup(
    "Quality vs latency",
    "Each point is one pipeline on the dev split. Points on the frontier are the ones no "
    "other pipeline beats on both quality and speed. Everything below the line is dominated.",
)

c1, c2, _ = st.columns([1, 1, 2], gap="small")
metric = c1.selectbox("Quality metric", METRICS, index=0)
log_x = c2.toggle("Log latency axis", value=True)
df = table("dev").sort_values(["p95_ms", metric], ascending=[True, False])

frontier_ids: list[str] = []
best = -1.0
for _, row in df.iterrows():
    if row[metric] > best:
        frontier_ids.append(row["config_id"])
        best = row[metric]
df["frontier"] = df["config_id"].isin(frontier_ids)
front = df[df["frontier"]].sort_values("p95_ms")

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=front["p95_ms"],
        y=front[metric],
        mode="lines",
        line={"color": ui.INK_3, "width": 1.5, "shape": "hv"},
        name="Pareto frontier",
        hoverinfo="skip",
    )
)
for mode_name, color in ui.MODE_COLORS.items():
    part = df[df["mode"] == mode_name]
    if part.empty:
        continue
    fig.add_trace(
        go.Scatter(
            x=part["p95_ms"],
            y=part[metric],
            mode="markers",
            name=mode_name,
            marker={
                "color": color,
                "size": [13 if f else 9 for f in part["frontier"]],
                "line": {"color": ui.SURFACE, "width": 2},
            },
            customdata=part[["pipeline", "chunker", "embedder", "config_id"]],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>%{customdata[1]} · %{customdata[2]}<br>"
                f"{metric} %{{y:.3f}} · p95 %{{x:.0f}} ms<extra></extra>"
            ),
        )
    )
# Selective direct labels: frontier points get a number that keys into the table below.
front = front.assign(n=range(1, len(front) + 1))
for _, row in front.iterrows():
    fig.add_annotation(
        x=math.log10(row["p95_ms"]) if log_x else row["p95_ms"],
        y=row[metric],
        text=f"<b>{row['n']}</b>",
        showarrow=False,
        yshift=15,
        font={"size": 11, "color": ui.INK},
    )
fig.update_xaxes(title="p95 total latency (ms)", type="log" if log_x else "linear")
fig.update_yaxes(title=metric)
event = st.plotly_chart(
    ui.style(fig, 540),
    theme=None,
    on_select="rerun",
    selection_mode="points",
    config={"displayModeBar": False},
)
st.caption(
    "Colour is the retrieval mode (the BM25 control counts as sparse). Larger points sit on "
    "the frontier and are numbered to match the table below. The three sparse points near "
    "zero are pgvector full-text search, which requires every query term to match (see "
    "Methodology). Select points to list them below."
)

points = event.selection.points if event and event.selection else []
picked = [p["customdata"][3] for p in points if p.get("customdata")]
shown = df[df["config_id"].isin(picked)].assign(n="") if picked else front
st.subheader("Selected pipelines" if picked else "Frontier pipelines")
st.dataframe(
    shown[["n", "pipeline", "chunker", "embedder", metric, "p95_ms", "config_id"]],
    hide_index=True,
    width="stretch",
    column_config={
        "n": st.column_config.Column("#", width="small"),
        "pipeline": "Pipeline",
        "chunker": "Chunks",
        "embedder": "Embedder",
        metric: st.column_config.NumberColumn(metric, format="%.3f"),
        "p95_ms": st.column_config.NumberColumn("p95 ms", format="%.0f"),
        "config_id": "Config id",
    },
)
