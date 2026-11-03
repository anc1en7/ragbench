"""Visual system for the dashboard: tokens, CSS, stat tiles and chart styling.

Colours follow the dataviz reference palette: categorical slots 1-3 (validated all-pairs
for scatter) for retrieval modes, ink tokens for text, a recessive hairline grid, and a
neutral gray for de-emphasised marks. Text never wears a series colour.
"""

from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st

SURFACE = "#fcfcfb"
CARD = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#6f6e69"
GRID = "#e9e8e4"
LINE = "#dcdbd6"
BLUE = "#2a78d6"
MUTED_MARK = "#b9b8b2"
MODE_COLORS = {"dense": "#2a78d6", "hybrid": "#eb6834", "sparse": "#1baf7a"}

CSS = f"""
<style>
.block-container {{ padding-top: 4.2rem; padding-bottom: 3rem; max-width: 1180px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
h1 {{ font-size: 2.0rem !important; font-weight: 650 !important; }}
h3 {{ font-size: 1.15rem !important; font-weight: 620 !important; margin-top: 1.4rem; }}
.rb-sub {{ color: {INK_2}; font-size: 1.0rem; margin: -0.6rem 0 1.4rem 0; max-width: 760px; }}
.rb-card {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 12px;
            padding: 16px 18px; height: 100%; min-height: 118px; }}
.rb-label {{ color: {INK_2}; font-size: 0.82rem; margin-bottom: 4px; }}
.rb-value {{ color: {INK}; font-size: 1.9rem; font-weight: 640; line-height: 1.15; }}
.rb-hero {{ color: {INK}; font-size: 3.4rem; font-weight: 660; line-height: 1.0;
            margin: 6px 0 8px 0; }}
.rb-note {{ color: {INK_3}; font-size: 0.8rem; margin-top: 6px; line-height: 1.35; }}
.rb-text {{ color: {INK_2}; font-size: 0.9rem; margin-top: 6px; line-height: 1.45; }}
.rb-chip {{ display: inline-block; border: 1px solid {LINE}; border-radius: 999px;
            padding: 1px 9px; margin: 0 6px 6px 0; font-size: 0.78rem; color: {INK_2};
            background: {CARD}; }}
.rb-hit {{ border-bottom: 1px solid {GRID}; padding: 7px 2px; font-size: 0.86rem; }}
.rb-hit:last-child {{ border-bottom: none; }}
.rb-rank {{ display: inline-block; width: 26px; color: {INK_3};
            font-variant-numeric: tabular-nums; }}
.rb-gold {{ color: #006300; font-weight: 600; }}
.rb-miss {{ color: {INK_3}; }}
.rb-doc {{ color: {INK}; }}
.rb-score {{ float: right; color: {INK_3}; font-variant-numeric: tabular-nums; }}
mark {{ background: #fff1c2; padding: 0 1px; }}
div[data-testid="stCaptionContainer"] p {{ color: {INK_3}; }}
</style>
"""


def setup(title: str, subtitle: str | None = None) -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.title(title)
    if subtitle:
        st.markdown(f'<div class="rb-sub">{subtitle}</div>', unsafe_allow_html=True)


def tile(label: str, value: str, note: str = "", hero: bool = False) -> None:
    cls = "rb-hero" if hero else "rb-value"
    note_html = f'<div class="rb-note">{note}</div>' if note else ""
    st.markdown(
        f'<div class="rb-card"><div class="rb-label">{escape(label)}</div>'
        f'<div class="{cls}">{escape(value)}</div>{note_html}</div>',
        unsafe_allow_html=True,
    )


def finding(value: str, label: str, text: str) -> None:
    st.markdown(
        f'<div class="rb-card"><div class="rb-label">{escape(label)}</div>'
        f'<div class="rb-value">{escape(value)}</div>'
        f'<div class="rb-text">{text}</div></div>',
        unsafe_allow_html=True,
    )


def style(fig: Any, height: int = 420, legend: bool = True) -> Any:
    """Recessive axes and hairline grid, ink text, consistent surface."""
    fig.update_layout(
        height=height,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"color": INK_2, "size": 12, "family": "Source Sans Pro, system-ui, sans-serif"},
        margin={"l": 10, "r": 16, "t": 36 if legend else 12, "b": 10},
        hoverlabel={"bgcolor": CARD, "bordercolor": LINE, "font": {"color": INK, "size": 12}},
        showlegend=legend,
        legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0, "title": None},
    )
    axis = {
        "gridcolor": GRID,
        "zeroline": False,
        "linecolor": LINE,
        "ticks": "",
        "title_font": {"color": INK_2, "size": 12},
        "automargin": True,
    }
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig
