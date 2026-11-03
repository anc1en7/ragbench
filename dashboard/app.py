"""RAGBench replay dashboard. Run: ``make dashboard``."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

VIEWS = Path(__file__).parent / "views"

st.set_page_config(page_title="RAGBench", page_icon="📊", layout="wide")
page = st.navigation(
    [
        st.Page(VIEWS / "overview.py", title="Overview", default=True),
        st.Page(VIEWS / "leaderboard.py", title="Leaderboard"),
        st.Page(VIEWS / "tradeoffs.py", title="Quality vs latency"),
        st.Page(VIEWS / "ablations.py", title="Ablations"),
        st.Page(VIEWS / "explorer.py", title="Query explorer"),
        st.Page(VIEWS / "methodology.py", title="Methodology"),
    ],
    position="top",
)
page.run()
