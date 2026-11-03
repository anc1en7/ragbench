"""Methodology and limitations, rendered from docs/ (Task 5.4)."""

from __future__ import annotations

import streamlit as st
from data import ROOT

st.set_page_config(page_title="Methodology · RAGBench", layout="wide")
methodology, limitations = st.tabs(["Methodology", "Limitations"])
with methodology:
    st.markdown((ROOT / "docs" / "METHODOLOGY.md").read_text(encoding="utf-8"))
with limitations:
    st.markdown((ROOT / "docs" / "LIMITATIONS.md").read_text(encoding="utf-8"))
