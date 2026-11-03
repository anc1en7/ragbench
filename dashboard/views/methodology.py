"""Methodology and limitations, rendered from docs/."""

from __future__ import annotations

import streamlit as st
import ui
from data import ROOT

ui.setup(
    "Methodology",
    "How the numbers were produced, and what they cannot tell you.",
)
methodology, limitations = st.tabs(["Methodology", "Limitations"])
with methodology:
    st.markdown((ROOT / "docs" / "METHODOLOGY.md").read_text(encoding="utf-8"))
with limitations:
    st.markdown((ROOT / "docs" / "LIMITATIONS.md").read_text(encoding="utf-8"))
