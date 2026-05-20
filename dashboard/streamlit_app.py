"""From Capabilities to Credit — Streamlit dashboard entry point.

Run with::

    streamlit run dashboard/streamlit_app.py

The five pages live in ``dashboard/pages/`` and each one reads the
pre-computed parquet/CSV artifacts in ``outputs/`` and ``data/processed/``.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title="From Capabilities to Credit",
    page_icon=":bar_chart:",
    layout="wide",
)

st.title("From Capabilities to Credit")
st.caption("A causal-and-cross-sectional study of EM development, 1990–2024 — Dan Allouche")

st.markdown(
    """
This dashboard sits on top of the analysis pipeline.

* **Country profile** — composite, sector scores, macro snapshot.
* **Cross-country** — ranking, heatmap, scatter explorer.
* **Causal evidence** — local-projection IRFs by sector shock.
* **Credit scorecard** — out-of-sample ordered-probit predictions.
* **Portfolio** — long-only top-quintile NAV vs MSCI EM.

All numbers shown come from the parquet/CSV files in `data/processed/`
and `outputs/`. The dashboard never recomputes; it just reads.
"""
)

st.info(
    "Use the sidebar (top left) to navigate. If a page is empty, run "
    "`python -m newperformers.pipeline all` first to populate outputs/."
)
