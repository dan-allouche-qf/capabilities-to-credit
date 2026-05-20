from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]

st.set_page_config(page_title="Portfolio", layout="wide")
st.title("KPI-sorted EM factor portfolio")


@st.cache_data
def load_summary() -> pd.DataFrame:
    return pd.read_csv(ROOT / "outputs/tables/portfolio_summary.csv")


@st.cache_data
def load_nav(name: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / f"outputs/tables/portfolio_{name}_nav.csv")
    df.columns = ["date", "nav"]
    df["date"] = pd.to_datetime(df["date"])
    return df


s = load_summary()
st.subheader("Performance summary")
st.dataframe(s.round(3), use_container_width=True, hide_index=True)

st.subheader("Cumulative NAV (start = 1.0, after 20 bps tcosts)")
lo = load_nav("long_only")
ls = load_nav("long_short")
lo["strategy"] = "Long-only top quintile"
ls["strategy"] = "Long-short top − bottom"
df = pd.concat([lo, ls], ignore_index=True)
fig = px.line(df, x="date", y="nav", color="strategy",
               color_discrete_map={"Long-only top quintile": "#0F4C5C",
                                    "Long-short top − bottom": "#E36414"})
fig.update_layout(template="plotly_white", height=440,
                   yaxis_title="Cumulative NAV (log)",
                   yaxis_type="log",
                   margin=dict(l=40, r=20, t=40, b=40))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Drawdown — long-only")
dd = lo.copy()
dd["dd"] = dd["nav"] / dd["nav"].cummax() - 1.0
fig = px.area(dd, x="date", y="dd", color_discrete_sequence=["#E36414"])
fig.update_layout(template="plotly_white", height=320,
                   yaxis_title="Drawdown", margin=dict(l=40, r=20, t=40, b=40))
st.plotly_chart(fig, use_container_width=True)
