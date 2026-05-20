from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]

st.set_page_config(page_title="Country profile", layout="wide")
st.title("Country profile")


@st.cache_data
def load_composite() -> pd.DataFrame:
    return pd.read_csv(ROOT / "outputs/tables/composite_score.csv")


@st.cache_data
def load_panel() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data/processed/panel_imputed.parquet")


composite = load_composite()
panel = load_panel()

iso3s = sorted(composite["iso3"].unique())
country = st.sidebar.selectbox("Country", iso3s, index=iso3s.index("SGP"))

sub = composite[composite["iso3"] == country].sort_values("year")
sectors = ["education", "energy", "research_innovation",
           "health", "housing_living", "security_stability"]

cols = st.columns(3)
with cols[0]:
    latest = sub["composite"].iloc[-1]
    earliest = sub["composite"].iloc[0]
    st.metric(label="Composite (latest)", value=f"{latest:.2f}",
              delta=f"{latest - earliest:+.2f} vs 1990")
with cols[1]:
    gdppc = panel[(panel["iso3"] == country)
                  & (panel["indicator"] == "NY.GDP.PCAP.CD")].sort_values("year")
    if not gdppc.empty:
        st.metric("GDP per capita (USD)", f"${gdppc['value'].iloc[-1]:,.0f}",
                  delta=f"×{gdppc['value'].iloc[-1] / gdppc['value'].iloc[0]:.1f} since 1990")
with cols[2]:
    rating = panel[(panel["iso3"] == country)
                   & (panel["indicator"] == "SP_RATING")].sort_values("year")
    if not rating.empty:
        latest_r = int(rating["value"].iloc[-1])
        st.metric("S&P rating (numeric 1–22)", latest_r,
                  delta=f"IG = {latest_r >= 13}")
    else:
        st.metric("S&P rating", "NR")

st.subheader("Composite score")
fig = go.Figure()
fig.add_trace(go.Scatter(x=sub["year"], y=sub["composite"], mode="lines",
                          line=dict(color="#0F4C5C", width=2.4),
                          name="Composite"))
fig.add_hline(y=0, line_color="#8D99AE", line_dash="dash", line_width=0.8)
fig.update_layout(template="plotly_white", height=300, margin=dict(l=40, r=20, t=40, b=40),
                  yaxis_title="Composite (standardised)")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Sector scores")
fig = go.Figure()
palette_cycle = ["#0F4C5C", "#E36414", "#FB8B24", "#5F0F40", "#2A9D8F", "#8D99AE"]
for i, s in enumerate(sectors):
    fig.add_trace(go.Scatter(x=sub["year"], y=sub[s], mode="lines", name=s,
                              line=dict(color=palette_cycle[i], width=1.6)))
fig.update_layout(template="plotly_white", height=380, margin=dict(l=40, r=20, t=40, b=40),
                  yaxis_title="Sector score (standardised)")
st.plotly_chart(fig, use_container_width=True)
