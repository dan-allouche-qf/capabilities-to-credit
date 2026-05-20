from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]

st.set_page_config(page_title="Cross-country", layout="wide")
st.title("Cross-country comparison")


@st.cache_data
def load() -> pd.DataFrame:
    return pd.read_csv(ROOT / "outputs/tables/composite_score.csv")


df = load()
years = sorted(df["year"].unique())
year = st.sidebar.select_slider("Year", years, value=int(df["year"].max()))

snap = df[df["year"] == year].copy()
snap["rank"] = snap["composite"].rank(ascending=False, method="min").astype(int)
snap = snap.sort_values("rank")

st.subheader(f"Ranking — {year}")
cols = st.columns([2, 3])
with cols[0]:
    st.dataframe(snap[["rank", "iso3", "composite"]].set_index("rank"),
                 height=620, use_container_width=True)
with cols[1]:
    fig = px.bar(snap, x="composite", y="iso3", orientation="h",
                  color="composite", color_continuous_scale=["#5F0F40", "#FFFFFF", "#0F4C5C"],
                  color_continuous_midpoint=0)
    fig.update_layout(template="plotly_white", height=620,
                       margin=dict(l=20, r=20, t=20, b=40),
                       yaxis=dict(autorange="reversed"),
                       xaxis_title="Composite (z-score)", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Sector heatmap")
sectors = ["education", "energy", "research_innovation", "health",
           "housing_living", "security_stability"]
heat = snap.set_index("iso3")[sectors]
fig = px.imshow(heat, color_continuous_scale=["#5F0F40", "#FFFFFF", "#0F4C5C"],
                color_continuous_midpoint=0, aspect="auto")
fig.update_layout(template="plotly_white", height=620,
                   margin=dict(l=20, r=20, t=20, b=40))
st.plotly_chart(fig, use_container_width=True)
