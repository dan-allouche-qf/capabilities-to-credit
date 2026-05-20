from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]

st.set_page_config(page_title="Causal evidence", layout="wide")
st.title("Causal evidence")


@st.cache_data
def load_lp() -> pd.DataFrame:
    return pd.read_csv(ROOT / "outputs/tables/lp_irf.csv")


@st.cache_data
def load_granger() -> pd.DataFrame:
    return pd.read_csv(ROOT / "outputs/tables/granger_panel.csv")


@st.cache_data
def load_synth(tag: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / f"outputs/tables/synthcontrol_{tag}.csv")


tab1, tab2, tab3 = st.tabs(["Local projections", "Panel Granger", "Synthetic control"])

with tab1:
    df = load_lp()
    shocks = sorted(df["shock"].unique())
    outcomes = sorted(df["outcome"].unique())
    cols = st.columns(2)
    with cols[0]:
        shock = st.selectbox("Sector shock", shocks, index=shocks.index("energy") if "energy" in shocks else 0)
    with cols[1]:
        outcome = st.selectbox("Outcome", outcomes, index=outcomes.index("log_gdppc_x100") if "log_gdppc_x100" in outcomes else 0)
    sub = df[(df["shock"] == shock) & (df["outcome"] == outcome)].sort_values("horizon")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["horizon"], y=sub["beta"], mode="lines+markers",
                              line=dict(color="#0F4C5C", width=2.4),
                              name="IRF"))
    fig.add_trace(go.Scatter(x=sub["horizon"], y=sub["beta"] + 1.645 * sub["se"],
                              line=dict(color="#0F4C5C", width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=sub["horizon"], y=sub["beta"] - 1.645 * sub["se"],
                              line=dict(color="#0F4C5C", width=0),
                              fill="tonexty", fillcolor="rgba(15,76,92,0.20)",
                              showlegend=True, name="90% DK CI"))
    fig.add_hline(y=0, line_dash="dash", line_color="#8D99AE", line_width=1)
    fig.update_layout(template="plotly_white", height=420,
                       title=f"{shock} → {outcome}",
                       xaxis_title="Horizon (years)",
                       yaxis_title="IRF coefficient",
                       margin=dict(l=40, r=20, t=60, b=40))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(sub.round(3), use_container_width=True, hide_index=True)

with tab2:
    g = load_granger()
    pivot = g.pivot(index="shock", columns="outcome", values="p_value").round(3)
    st.subheader("Dumitrescu-Hurlin panel Granger — p-values")
    st.dataframe(pivot.style.background_gradient(cmap="Blues_r", vmin=0, vmax=0.5),
                  use_container_width=True)
    fig = px.imshow(pivot, color_continuous_scale="Blues_r",
                     range_color=(0, 0.5), text_auto=True)
    fig.update_layout(template="plotly_white", height=420, margin=dict(l=40, r=20, t=20, b=40))
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    tag = st.selectbox("Episode", ["rwanda_2000", "singapore_1990"])
    sc = load_synth(tag)
    treatment_year = 2000 if tag.endswith("2000") else 1990
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sc["year"], y=sc["actual"], name="Actual",
                              line=dict(color="#E36414", width=2.4)))
    fig.add_trace(go.Scatter(x=sc["year"], y=sc["synthetic"], name="Synthetic",
                              line=dict(color="#0F4C5C", width=2.0, dash="dash")))
    fig.add_vline(x=treatment_year, line_color="#8D99AE", line_dash="dash")
    fig.update_layout(template="plotly_white", height=440,
                       yaxis_title="GDPpc (PPP, 2017 USD)",
                       margin=dict(l=40, r=20, t=20, b=40))
    st.plotly_chart(fig, use_container_width=True)
    weights = pd.read_csv(ROOT / f"outputs/tables/synthcontrol_{tag}_weights.csv",
                           index_col=0)
    weights.columns = ["weight"]
    st.subheader("Optimal donor weights")
    st.dataframe(weights.round(3))
