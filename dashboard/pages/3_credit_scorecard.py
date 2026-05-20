from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]

st.set_page_config(page_title="Credit scorecard", layout="wide")
st.title("Sovereign credit scorecard")


@st.cache_data
def load_scorecard() -> pd.DataFrame:
    return pd.read_csv(ROOT / "outputs/tables/credit_oos_scorecard.csv")


@st.cache_data
def load_predictions() -> pd.DataFrame:
    return pd.read_csv(ROOT / "outputs/tables/credit_oos_predictions.csv")


sc = load_scorecard()
pred = load_predictions()

st.subheader("Out-of-sample scorecard")
st.dataframe(sc.round(3), use_container_width=True, hide_index=True)

st.subheader("Calibration — predicted vs actual rating")
fig = px.scatter(pred, x="actual", y="predicted", color="iso3", hover_data=["year"])
fig.add_shape(type="line", x0=1, y0=1, x1=22, y1=22, line=dict(color="#8D99AE", dash="dash"))
fig.update_layout(template="plotly_white", height=560,
                   xaxis_title="Actual S&P rating (numeric 1–22)",
                   yaxis_title="OOS predicted rating",
                   margin=dict(l=40, r=20, t=40, b=40))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Country trajectory")
country = st.sidebar.selectbox("Country", sorted(pred["iso3"].unique()),
                                index=0)
sub = pred[pred["iso3"] == country].sort_values("year")
fig = px.line(sub, x="year", y=["actual", "predicted"], markers=True,
               color_discrete_sequence=["#0F4C5C", "#E36414"])
fig.update_layout(template="plotly_white", height=380,
                   yaxis_title="Numeric rating", margin=dict(l=40, r=20, t=40, b=40))
st.plotly_chart(fig, use_container_width=True)
