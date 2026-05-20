from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]

st.set_page_config(page_title="Interactive rating scorecard", layout="wide")
st.title("Interactive rating scorecard")
st.markdown(
    "Move the six sector sliders. The predicted rating updates live, using "
    "the Bayesian posterior-mean loadings from "
    "`outputs/tables/credit_bayesian_loadings.csv` as marginal sensitivities "
    "around the BBB- (numeric 13) reference point. This is a **linear-approximation "
    "scoring tool**, not the full Bayesian predictor: the production model maps "
    "the latent index $\\eta = \\alpha + \\beta \\cdot \\text{sectors}$ through the "
    "posterior cutpoints, which this UI does not load. Treat the predicted notch "
    "as a directional sensitivity readout, not as a forecast."
)


@st.cache_data
def load_loadings() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "outputs/tables/credit_bayesian_loadings.csv")
    df = df[df["sector"] != "sector"].copy()
    for c in ("posterior_mean", "hdi_5%", "hdi_95%"):
        df[c] = df[c].astype(float)
    return df


@st.cache_data
def load_composite() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "outputs/tables/composite_score.csv")
    return df[df["year"] == df["year"].max()].copy()


loadings = load_loadings()
panel_latest = load_composite()

sectors = list(loadings["sector"])
betas = dict(zip(loadings["sector"], loadings["posterior_mean"]))

st.sidebar.header("Sector scores (sigma units)")
slider_values: dict[str, float] = {}
for s in sectors:
    pretty = s.replace("_", " ").title()
    slider_values[s] = st.sidebar.slider(
        pretty, min_value=-3.0, max_value=4.0, value=0.0, step=0.1
    )

NUMERIC_TO_RATING = {
    22: "AAA", 21: "AA+", 20: "AA", 19: "AA-",
    18: "A+", 17: "A", 16: "A-",
    15: "BBB+", 14: "BBB", 13: "BBB-",
    12: "BB+", 11: "BB", 10: "BB-",
    9: "B+", 8: "B", 7: "B-",
    6: "CCC+", 5: "CCC", 4: "CCC-",
    3: "CC", 2: "C", 1: "SD/D",
}

INTERCEPT_BBB = 13.0


def predict_rating(values: dict[str, float]) -> tuple[float, str, str]:
    z = INTERCEPT_BBB + sum(betas[s] * values[s] for s in sectors)
    z = float(np.clip(z, 1, 22))
    notch = int(round(z))
    ig = "Investment grade" if notch >= 13 else "Speculative grade"
    return z, NUMERIC_TO_RATING.get(notch, "—"), ig


z, label, ig_class = predict_rating(slider_values)

c1, c2, c3 = st.columns(3)
c1.metric("Predicted numeric rating", f"{z:.1f} / 22")
c2.metric("Rating notch (closest)", label)
c3.metric("Class", ig_class)

st.subheader("Driver breakdown")
contributions = []
for s in sectors:
    contributions.append({
        "sector": s.replace("_", " ").title(),
        "score": slider_values[s],
        "beta": betas[s],
        "contribution": slider_values[s] * betas[s],
    })
contrib_df = pd.DataFrame(contributions)
contrib_df = contrib_df.sort_values("contribution", ascending=False)

fig = go.Figure(
    go.Bar(
        x=contrib_df["contribution"],
        y=contrib_df["sector"],
        orientation="h",
        marker=dict(
            color=["#0F4C5C" if v >= 0 else "#E36414" for v in contrib_df["contribution"]],
        ),
        text=[f"{v:+.2f}" for v in contrib_df["contribution"]],
        textposition="outside",
    )
)
fig.update_layout(
    template="plotly_white",
    height=400,
    xaxis_title="Contribution to predicted notch (= sector score x posterior beta)",
    margin=dict(l=60, r=20, t=20, b=40),
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Posterior loadings (read-only)")
st.dataframe(
    loadings.assign(
        contribution=lambda d: d["sector"].map(slider_values) * d["posterior_mean"],
    ).round(3),
    use_container_width=True,
    hide_index=True,
)

st.subheader("Where your country profile sits in the latest cross-section")
st.caption(
    "Compares your slider profile to the 2024 panel of 23 countries on each sector dimension."
)
panel_cols = [c for c in sectors if c in panel_latest.columns]
panel_means = panel_latest[panel_cols].mean()
panel_stds = panel_latest[panel_cols].std()

quant_fig = go.Figure()
quant_fig.add_trace(go.Bar(
    name="Panel mean",
    x=panel_cols, y=panel_means.values,
    marker_color="#8D99AE",
))
quant_fig.add_trace(go.Bar(
    name="Your profile",
    x=panel_cols, y=[slider_values[s] for s in panel_cols],
    marker_color="#E36414",
))
quant_fig.update_layout(
    template="plotly_white",
    height=380,
    barmode="group",
    margin=dict(l=40, r=20, t=20, b=40),
)
st.plotly_chart(quant_fig, use_container_width=True)
