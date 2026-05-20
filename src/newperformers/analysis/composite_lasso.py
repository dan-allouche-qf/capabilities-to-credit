"""Data-driven sector weights via Lasso CV.

The headline composite is the equal-weighted mean of the six PCA sector
scores. The Bayesian credit posterior already shows the data prefer
loadings concentrated on energy (positive) and health (negative). This
module formalises that by letting Lasso pick the weights directly,
using the full country-year panel of (sector score → S&P rating) as
the training set.

Outputs
-------
- ``outputs/tables/composite_lasso_weights.csv`` — selected coefficients.
- ``outputs/tables/composite_lasso_stability.csv`` — selection frequency
  across 200 bootstrap replicates.
- ``outputs/tables/composite_weights_comparison.csv`` — Kendall τ between
  the equal-weighted, AHP and Lasso-weighted composite rankings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

from . import composite as comp_mod
from ..utils.logging import get_logger
from ..utils.paths import OUT_TABLES, ensure_dirs

log = get_logger(__name__)

SECTORS = ["education", "energy", "research_innovation",
           "health", "housing_living", "security_stability"]


def _feature_frame(composite: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    rating = (panel[panel["indicator"] == "SP_RATING"][["iso3", "year", "value"]]
              .rename(columns={"value": "rating"}))
    base = composite[["iso3", "year"] + SECTORS]
    return base.merge(rating, on=["iso3", "year"], how="inner").dropna()


def fit_lasso(composite: pd.DataFrame, panel: pd.DataFrame) -> dict[str, float]:
    ensure_dirs()
    df = _feature_frame(composite, panel)
    X = df[SECTORS].to_numpy(dtype=float)
    y = df["rating"].to_numpy(dtype=float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = LassoCV(cv=5, random_state=20260519, max_iter=5000).fit(Xs, y)
    coefs = dict(zip(SECTORS, model.coef_.tolist(), strict=True))
    log.info("Lasso CV alpha = %.4f, n = %d", model.alpha_, len(y))
    log.info("Selected coefficients:\n%s",
             pd.Series(coefs).round(3).to_string())
    pd.DataFrame([{"sector": s, "coef": c, "alpha": float(model.alpha_)}
                  for s, c in coefs.items()]).to_csv(
        OUT_TABLES / "composite_lasso_weights.csv", index=False)
    return coefs


def selection_stability(composite: pd.DataFrame, panel: pd.DataFrame,
                         n_boot: int = 200, seed: int = 20260519) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = _feature_frame(composite, panel)
    X = df[SECTORS].to_numpy(dtype=float)
    y = df["rating"].to_numpy(dtype=float)
    n = len(y)
    counts = {s: 0 for s in SECTORS}
    coef_sums = {s: 0.0 for s in SECTORS}
    successes = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Xb, yb = X[idx], y[idx]
        scaler = StandardScaler()
        Xs = scaler.fit_transform(Xb)
        try:
            model = LassoCV(cv=5, random_state=int(rng.integers(0, 2**31)),
                             max_iter=2000).fit(Xs, yb)
        except Exception:  # noqa: BLE001
            continue
        successes += 1
        for s, c in zip(SECTORS, model.coef_, strict=True):
            if abs(c) > 1e-6:
                counts[s] += 1
            coef_sums[s] += float(c)
    rows = []
    for s in SECTORS:
        rows.append({"sector": s,
                      "selection_freq": counts[s] / max(successes, 1),
                      "mean_coef": coef_sums[s] / max(successes, 1)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / "composite_lasso_stability.csv", index=False)
    log.info("Lasso stability (n_boot=%d):\n%s", successes,
             out.round(3).to_string(index=False))
    return out


def compare_weight_schemes(composite: pd.DataFrame,
                            lasso_coefs: dict[str, float]) -> pd.DataFrame:
    """Compare equal-weight / AHP / Lasso composites by Kendall τ on rank."""
    end_year = int(composite["year"].max())

    def _rank(weights: dict[str, float]) -> pd.Series:
        w = pd.Series(weights, dtype=float)
        w = w / w.abs().sum() if w.abs().sum() > 0 else w
        snap = composite[composite["year"] == end_year]
        score = (snap[SECTORS] * w).sum(axis=1)
        score.index = snap["iso3"].values
        return score.rank(ascending=False)

    equal_w = {s: 1.0 for s in SECTORS}
    ahp = comp_mod.AHP_WEIGHTS
    lasso_abs = {s: abs(v) for s, v in lasso_coefs.items()}

    eq_rank = _rank(equal_w)
    ahp_rank = _rank(ahp)
    lasso_rank = _rank(lasso_abs)

    pairs = [("equal", "ahp", eq_rank, ahp_rank),
             ("equal", "lasso", eq_rank, lasso_rank),
             ("ahp", "lasso", ahp_rank, lasso_rank)]
    rows = []
    for a, b, ra, rb in pairs:
        common = ra.index.intersection(rb.index)
        tau, p = kendalltau(ra.loc[common], rb.loc[common])
        rows.append({"weighting_a": a, "weighting_b": b,
                      "kendall_tau": float(tau), "p_value": float(p),
                      "n": int(len(common))})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / "composite_weights_comparison.csv", index=False)
    log.info("Weight-scheme rank comparison:\n%s",
             out.round(3).to_string(index=False))
    return out


def run_all(composite: pd.DataFrame, panel: pd.DataFrame) -> dict[str, object]:
    coefs = fit_lasso(composite, panel)
    stab = selection_stability(composite, panel)
    cmp_df = compare_weight_schemes(composite, coefs)
    return {"coefs": coefs, "stability": stab, "compare": cmp_df}
