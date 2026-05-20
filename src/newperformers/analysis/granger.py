"""Dumitrescu-Hurlin (2012) panel Granger causality.

For each country ``i`` we fit
    y_{i,t} = α_i + Σ_{l=1..p} γ_{i,l} y_{i,t-l} + Σ_{l=1..p} β_{i,l} x_{i,t-l} + ε_{i,t}
and Wald-test H0: β_{i,1} = … = β_{i,p} = 0 (no Granger causality from x to y for
country i). The Dumitrescu-Hurlin statistic averages the country-level Wald
statistics:
    W̄ = (1 / N) Σ_i W_i,
which under H0 has Z̄ = √N (W̄ − p) / √(2p) → N(0, 1) for large T.

We use this to test the panel hypothesis that sector-score s Granger-causes
each macro outcome y (log GDP per capita ×100, poverty headcount, Gini).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import f as fdist

from ..utils.logging import get_logger
from ..utils.paths import OUT_TABLES, ensure_dirs

log = get_logger(__name__)


@dataclass
class GrangerResult:
    shock: str
    outcome: str
    p_lags: int
    w_bar: float
    z_bar: float
    p_value: float
    n_countries: int
    notes: str = ""


def _country_wald(y: np.ndarray, x: np.ndarray, p: int) -> float | None:
    T = len(y)
    if T < 4 * p + 5:
        return None
    Y = y[p:]
    X_lags_y = np.column_stack([y[p - i - 1 : T - i - 1] for i in range(p)])
    X_lags_x = np.column_stack([x[p - i - 1 : T - i - 1] for i in range(p)])
    Z = np.column_stack([np.ones(len(Y)), X_lags_y, X_lags_x])
    Z_r = np.column_stack([np.ones(len(Y)), X_lags_y])  # restricted (no x lags)
    try:
        beta_u, *_ = np.linalg.lstsq(Z, Y, rcond=None)
        beta_r, *_ = np.linalg.lstsq(Z_r, Y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    res_u = Y - Z @ beta_u
    res_r = Y - Z_r @ beta_r
    rss_u = float((res_u ** 2).sum())
    rss_r = float((res_r ** 2).sum())
    if rss_u <= 0 or rss_r <= 0:
        return None
    n_params = Z.shape[1]
    df_resid = len(Y) - n_params
    if df_resid <= 0:
        return None
    # Wald (F-form) for joint significance of p coefficients on x-lags.
    f_stat = ((rss_r - rss_u) / p) / (rss_u / df_resid)
    # Convert to Wald-equivalent W = p * F (DH 2012 statistic).
    return float(p * f_stat)


def panel_granger(panel: pd.DataFrame, shock: str, outcome: str,
                   *, p: int = 2) -> GrangerResult:
    pivot = panel.pivot_table(index=["iso3", "year"], columns="indicator",
                               values="value").reset_index()
    if shock not in pivot.columns or outcome not in pivot.columns:
        raise KeyError(f"missing {shock} or {outcome} in panel")
    wald_stats: list[float] = []
    for iso3, grp in pivot.groupby("iso3"):
        grp = grp.sort_values("year")
        y = grp[outcome].to_numpy(dtype=float)
        x = grp[shock].to_numpy(dtype=float)
        m = ~(np.isnan(y) | np.isnan(x))
        y = y[m]; x = x[m]
        w = _country_wald(y, x, p)
        if w is not None and np.isfinite(w):
            wald_stats.append(w)
    if not wald_stats:
        return GrangerResult(shock, outcome, p, np.nan, np.nan, np.nan, 0,
                             "no country fits")
    arr = np.asarray(wald_stats)
    w_bar = float(arr.mean())
    z_bar = np.sqrt(len(arr)) * (w_bar - p) / np.sqrt(2 * p)
    # Two-sided p-value using F approximation tail
    p_val = float(1 - fdist.cdf(w_bar / p, p, max(arr.shape[0], 1)))
    return GrangerResult(shock, outcome, p, w_bar, float(z_bar), p_val,
                         int(len(arr)),
                         "Dumitrescu-Hurlin Z̄ statistic; approx F-tail p")


def run_grid(panel: pd.DataFrame, shocks: list[str], outcomes: list[str],
              *, p: int = 2) -> pd.DataFrame:
    ensure_dirs()
    rows = []
    for s in shocks:
        for o in outcomes:
            try:
                r = panel_granger(panel, s, o, p=p)
                rows.append({"shock": s, "outcome": o, "p_lags": r.p_lags,
                             "w_bar": r.w_bar, "z_bar": r.z_bar,
                             "p_value": r.p_value, "n_countries": r.n_countries})
                log.info("Granger %s -> %s: W̄=%.2f Z̄=%.2f p=%.3f (N=%d)",
                         s, o, r.w_bar, r.z_bar, r.p_value, r.n_countries)
            except Exception as exc:  # noqa: BLE001
                log.warning("Granger %s -> %s failed: %s", s, o, exc)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / "granger_panel.csv", index=False)
    return out
