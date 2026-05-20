"""Panel unit-root and cointegration tests.

Two procedures are implemented:

1. **CIPS** (\textcite{pesaran2007}) — Cross-sectionally Augmented IPS panel
   unit-root test. For each country ``i`` we estimate the ADF regression
   augmented with the cross-sectional average of ``y`` (and its lagged
   differences) as additional regressors. The country-level t-statistic
   on the lagged level is denoted CADF_i; the CIPS statistic is the
   average across countries. Critical values are from Pesaran (2007)
   Table II for the case with intercept only.

2. **Westerlund-style panel cointegration** (\textcite{westerlund2007})
   — for each ``i`` we estimate the cointegrating regression
   ``y_{i,t} = α_i + β_i x_{i,t} + u_{i,t}`` and apply ADF on the residuals.
   The group-mean test statistic (average t-stat across countries) is
   compared against the standard-normal asymptote, which over-rejects
   in small panels; this is documented in the methodology section.

Output: ``outputs/tables/cointegration_results.csv`` with one row per
(test, series_or_pair).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..utils.logging import get_logger
from ..utils.paths import OUT_TABLES, ensure_dirs

log = get_logger(__name__)

# Pesaran (2007) Table II — CIPS critical values for "intercept only".
# Linearly interpolated between (N=10, T=20) and (N=20, T=30); for the
# (N≈22, T≈35) panel of this paper the published values are appropriate.
CIPS_CRITICAL = {
    "intercept": {0.10: -2.07, 0.05: -2.15, 0.01: -2.30},
}


def _adf_tstat(y: np.ndarray, lag: int = 1, exog: np.ndarray | None = None) -> float:
    """Augmented Dickey–Fuller t-statistic on the lagged level coefficient.

    Δy_t = α + ρ y_{t-1} + Σ_{k=1..lag} φ_k Δy_{t-k} + Z γ + ε_t
    """
    y = np.asarray(y, dtype=float)
    if len(y) < lag + 4:
        return float("nan")
    dy = np.diff(y)
    n = len(dy) - lag
    if n < 4:
        return float("nan")
    y_lag = y[lag:-1]
    intercept = np.ones(n)
    rows = [intercept, y_lag]
    for k in range(1, lag + 1):
        rows.append(dy[lag - k: -k] if k > 0 else dy[lag:])
    if exog is not None:
        Z = np.asarray(exog, dtype=float)
        # Align to last n observations.
        Z = Z[-n:]
        if Z.ndim == 1:
            rows.append(Z)
        else:
            for col in Z.T:
                rows.append(col)
    X = np.column_stack(rows)
    Y = dy[lag:]
    try:
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        resid = Y - X @ beta
        sigma2 = (resid ** 2).sum() / max(len(Y) - X.shape[1], 1)
        XtX_inv = np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.diag(sigma2 * XtX_inv))
        rho_index = 1  # second column is y_lag
        if se[rho_index] <= 0 or not np.isfinite(se[rho_index]):
            return float("nan")
        return float(beta[rho_index] / se[rho_index])
    except np.linalg.LinAlgError:
        return float("nan")


def cips(panel: pd.DataFrame, indicator: str, *, lag: int = 1) -> dict[str, float]:
    """Pesaran (2007) CIPS panel unit-root test for one indicator."""
    pivot = (panel[panel["indicator"] == indicator]
             .pivot_table(index="year", columns="iso3", values="value")
             .sort_index())
    pivot = pivot.dropna(how="any", axis=1)  # keep balanced countries only
    if pivot.shape[1] < 5:
        return {"indicator": indicator, "n_countries": int(pivot.shape[1]),
                "cips_stat": float("nan"), "p_005_critical": np.nan,
                "reject_at_5pct": False}
    y_bar = pivot.mean(axis=1).to_numpy()
    dy_bar = np.diff(y_bar)
    cadfs = []
    for iso3 in pivot.columns:
        y = pivot[iso3].to_numpy()
        # Cross-sectional augmenters: y_bar lagged level + lag dy_bar
        z_cols = [y_bar[lag:-1]] + [dy_bar[lag - k: -k] if k > 0 else dy_bar[lag:]
                                      for k in range(1, lag + 1)]
        z = np.column_stack(z_cols)
        t = _adf_tstat(y, lag=lag, exog=z)
        if np.isfinite(t):
            cadfs.append(t)
    cips_stat = float(np.mean(cadfs)) if cadfs else float("nan")
    crit_5 = CIPS_CRITICAL["intercept"][0.05]
    return {
        "test": "CIPS",
        "indicator": indicator,
        "n_countries": len(cadfs),
        "cips_stat": cips_stat,
        "p_005_critical": crit_5,
        "p_010_critical": CIPS_CRITICAL["intercept"][0.10],
        "p_001_critical": CIPS_CRITICAL["intercept"][0.01],
        "reject_unit_root_at_5pct": cips_stat < crit_5,
    }


def westerlund_group_t(panel: pd.DataFrame, y_indicator: str,
                        x_indicator: str, *, lag: int = 1) -> dict[str, float]:
    """Westerlund-style group-mean panel cointegration t-statistic.

    Per-country: regress y on x with an intercept, apply ADF on residuals.
    The group t-statistic averages the country-level ADF t-stats. Compared
    against the standard normal as an asymptotic approximation (Westerlund's
    proper distribution depends on serial-correlation correction and is
    bootstrapped in the original paper).
    """
    pivot_y = (panel[panel["indicator"] == y_indicator]
                .pivot_table(index="year", columns="iso3", values="value")
                .sort_index())
    pivot_x = (panel[panel["indicator"] == x_indicator]
                .pivot_table(index="year", columns="iso3", values="value")
                .sort_index())
    common = pivot_y.index.intersection(pivot_x.index)
    pivot_y = pivot_y.loc[common]
    pivot_x = pivot_x.loc[common]
    common_cols = pivot_y.columns.intersection(pivot_x.columns)
    pivot_y = pivot_y[common_cols].dropna(how="any", axis=1)
    pivot_x = pivot_x[pivot_y.columns]

    t_stats = []
    for iso3 in pivot_y.columns:
        y = pivot_y[iso3].dropna()
        x = pivot_x[iso3].reindex(y.index).dropna()
        n = min(len(y), len(x))
        if n < 12:
            continue
        y_arr = y.iloc[-n:].to_numpy()
        x_arr = x.iloc[-n:].to_numpy()
        X = np.column_stack([np.ones(n), x_arr])
        try:
            beta, *_ = np.linalg.lstsq(X, y_arr, rcond=None)
        except np.linalg.LinAlgError:
            continue
        resid = y_arr - X @ beta
        t = _adf_tstat(resid, lag=lag)
        if np.isfinite(t):
            t_stats.append(t)
    if not t_stats:
        return {"test": "Westerlund_Gt", "y": y_indicator, "x": x_indicator,
                "n_countries": 0, "group_t": float("nan"),
                "pvalue_normal": float("nan"),
                "reject_no_cointegration_at_5pct": False}
    G_t = float(np.mean(t_stats))
    # One-sided normal p-value: cointegration ⇒ residuals stationary ⇒ G_t very negative.
    p_norm = float(norm.cdf(G_t))
    return {
        "test": "Westerlund_Gt",
        "y": y_indicator,
        "x": x_indicator,
        "n_countries": len(t_stats),
        "group_t": G_t,
        "pvalue_normal": p_norm,
        "reject_no_cointegration_at_5pct": p_norm < 0.05,
    }


def run_panel_tests(panel: pd.DataFrame, composite: pd.DataFrame | None = None
                     ) -> pd.DataFrame:
    """Run CIPS on log GDP per capita, the composite, each sector score,
    and Westerlund-style cointegration tests between the composite and
    log GDP per capita.
    """
    ensure_dirs()
    rows: list[dict] = []

    # CIPS on level indicators
    cips_targets = [
        ("NY.GDP.PCAP.CD", "log"),     # log GDP per capita
        ("FP.CPI.TOTL.ZG", "level"),   # CPI inflation, % (already a rate)
        ("GC.DOD.TOTL.GD.ZS", "level"),
    ]
    for code, mode in cips_targets:
        df = panel[panel["indicator"] == code].copy()
        if mode == "log":
            df["value"] = np.log(df["value"].clip(lower=1.0))
        res = cips(df, code, lag=1)
        res["transform"] = mode
        rows.append(res)
        log.info("CIPS [%s (%s)]: stat=%.3f vs 5%% crit %.2f (reject=%s)",
                 code, mode, res["cips_stat"], res["p_005_critical"],
                 res["reject_unit_root_at_5pct"])

    # CIPS on composite + each sector score
    if composite is not None:
        cols = [c for c in composite.columns if c not in {"iso3", "year"}]
        for col in cols:
            df = composite[["iso3", "year", col]].rename(columns={col: "value"}).copy()
            df["indicator"] = f"composite::{col}"
            res = cips(df, f"composite::{col}", lag=1)
            res["transform"] = "score"
            rows.append(res)
            log.info("CIPS [%s]: stat=%.3f", col, res["cips_stat"])

    # Westerlund-style cointegration: composite vs log GDP/capita
    if composite is not None and "composite" in composite.columns:
        gdp = panel[panel["indicator"] == "NY.GDP.PCAP.CD"].copy()
        gdp["value"] = np.log(gdp["value"].clip(lower=1.0))
        gdp["indicator"] = "log_gdppc"
        comp_long = composite[["iso3", "year", "composite"]].rename(
            columns={"composite": "value"}).copy()
        comp_long["indicator"] = "composite_score"
        merged = pd.concat([gdp[["iso3", "year", "indicator", "value"]],
                              comp_long[["iso3", "year", "indicator", "value"]]],
                             ignore_index=True)
        res = westerlund_group_t(merged, "log_gdppc", "composite_score", lag=1)
        rows.append(res)
        log.info("Westerlund Gt: y=log_gdppc x=composite_score, Gt=%.3f, p=%.3f",
                 res["group_t"], res["pvalue_normal"])

    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / "cointegration_results.csv", index=False)
    return out
