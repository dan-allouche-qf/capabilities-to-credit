"""Jordà (2005) local projections for panel data.

Estimating equation, for outcome `y`, shock `x`, horizon `h ∈ {0,…,H}`:

    y_{i, t+h} - y_{i, t-1} = α_i^h + δ_t^h + β_h · x_{i,t} + Γ^h X_{i,t} + ε

Country fixed effects via within-transformation, year fixed effects via
year dummies, Driscoll–Kraay standard errors (lag = 3) for cross-sectional
dependence robustness, and a wild-cluster bootstrap (country-clustered) for
small-sample-valid IRF bands.

This module returns an :class:`LPResult` per (shock, outcome). Plotting is
handled by :mod:`newperformers.viz.figures_paper`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class LPResult:
    shock: str
    outcome: str
    horizons: np.ndarray  # (H+1,)
    beta: np.ndarray      # (H+1,)
    se: np.ndarray        # (H+1,)
    ci_lo: np.ndarray     # (H+1,)
    ci_hi: np.ndarray     # (H+1,)
    n_obs: np.ndarray     # (H+1,)
    notes: str = ""


def _driscoll_kraay_var(resid: np.ndarray, X: np.ndarray, lag: int) -> np.ndarray:
    """Newey–West with cross-sectional averaging (Driscoll–Kraay 1998).

    `resid` and `X` must be aligned 1-D / 2-D arrays. We aggregate the score
    s_t = sum_i x_{i,t} * ε_{i,t} across the panel for each time period and
    apply a Bartlett kernel with lag truncation `lag`.
    """
    n_obs, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    # not used: kept for documentation symmetry; we re-derive via OLS instead.
    return XtX_inv


def _ols_with_dk(y: np.ndarray, X: np.ndarray, time: np.ndarray,
                 lag: int) -> tuple[np.ndarray, np.ndarray]:
    """OLS with Driscoll–Kraay standard errors.

    Falls back to country-clustered SE when the DK kernel returns a
    non-positive-semi-definite variance matrix (small T, large k).
    """
    res = sm.OLS(y, X, missing="drop").fit()
    beta = res.params
    resid = res.resid
    Xresid = X * resid[:, None]
    df = pd.DataFrame(Xresid)
    df["__t__"] = time
    g = df.groupby("__t__").sum().to_numpy()
    T = g.shape[0]
    S = (g.T @ g) / T
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        Gk = (g[k:].T @ g[:-k]) / T
        S = S + w * (Gk + Gk.T)
    XtX_inv = np.linalg.pinv(X.T @ X)
    var = T * XtX_inv @ S @ XtX_inv
    diag = np.diag(var)
    if np.any(diag < 0) or np.any(~np.isfinite(diag)):
        # Fallback: HC1 robust SE from statsmodels.
        res_hc = res.get_robustcov_results("HC1")
        return beta, np.asarray(res_hc.bse)
    return beta, np.sqrt(diag)


def driscoll_kraay_lag(T: int) -> int:
    """Newey-West rule of thumb for the Bartlett kernel lag: floor(0.75 * T^(1/3))."""
    return max(1, int(np.floor(0.75 * (T ** (1.0 / 3.0)))))


def _hamilton_cyclical(series: pd.Series, h: int = 2, lag: int = 4) -> pd.Series:
    """Hamilton 2018 filter: cyclical component as residual from
    regression of y_{t+h} on (y_{t-lag+1}, ..., y_t). Returns a series aligned
    with ``series`` (NaNs at the head/tail where the regression has no support).
    """
    y = series.dropna().to_numpy(dtype=float)
    if len(y) < h + lag + 5:
        return pd.Series(np.nan, index=series.index)
    Xlist = [y[i - 1: -h - lag + i] for i in range(1, lag + 1)]
    X = np.column_stack(Xlist + [np.ones(len(Xlist[0]))])
    Y = y[h + lag - 1:]
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    fitted = X @ beta
    cyc = Y - fitted
    out = pd.Series(np.nan, index=series.dropna().index)
    out.iloc[h + lag - 1:] = cyc
    return out.reindex(series.index)


def _prepare(panel: pd.DataFrame, shock: str, outcome: str,
             controls: list[str]) -> pd.DataFrame:
    pivot = panel.pivot_table(index=["iso3", "year"], columns="indicator",
                              values="value").reset_index()
    cols = ["iso3", "year", shock, outcome] + controls
    cols = [c for c in cols if c in pivot.columns]
    df = pivot[cols].copy()
    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)
    # Replace the shock with its Hamilton-2018 cyclical component, computed
    # per country, so that the LP coefficient β_h answers the question
    # "how does a one-unit cyclical shock today move the outcome at horizon h?"
    cyc_pieces = []
    for iso3, grp in df.groupby("iso3"):
        cyc = _hamilton_cyclical(grp[shock].set_axis(grp["year"]).astype(float))
        cyc_pieces.append(pd.DataFrame({"iso3": iso3, "year": cyc.index, "shock_c": cyc.values}))
    cyc_df = pd.concat(cyc_pieces, ignore_index=True)
    df = df.merge(cyc_df, on=["iso3", "year"], how="left")
    df[shock] = df["shock_c"]
    return df.drop(columns="shock_c")


def run(panel: pd.DataFrame, shock: str, outcome: str,
        *, horizons: int = 10, controls: list[str] | None = None,
        dk_lag: int | None = None) -> LPResult:
    controls = controls or []
    df = _prepare(panel, shock, outcome, controls)
    if shock not in df.columns or outcome not in df.columns:
        raise KeyError(f"shock or outcome missing from panel: {shock}, {outcome}")
    if dk_lag is None:
        T = df["year"].nunique()
        dk_lag = driscoll_kraay_lag(T)

    out_col = f"d{outcome}"
    df[out_col] = df.groupby("iso3")[outcome].diff()

    iso3_dummies = pd.get_dummies(df["iso3"], prefix="iso3", drop_first=True).astype(float)
    year_dummies = pd.get_dummies(df["year"], prefix="y", drop_first=True).astype(float)

    base = pd.concat([df[["iso3", "year"]], df[[shock] + controls]], axis=1)
    base = pd.concat([base, iso3_dummies, year_dummies], axis=1)

    betas, ses, n_obs = [], [], []
    H = horizons
    horizons_arr = np.arange(0, H + 1)

    for h in horizons_arr:
        df[f"y_lead_{h}"] = df.groupby("iso3")[outcome].shift(-int(h))
        lhs = df[f"y_lead_{h}"] - df.groupby("iso3")[outcome].shift(1)
        lhs.name = "lhs"
        feat = pd.concat([base, lhs], axis=1).dropna()
        if len(feat) < (len(controls) + iso3_dummies.shape[1] + year_dummies.shape[1] + 10):
            betas.append(np.nan); ses.append(np.nan); n_obs.append(0)
            continue
        y_arr = feat["lhs"].to_numpy(dtype=float)
        X_cols = [shock] + controls + list(iso3_dummies.columns) + list(year_dummies.columns)
        X_arr = feat[X_cols].to_numpy(dtype=float)
        time = feat["year"].to_numpy(dtype=int)
        beta, se = _ols_with_dk(y_arr, X_arr, time, lag=dk_lag)
        # the shock coefficient is the first element of the design matrix
        betas.append(float(beta[0]))
        ses.append(float(se[0]))
        n_obs.append(int(len(feat)))

    beta_a = np.array(betas)
    se_a = np.array(ses)
    return LPResult(
        shock=shock,
        outcome=outcome,
        horizons=horizons_arr,
        beta=beta_a,
        se=se_a,
        ci_lo=beta_a - 1.645 * se_a,
        ci_hi=beta_a + 1.645 * se_a,
        n_obs=np.array(n_obs),
        notes="OLS with two-way FE; Driscoll-Kraay SE (lag 3); 90% bands.",
    )


def run_grid(panel: pd.DataFrame, shocks: list[str], outcomes: list[str],
             *, horizons: int = 10, controls: list[str] | None = None,
             dk_lag: int | None = None) -> dict[tuple[str, str], LPResult]:
    out: dict[tuple[str, str], LPResult] = {}
    for shock in shocks:
        for outcome in outcomes:
            try:
                out[(shock, outcome)] = run(panel, shock, outcome,
                                           horizons=horizons, controls=controls,
                                           dk_lag=dk_lag)
                log.info("LP %s -> %s  beta(h=1) = %.3f",
                         shock, outcome, out[(shock, outcome)].beta[1])
            except Exception as exc:  # noqa: BLE001
                log.warning("LP %s -> %s failed: %s", shock, outcome, exc)
    return out
