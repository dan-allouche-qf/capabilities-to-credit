"""EM sovereign-credit backtest using the KPI composite as the sort signal.

The Part-III equity backtest fails: the KPI signal sits at the 0th
percentile of random EM equity portfolios. The conclusion of the paper
claims the same signal *should* work in credit because ratings re-price
slowly. This module tests that claim using freely available EM bond
instruments:

    1. **Aggregate test** — time-series regression of monthly EMB returns
       on the *cross-sectional dispersion* of the lagged composite score.
       If KPI improvement at the panel level predicts EMB returns, that's
       the headline evidence.
    2. **Credit-sorted equity overlay** — country ETF returns sorted by
       the country's S&P numeric rating predicted by the Bayesian model.
       Goes long predicted-upgrade tickers, short predicted-downgrades.
       Uses the existing equity universe because country-level sovereign
       bond ETFs are not free.
    3. **Factor decomposition** — adds EMB to the risk-factor regression
       of the headline KPI equity portfolio, to surface whether the KPI
       signal loads more on credit beta than on equity beta.

Outputs land in ``outputs/tables/portfolio_credit_*.csv``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..io import yfinance as yf_io
from ..utils.config import countries
from ..utils.logging import get_logger
from ..utils.paths import OUT_TABLES, ensure_dirs

log = get_logger(__name__)


def _monthly_returns(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    df = prices[prices["ticker"].isin(tickers)].copy()
    df["date"] = pd.to_datetime(df["date"])
    wide = (df.pivot_table(index="date", columns="ticker", values="adj_close")
              .sort_index())
    monthly = wide.resample("ME").last()
    return monthly.pct_change()


def aggregate_signal_vs_emb(composite: pd.DataFrame, *,
                              start: str = "2008-01-01",
                              end: str = "2024-12-31") -> dict[str, float]:
    """Time-series test: does panel-mean composite predict next-month EMB?

    For each calendar month t, the signal is the year-over-year change in
    the cross-sectional mean composite at year(t)-1 (because composite
    data is annual). The dependent variable is the monthly EMB return at
    month t. Reports OLS coefficient, t-stat, and R^2.
    """
    ensure_dirs()
    prices = yf_io.fetch()
    if prices.empty:
        return {}
    monthly = _monthly_returns(prices, ["EMB"])
    if "EMB" not in monthly.columns:
        log.warning("EMB not in yfinance cache — credit aggregate test skipped")
        return {}
    emb = monthly["EMB"].dropna()
    emb = emb.loc[(emb.index >= pd.Timestamp(start))
                   & (emb.index <= pd.Timestamp(end))]
    if emb.empty:
        return {}

    annual_mean = (composite.groupby("year")["composite"].mean()
                   .sort_index())
    annual_dmean = annual_mean.diff()  # YoY change in cross-sectional mean
    annual_dmean = annual_dmean.shift(1)  # lag one year — publication lag

    # Map each month to the prior calendar year's signal.
    sig = pd.Series(index=emb.index, dtype=float)
    for ts in emb.index:
        year = int(ts.year)
        if year in annual_dmean.index and not pd.isna(annual_dmean.loc[year]):
            sig.loc[ts] = float(annual_dmean.loc[year])
    df = pd.DataFrame({"emb_ret": emb, "signal": sig}).dropna()
    if len(df) < 24:
        return {}
    X = np.column_stack([np.ones(len(df)), df["signal"].to_numpy()])
    y = df["emb_ret"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n = len(y)
    sigma2 = float((resid ** 2).sum() / max(n - 2, 1))
    XtX_inv = np.linalg.pinv(X.T @ X)
    se = float(np.sqrt(sigma2 * XtX_inv[1, 1]))
    tstat = float(beta[1] / se) if se > 0 else float("nan")
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    out = {"n_months": int(n), "beta": float(beta[1]), "se": se,
           "tstat": tstat, "r2": r2, "intercept": float(beta[0])}
    pd.DataFrame([out]).to_csv(OUT_TABLES / "portfolio_credit_aggregate.csv",
                                 index=False)
    log.info("Aggregate KPI vs EMB: β=%.4f, t=%.2f, R²=%.3f, n=%d",
             out["beta"], out["tstat"], out["r2"], out["n_months"])
    return out


def credit_factor_decomposition() -> pd.DataFrame:
    """Add EMB to the existing portfolio's factor regression.

    Re-runs the OLS regression of the long-only KPI strategy returns on
    (EEM, URTH, UUP, USO, EMB). Compares loadings to the original 4-factor
    decomposition.
    """
    ensure_dirs()
    nav = pd.read_csv(OUT_TABLES / "portfolio_long_only_nav.csv",
                       parse_dates=[0])
    nav.columns = ["date", "nav"]
    nav["date"] = pd.to_datetime(nav["date"])
    nav = nav.sort_values("date").set_index("date")
    strat_ret = nav["nav"].pct_change().dropna()

    prices = yf_io.fetch()
    if prices.empty:
        return pd.DataFrame()
    factors = ["EEM", "URTH", "UUP", "USO", "EMB"]
    monthly = _monthly_returns(prices, factors).dropna(how="all")
    monthly = monthly.reindex(strat_ret.index)
    factor_df = monthly[factors].dropna()
    common = strat_ret.index.intersection(factor_df.index)
    if len(common) < 36:
        return pd.DataFrame()

    y = strat_ret.loc[common].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(common))]
                          + [factor_df[c].loc[common].to_numpy() for c in factors])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rows = [{"factor": "alpha", "coefficient": float(beta[0]),
              "r2": r2, "n": int(len(common))}]
    for k, f in enumerate(factors, start=1):
        rows.append({"factor": f, "coefficient": float(beta[k]),
                      "r2": r2, "n": int(len(common))})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / "portfolio_credit_factor.csv", index=False)
    log.info("Credit-augmented factor decomposition: β_EMB=%.3f, R²=%.3f",
             out[out["factor"] == "EMB"]["coefficient"].iloc[0], r2)
    return out


def credit_signal_overlay(composite: pd.DataFrame,
                           predicted_rating_change: pd.DataFrame | None = None,
                           *, start: str = "2008-01-01",
                           end: str = "2024-12-31",
                           tcost_bps: float = 20.0) -> dict[str, float]:
    """Long predicted-upgrades, short predicted-downgrades equity overlay.

    Sorting variable: the country's *cross-sectional* composite z-score
    rank (used as a proxy for credit-quality momentum, which the AUC-0.81
    OOS scorecard says predicts rating). Long the top-3, short the
    bottom-3, equal-weighted within each leg, monthly rebal. The point of
    the test is to check whether a credit-flavoured tilt avoids the
    failure mode of the headline quintile sort.
    """
    ensure_dirs()
    cs = countries()
    ticker_map = {c.etf_ticker: c.iso3 for c in cs.values()
                  if c.tradable and c.etf_ticker}

    prices = yf_io.fetch()
    if prices.empty:
        return {}
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices[(prices["date"] >= pd.Timestamp(start))
                     & (prices["date"] <= pd.Timestamp(end))]
    tickers = [t for t in ticker_map if t in prices["ticker"].unique()]
    rets = _monthly_returns(prices, tickers)
    common_dates = rets.index

    # Convert annual composite to month-end signal with 6-month publication lag.
    sig_grid = pd.DataFrame(index=common_dates, columns=tickers, dtype=float)
    for tkr, iso3 in ticker_map.items():
        if tkr not in tickers:
            continue
        sub = composite[composite["iso3"] == iso3][
            ["year", "composite"]].dropna().sort_values("year")
        for _, row in sub.iterrows():
            available_from = pd.Timestamp(int(row["year"]) + 1, 1, 1) + pd.DateOffset(months=6)
            sig_grid.loc[sig_grid.index >= available_from, tkr] = float(row["composite"])

    weights = pd.DataFrame(0.0, index=common_dates, columns=tickers)
    pnl = pd.Series(0.0, index=common_dates)
    prev_w = pd.Series(0.0, index=tickers)
    tc = tcost_bps / 1e4
    k = 3
    for t in common_dates:
        z = sig_grid.loc[t].dropna()
        if len(z) < 2 * k:
            continue
        longs = z.nlargest(k).index
        shorts = z.nsmallest(k).index
        w_t = pd.Series(0.0, index=tickers)
        w_t.loc[longs] = 1.0 / k
        w_t.loc[shorts] = -1.0 / k
        weights.loc[t] = w_t
        turnover = (w_t - prev_w).abs().sum()
        pnl.loc[t] = (w_t * rets.loc[t]).sum() - turnover * tc
        prev_w = w_t
    nav = (1 + pnl.fillna(0)).cumprod()
    ann_ret = nav.iloc[-1] ** (12 / max(len(nav), 1)) - 1
    ann_vol = pnl.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    max_dd = float((nav / nav.cummax() - 1).min())

    out = {"strategy": "credit_overlay_top3_minus_bottom3",
           "ann_return": float(ann_ret), "ann_vol": float(ann_vol),
           "sharpe": float(sharpe), "max_dd": max_dd,
           "n_months": int(len(nav))}
    pd.DataFrame([out]).to_csv(OUT_TABLES / "portfolio_credit_overlay.csv",
                                 index=False)
    nav.rename("nav").to_csv(OUT_TABLES / "portfolio_credit_overlay_nav.csv")
    log.info("Credit overlay: Sharpe=%.2f, ann.ret=%.2f%%, max DD=%.0f%%",
             out["sharpe"], 100 * out["ann_return"], 100 * out["max_dd"])
    return out


def run_all(composite: pd.DataFrame) -> dict[str, object]:
    log.info("=== Credit portfolio backtest ===")
    agg = aggregate_signal_vs_emb(composite)
    factor = credit_factor_decomposition()
    overlay = credit_signal_overlay(composite)
    return {"aggregate": agg, "factor": factor, "overlay": overlay}
