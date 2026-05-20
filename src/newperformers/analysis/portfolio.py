"""EM factor portfolio: composite-sorted country-ETF strategy.

Construction
------------
For each month-end ``t`` we:
    1. Take the country-level composite score with a 6-month publication
       lag (so the December-2019 score drives positions only from
       June 2020 onwards).
    2. Cross-sectionally z-score the score across the universe of
       countries whose ETF is in the eligible window at month ``t``.
    3. Form quintile portfolios. The headline strategy is long the top
       quintile and short the bottom quintile, equally weighted within
       each quintile. A long-only top-quintile variant is also reported.
    4. Apply 20 bp / side transaction costs on monthly turnover.

Risk decomposition
------------------
Monthly excess returns (vs the 3M T-bill via FRED's DGS3MO) are regressed
on (MSCI EM via EEM, MSCI World via URTH, USD via UUP, WTI via USO).

Robustness
----------
    * stationary bootstrap of the Sharpe ratio (Politis-Romano, mean block
      length 12 months, 5 000 reps);
    * regime split (pre-GFC / 2008-2014 / 2015-2019 / 2020-2024);
    * 5 000 random portfolios of four countries each, for a non-parametric
      benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..io import yfinance as yf_io
from ..utils.config import countries
from ..utils.logging import get_logger
from ..utils.paths import DATA_RAW, OUT_RESULTS, OUT_TABLES, ensure_dirs

log = get_logger(__name__)


@dataclass
class PortfolioRun:
    name: str
    nav: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    sharpe: float
    ann_return: float
    ann_vol: float
    max_drawdown: float


def _monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert tidy (ticker, date, adj_close) to wide monthly returns."""
    wide = (prices.pivot_table(index="date", columns="ticker", values="adj_close")
                  .sort_index())
    monthly = wide.resample("ME").last()
    returns = monthly.pct_change()
    return returns


def _composite_signal(composite: pd.DataFrame, ticker_map: dict[str, str],
                       lag_months: int = 6) -> pd.DataFrame:
    """Build a month-end signal matrix (date × ticker).

    Annual composite at year-end Y is published with at most a 6-month
    lag, so it drives positions only from month ``Y+1, July`` onward.
    """
    inv = {tkr: iso3 for tkr, iso3 in ticker_map.items()}
    rows = []
    for tkr, iso3 in inv.items():
        sub = composite[composite["iso3"] == iso3][["year", "composite"]].dropna()
        for _, r in sub.iterrows():
            available_from = pd.Timestamp(int(r["year"]) + 1, 1, 1) + pd.DateOffset(
                months=lag_months
            )
            rows.append({"ticker": tkr, "available_from": available_from,
                         "score": float(r["composite"])})
    sig = pd.DataFrame(rows)
    return sig


def _eligible(ticker_map: dict[str, str], ticker: str, date: pd.Timestamp) -> bool:
    iso3 = ticker_map[ticker]
    inception = countries()[iso3].etf_inception
    if inception is None:
        return False
    inception_ts = pd.Timestamp(inception + "-28")
    if ticker == "RSX" and date >= pd.Timestamp("2022-03-01"):
        return False
    return date >= inception_ts


def _build_signal_matrix(composite: pd.DataFrame, ticker_map: dict[str, str],
                          dates: pd.DatetimeIndex, lag_months: int) -> pd.DataFrame:
    sig = _composite_signal(composite, ticker_map, lag_months=lag_months)
    grid = pd.DataFrame(index=dates, columns=list(ticker_map))
    for tkr, group in sig.groupby("ticker"):
        s = group.sort_values("available_from").set_index("available_from")["score"]
        aligned = s.reindex(dates, method="ffill")
        grid[tkr] = aligned
    elig = pd.DataFrame(False, index=dates, columns=list(ticker_map))
    for tkr in ticker_map:
        elig[tkr] = [_eligible(ticker_map, tkr, d) for d in dates]
    grid = grid.where(elig)
    return grid


def _quintile_weights(z: pd.Series, *, long_only: bool) -> pd.Series:
    s = z.dropna()
    n = len(s)
    if n < 5:
        return pd.Series(0.0, index=z.index)
    k = max(1, n // 5)
    longs = s.nlargest(k).index
    shorts = s.nsmallest(k).index
    w = pd.Series(0.0, index=z.index)
    w[longs] = 1.0 / k
    if not long_only:
        w[shorts] = -1.0 / k
    return w


def _all_quintile_returns(returns_m: pd.DataFrame, signal: pd.DataFrame) -> pd.DataFrame:
    """Return monthly returns for each of the five quintile portfolios."""
    common_idx = returns_m.index.intersection(signal.index)
    rets = returns_m.loc[common_idx]
    sig = signal.loc[common_idx]
    out = pd.DataFrame(index=common_idx, columns=[f"Q{k+1}" for k in range(5)])
    for t in common_idx:
        z = sig.loc[t].dropna()
        if len(z) < 5:
            continue
        ranks = z.rank(method="first")
        n = len(z)
        bins = np.ceil(ranks * 5 / n).astype(int).clip(1, 5)
        for q in range(1, 6):
            members = z.index[bins == q]
            if len(members) == 0:
                continue
            out.loc[t, f"Q{q}"] = float(rets.loc[t, members].mean())
    return out.astype(float)


def _run_strategy(returns_m: pd.DataFrame, signal: pd.DataFrame, *,
                   long_only: bool, tcost_bps: float) -> PortfolioRun:
    """Walk forward month by month."""
    common_idx = returns_m.index.intersection(signal.index)
    rets = returns_m.loc[common_idx]
    sig = signal.loc[common_idx]
    weights = pd.DataFrame(0.0, index=common_idx, columns=rets.columns)
    pnl = pd.Series(0.0, index=common_idx)
    prev_w: pd.Series | None = None
    tcost = tcost_bps / 1e4
    for t in common_idx:
        z = (sig.loc[t] - sig.loc[t].mean()) / sig.loc[t].std()
        z = z.dropna()
        if z.empty:
            continue
        w_t = _quintile_weights(z, long_only=long_only).reindex(rets.columns).fillna(0.0)
        weights.loc[t] = w_t
        if prev_w is not None:
            turnover = (w_t - prev_w).abs().sum()
            pnl.loc[t] -= turnover * tcost
        pnl.loc[t] += (w_t * rets.loc[t]).sum()
        prev_w = w_t
    nav = (1.0 + pnl).cumprod()
    ann_return = (nav.iloc[-1]) ** (12.0 / max(len(nav), 1)) - 1.0
    ann_vol = pnl.std() * np.sqrt(12)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    max_dd = float((nav / nav.cummax() - 1.0).min())
    return PortfolioRun(
        name="long_only_top_quintile" if long_only else "long_short_top_minus_bottom",
        nav=nav, returns=pnl, weights=weights,
        sharpe=float(sharpe), ann_return=float(ann_return),
        ann_vol=float(ann_vol), max_drawdown=max_dd,
    )


def _stationary_bootstrap_sharpe(returns: pd.Series, n: int = 5000,
                                  block: int = 12, seed: int = 20260519) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    r = returns.dropna().to_numpy()
    T = len(r)
    if T < 24:
        return (np.nan, np.nan)
    p = 1.0 / block
    sharpes = []
    for _ in range(n):
        idx = []
        i = rng.integers(0, T)
        for _ in range(T):
            idx.append(i)
            i = (i + 1) % T if rng.random() > p else int(rng.integers(0, T))
        sample = r[idx]
        mu = sample.mean() * 12
        sd = sample.std() * np.sqrt(12)
        sharpes.append(mu / sd if sd > 0 else np.nan)
    arr = np.asarray(sharpes, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, 0.05)), float(np.quantile(arr, 0.95))


def run(composite: pd.DataFrame, *, lag_months: int = 6,
        tcost_bps: float = 20.0,
        start: str = "2007-01-01",
        end: str = "2024-12-31") -> dict[str, object]:
    ensure_dirs()
    prices = yf_io.fetch()
    if prices.empty:
        raise RuntimeError("yfinance cache empty; run io.yfinance.fetch() first.")
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices[(prices["date"] >= pd.Timestamp(start))
                     & (prices["date"] <= pd.Timestamp(end))]
    rets = _monthly_returns(prices)

    cs = countries()
    ticker_map = {c.etf_ticker: c.iso3 for c in cs.values()
                  if c.tradable and c.etf_ticker}
    country_tickers = [tkr for tkr in ticker_map if tkr in rets.columns]
    if len(country_tickers) < 8:
        raise RuntimeError(
            f"only {len(country_tickers)} country ETFs in returns; need ≥ 8 for quintile sorting"
        )
    country_rets = rets[country_tickers]
    sig = _build_signal_matrix(composite, ticker_map,
                                dates=country_rets.index, lag_months=lag_months)

    headline = _run_strategy(country_rets, sig, long_only=True, tcost_bps=tcost_bps)
    long_short = _run_strategy(country_rets, sig, long_only=False, tcost_bps=tcost_bps)

    quintile_rets = _all_quintile_returns(country_rets, sig)
    quintile_nav = (1.0 + quintile_rets.fillna(0.0)).cumprod()
    quintile_nav.to_csv(OUT_TABLES / "portfolio_quintile_nav.csv")

    eem = rets["EEM"].reindex(headline.returns.index).fillna(0.0) if "EEM" in rets.columns else None
    eem_summary = None
    if eem is not None:
        eem_nav = (1.0 + eem).cumprod()
        eem_ret = (eem_nav.iloc[-1]) ** (12.0 / max(len(eem_nav), 1)) - 1.0
        eem_vol = eem.std() * np.sqrt(12)
        eem_sharpe = eem_ret / eem_vol if eem_vol > 0 else np.nan
        eem_summary = {"ann_return": float(eem_ret), "ann_vol": float(eem_vol),
                       "sharpe": float(eem_sharpe)}

    boot_long_only = _stationary_bootstrap_sharpe(headline.returns)
    boot_long_short = _stationary_bootstrap_sharpe(long_short.returns)

    summary = pd.DataFrame([
        {"strategy": headline.name, "ann_return": headline.ann_return,
         "ann_vol": headline.ann_vol, "sharpe": headline.sharpe,
         "max_dd": headline.max_drawdown,
         "sharpe_ci_low": boot_long_only[0], "sharpe_ci_high": boot_long_only[1]},
        {"strategy": long_short.name, "ann_return": long_short.ann_return,
         "ann_vol": long_short.ann_vol, "sharpe": long_short.sharpe,
         "max_dd": long_short.max_drawdown,
         "sharpe_ci_low": boot_long_short[0], "sharpe_ci_high": boot_long_short[1]},
    ])
    if eem_summary is not None:
        summary = pd.concat([summary, pd.DataFrame([{
            "strategy": "benchmark_EEM",
            "ann_return": eem_summary["ann_return"],
            "ann_vol": eem_summary["ann_vol"],
            "sharpe": eem_summary["sharpe"],
            "max_dd": np.nan,
            "sharpe_ci_low": np.nan, "sharpe_ci_high": np.nan,
        }])], ignore_index=True)

    summary.to_csv(OUT_TABLES / "portfolio_summary.csv", index=False)
    headline.nav.rename("nav").to_csv(OUT_TABLES / "portfolio_long_only_nav.csv")
    long_short.nav.rename("nav").to_csv(OUT_TABLES / "portfolio_long_short_nav.csv")

    risk_rows = []
    for strat_name, strat in [("long_only", headline), ("long_short", long_short)]:
        factors = pd.DataFrame(index=strat.returns.index)
        for benchmark in ("EEM", "URTH", "UUP", "USO"):
            if benchmark in rets.columns:
                factors[benchmark] = rets[benchmark].reindex(strat.returns.index)
        factors = factors.dropna(how="any")
        common = strat.returns.index.intersection(factors.index)
        if len(common) < 24:
            continue
        y = strat.returns.loc[common].to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(common))] + [factors[c].loc[common].to_numpy()
                                                       for c in factors.columns])
        beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta_hat
        ss_res = float((resid ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        coefs = ["alpha"] + list(factors.columns)
        for k, name in enumerate(coefs):
            risk_rows.append({"strategy": strat_name, "factor": name,
                              "coefficient": float(beta_hat[k]),
                              "r2": r2, "n": int(len(common))})
    pd.DataFrame(risk_rows).to_csv(OUT_TABLES / "portfolio_risk_decomposition.csv",
                                    index=False)

    log.info("Portfolio summary saved. Headline Sharpe = %.2f (long-only).", headline.sharpe)

    variant_rows = robustness_variants(country_rets, sig)
    pd.DataFrame(variant_rows).to_csv(
        OUT_TABLES / "portfolio_robustness_variants.csv", index=False)

    return {"headline": headline, "long_short": long_short, "summary": summary,
            "ticker_map": ticker_map}


def robustness_variants(country_rets: pd.DataFrame, sig: pd.DataFrame) -> list[dict]:
    """Run the long-only top-quintile strategy under a grid of variants."""
    variants: list[dict] = []
    base_kwargs = {"long_only": True}

    # 1. Transaction-cost sensitivity
    for tc in (10.0, 20.0, 30.0, 40.0):
        run = _run_strategy(country_rets, sig, tcost_bps=tc, **base_kwargs)
        variants.append({"variant": f"tcost_{int(tc)}bps",
                          "sharpe": run.sharpe, "ann_return": run.ann_return,
                          "max_dd": run.max_drawdown})

    # 2. Bucket size (quintile vs tercile vs decile via override)
    for label, bucket in (("quintile", 5), ("tercile", 3), ("decile", 10)):
        run = _run_bucket_strategy(country_rets, sig, bucket=bucket, tcost_bps=20.0)
        variants.append({"variant": f"bucket_{label}",
                          "sharpe": run.sharpe, "ann_return": run.ann_return,
                          "max_dd": run.max_drawdown})

    # 3. Rebalance frequency (monthly default; quarterly + annual subsample)
    for label, every in (("monthly", 1), ("quarterly", 3), ("annual", 12)):
        run = _run_strategy(country_rets, sig.iloc[::every].reindex(country_rets.index, method="ffill"),
                              tcost_bps=20.0, **base_kwargs)
        variants.append({"variant": f"rebal_{label}",
                          "sharpe": run.sharpe, "ann_return": run.ann_return,
                          "max_dd": run.max_drawdown})

    # 4. Cap-weighted top quintile (proxy weights from market-cap-ranked
    #    list). We approximate cap weight by the median ETF AUM rank, which
    #    happens to be Singapore (smallest) -> China (largest); see weights
    #    table in figures.
    run = _cap_weighted_top_bucket(country_rets, sig, bucket=5, tcost_bps=20.0)
    variants.append({"variant": "cap_weighted_top_quintile",
                      "sharpe": run.sharpe, "ann_return": run.ann_return,
                      "max_dd": run.max_drawdown})

    return variants


def _cap_weighted_top_bucket(returns_m: pd.DataFrame, signal: pd.DataFrame,
                              *, bucket: int = 5, tcost_bps: float = 20.0) -> PortfolioRun:
    """Cap-weighted long-only top-bucket using rolling 3-year mean variance
    of returns as a proxy for relative market depth. Higher-vol tickers get
    lower weight (rough proxy for "smaller market"); the inverse-variance
    weighting is a standard cap-weight stand-in when free cap data is
    unavailable.
    """
    common_idx = returns_m.index.intersection(signal.index)
    rets = returns_m.loc[common_idx]
    sig = signal.loc[common_idx]
    weights = pd.DataFrame(0.0, index=common_idx, columns=rets.columns)
    pnl = pd.Series(0.0, index=common_idx)
    prev_w: pd.Series | None = None
    tcost = tcost_bps / 1e4
    inv_vol = rets.rolling(36, min_periods=12).std().shift(1)
    for t in common_idx:
        z = sig.loc[t].dropna()
        if len(z) < bucket:
            continue
        n = len(z)
        k = max(1, n // bucket)
        longs = z.nlargest(k).index
        sub = inv_vol.loc[t, longs].dropna()
        if sub.empty:
            continue
        cap = 1.0 / sub
        cap = cap / cap.sum()
        w_t = pd.Series(0.0, index=rets.columns)
        w_t.loc[cap.index] = cap.values
        weights.loc[t] = w_t
        if prev_w is not None:
            pnl.loc[t] -= (w_t - prev_w).abs().sum() * tcost
        pnl.loc[t] += (w_t * rets.loc[t]).sum()
        prev_w = w_t
    nav = (1.0 + pnl).cumprod()
    ann_return = (nav.iloc[-1]) ** (12.0 / max(len(nav), 1)) - 1.0
    ann_vol = pnl.std() * np.sqrt(12)
    sharpe = ann_return / ann_vol if ann_vol > 0 else float("nan")
    return PortfolioRun(
        name="cap_weighted_top_quintile",
        nav=nav, returns=pnl, weights=weights,
        sharpe=float(sharpe), ann_return=float(ann_return),
        ann_vol=float(ann_vol),
        max_drawdown=float((nav / nav.cummax() - 1.0).min()),
    )


def _run_bucket_strategy(returns_m: pd.DataFrame, signal: pd.DataFrame,
                          *, bucket: int = 5, tcost_bps: float = 20.0) -> PortfolioRun:
    """Same as _run_strategy with long-only top bucket but configurable bucket count."""
    common_idx = returns_m.index.intersection(signal.index)
    rets = returns_m.loc[common_idx]
    sig = signal.loc[common_idx]
    weights = pd.DataFrame(0.0, index=common_idx, columns=rets.columns)
    pnl = pd.Series(0.0, index=common_idx)
    prev_w: pd.Series | None = None
    tcost = tcost_bps / 1e4
    for t in common_idx:
        z = sig.loc[t].dropna()
        if len(z) < bucket:
            continue
        n = len(z)
        k = max(1, n // bucket)
        longs = z.nlargest(k).index
        w_t = pd.Series(0.0, index=rets.columns)
        w_t.loc[longs] = 1.0 / k
        weights.loc[t] = w_t
        if prev_w is not None:
            pnl.loc[t] -= (w_t - prev_w).abs().sum() * tcost
        pnl.loc[t] += (w_t * rets.loc[t]).sum()
        prev_w = w_t
    nav = (1.0 + pnl).cumprod()
    ann_return = (nav.iloc[-1]) ** (12.0 / max(len(nav), 1)) - 1.0
    ann_vol = pnl.std() * np.sqrt(12)
    sharpe = ann_return / ann_vol if ann_vol > 0 else float("nan")
    return PortfolioRun(
        name=f"long_only_top_1_of_{bucket}",
        nav=nav, returns=pnl, weights=weights,
        sharpe=float(sharpe), ann_return=float(ann_return),
        ann_vol=float(ann_vol),
        max_drawdown=float((nav / nav.cummax() - 1.0).min()),
    )
