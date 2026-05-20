"""Robustness diagnostics referenced in Section 8 of the paper.

Produces three tables consumed by the LaTeX prose:

    1. ``robustness_imputed_vs_raw.csv`` — composite rank correlation and
       OOS credit AUC under both panels.
    2. ``robustness_subperiod_sharpe.csv`` — long-only portfolio Sharpe
       computed over four pre-specified sub-periods.
    3. ``robustness_random_portfolios.csv`` — distribution of Sharpe
       ratios across 5 000 random four-country portfolios drawn from
       the tradable universe, with the KPI strategy's percentile rank.

Also records the imputation-tier breakdown used in Appendix B.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import composite as comp_mod
from . import credit as credit_mod
from ..io import yfinance as yf_io
from ..utils.config import countries
from ..utils.logging import get_logger
from ..utils.paths import DATA_PROCESSED, OUT_TABLES, ensure_dirs

log = get_logger(__name__)


def imputed_vs_raw(imputed_panel: pd.DataFrame) -> dict[str, float]:
    """Re-run composite scoring + credit OOS on the raw panel; compare."""
    raw_panel = pd.read_parquet(DATA_PROCESSED / "panel_raw.parquet")

    imp_fits = comp_mod.sector_scores(imputed_panel)
    imp_comp = comp_mod.composite_score(imp_fits)

    raw_fits = comp_mod.sector_scores(raw_panel)
    raw_comp = comp_mod.composite_score(raw_fits)

    year = int(min(imp_comp["year"].max(), raw_comp["year"].max()))
    imp_snap = imp_comp[imp_comp["year"] == year].set_index("iso3")["composite"]
    raw_snap = raw_comp[raw_comp["year"] == year].set_index("iso3")["composite"]
    common = imp_snap.index.intersection(raw_snap.index)
    rho, _ = spearmanr(imp_snap.loc[common], raw_snap.loc[common])

    imp_credit = credit_mod.fit_models(imp_comp, imputed_panel)
    raw_credit = credit_mod.fit_models(raw_comp, raw_panel)
    imp_auc = float(imp_credit["scorecards"]
                    .query("model == 'sectors_only'")["auc_ig"].iloc[0])
    raw_auc = float(raw_credit["scorecards"]
                    .query("model == 'sectors_only'")["auc_ig"].iloc[0])

    out = {"rank_spearman": float(rho), "auc_imputed": imp_auc, "auc_raw": raw_auc,
           "n_countries": int(len(common))}
    log.info("Robustness imputed-vs-raw: Spearman=%.3f, AUC %.3f vs %.3f",
             out["rank_spearman"], out["auc_imputed"], out["auc_raw"])
    return out


def subperiod_sharpe(rf_annual: float = 0.025) -> pd.DataFrame:
    """Sub-period Sharpe of the long-only top-quintile portfolio."""
    df = pd.read_csv(OUT_TABLES / "portfolio_long_only_nav.csv")
    df.columns = ["date", "nav"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    rets = df["nav"].pct_change().dropna()
    rf_m = (1 + rf_annual) ** (1 / 12) - 1

    rows = []
    for start, end, label in [
        ("2007-01-01", "2010-12-31", "2007-2010 incl. GFC"),
        ("2011-01-01", "2014-12-31", "2011-2014"),
        ("2015-01-01", "2019-12-31", "2015-2019"),
        ("2020-01-01", "2024-12-31", "2020-2024"),
    ]:
        sub = rets.loc[start:end]
        if sub.empty:
            continue
        mu = sub.mean() * 12
        sd = sub.std() * np.sqrt(12)
        sharpe = (mu - rf_annual) / sd if sd > 0 else np.nan
        rows.append({"period": label, "start": start[:7], "end": end[:7],
                     "n_months": int(len(sub)),
                     "ann_return": float(mu), "ann_vol": float(sd),
                     "sharpe": float(sharpe)})
    return pd.DataFrame(rows)


def random_portfolio_benchmark(*, n: int = 5000, k: int = 4,
                                seed: int = 20260519,
                                start: str = "2007-01-01",
                                end: str = "2024-12-31",
                                tcost_bps: float = 20.0) -> dict[str, object]:
    """Sharpe distribution under random 4-country long-only baskets."""
    cs = countries()
    tradable_tickers = [c.etf_ticker for c in cs.values()
                        if c.tradable and c.etf_ticker]
    prices = yf_io.fetch()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices[(prices["date"] >= start) & (prices["date"] <= end)]
    wide = (prices.pivot_table(index="date", columns="ticker", values="adj_close")
                 .sort_index())
    monthly = wide.resample("ME").last()
    monthly = monthly[[t for t in tradable_tickers if t in monthly.columns]]
    rets = monthly.pct_change().dropna(how="all")
    # Russia ETF: hard cut Feb-2022
    if "RSX" in rets.columns:
        rets.loc[rets.index > pd.Timestamp("2022-02-28"), "RSX"] = np.nan

    rng = np.random.default_rng(seed)
    universe = list(rets.columns)
    sharpes = []
    for _ in range(n):
        pick = list(rng.choice(universe, size=k, replace=False))
        port = rets[pick].mean(axis=1).dropna()
        if len(port) < 24:
            continue
        mu = port.mean() * 12
        sd = port.std() * np.sqrt(12)
        if sd > 0:
            sharpes.append(mu / sd)
    arr = np.asarray(sharpes, dtype=float)
    arr = arr[np.isfinite(arr)]

    summary = pd.read_csv(OUT_TABLES / "portfolio_summary.csv")
    kpi_sharpe = float(summary[summary["strategy"] == "long_only_top_quintile"][
                       "sharpe"].iloc[0])
    pct = float((arr < kpi_sharpe).mean())
    out = {
        "kpi_sharpe": kpi_sharpe,
        "percentile_rank": pct,
        "random_mean": float(arr.mean()),
        "random_std": float(arr.std()),
        "random_p05": float(np.quantile(arr, 0.05)),
        "random_p95": float(np.quantile(arr, 0.95)),
        "n_random": int(len(arr)),
    }
    log.info("Random-portfolio benchmark: KPI Sharpe %.3f at percentile %.2f "
             "(random mean %.3f, std %.3f)",
             out["kpi_sharpe"], out["percentile_rank"],
             out["random_mean"], out["random_std"])
    return out, arr


def imputation_tier_counts() -> dict[str, int]:
    """Re-derive the imputation pipeline contributions from raw vs imputed."""
    raw = pd.read_parquet(DATA_PROCESSED / "panel_raw.parquet")
    imp = pd.read_parquet(DATA_PROCESSED / "panel_imputed.parquet")
    raw_keys = set(zip(raw["iso3"], raw["year"], raw["indicator"]))
    imp_keys = set(zip(imp["iso3"], imp["year"], imp["indicator"]))
    added = imp_keys - raw_keys
    out = {"raw_cells": len(raw_keys),
           "imputed_cells": len(imp_keys),
           "filled_by_pipeline": len(added)}
    log.info("Imputation pipeline filled %d cells (raw=%d, imputed=%d)",
             out["filled_by_pipeline"], out["raw_cells"], out["imputed_cells"])
    return out


def run_all(imputed_panel: pd.DataFrame) -> dict[str, object]:
    ensure_dirs()
    ir = imputed_vs_raw(imputed_panel)
    pd.DataFrame([ir]).to_csv(OUT_TABLES / "robustness_imputed_vs_raw.csv",
                               index=False)

    sub = subperiod_sharpe()
    sub.to_csv(OUT_TABLES / "robustness_subperiod_sharpe.csv", index=False)

    rand, arr = random_portfolio_benchmark()
    pd.DataFrame([rand]).to_csv(OUT_TABLES / "robustness_random_portfolios.csv",
                                 index=False)
    pd.Series(arr, name="random_sharpe").to_csv(
        OUT_TABLES / "robustness_random_portfolios_distribution.csv", index=False)

    tier = imputation_tier_counts()
    pd.DataFrame([tier]).to_csv(OUT_TABLES / "robustness_imputation_tiers.csv",
                                 index=False)
    return {"imputed_vs_raw": ir, "subperiod_sharpe": sub,
            "random": rand, "tier_counts": tier}
