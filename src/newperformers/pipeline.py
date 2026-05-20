"""End-to-end CLI: ``python -m newperformers.pipeline [stage]``.

Stages
------
data       Fetch the raw panel and write data/processed/panel_*.parquet.
analysis   Composite + local projections (saves outputs/tables/*.csv).
figures    Render the paper-grade PDF figures.
all        Run the three above in order.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

from .analysis import cointegration as coint_mod
from .analysis import composite as comp_mod
from .analysis import composite_lasso as lasso_mod
from .analysis import credit as credit_mod
from .analysis import credit_bayesian as bayes_mod
from .analysis import granger as granger_mod
from .analysis import local_projections as lp_mod
from .analysis import portfolio as portfolio_mod
from .analysis import portfolio_credit as pc_mod
from .analysis import rating_change as rc_mod
from .analysis import robustness as robust_mod
from .analysis import survival as surv_mod
from .analysis import synthetic_control as syn_mod
from .etl import merge as merge_mod
from .etl import validate as val_mod
from .io import yfinance as yf_io
from .utils import seeds
from .utils.logging import get_logger
from .utils.paths import OUT_TABLES, ensure_dirs
from .viz import figures_paper

log = get_logger(__name__)


def stage_data(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("=== Stage: data ===")
    raw, imputed = merge_mod.build(refresh=refresh)
    diags = val_mod.report(raw, start=1990, end=2024)
    ensure_dirs()
    diags["coverage"].to_csv(OUT_TABLES / "coverage_matrix.csv")
    log.info("Wrote outputs/tables/coverage_matrix.csv")
    return raw, imputed


def stage_analysis(imputed: pd.DataFrame) -> dict[str, object]:
    log.info("=== Stage: analysis ===")
    fits = comp_mod.sector_scores(imputed)
    composite = comp_mod.composite_score(fits)
    comp_mod.save_tables(fits, composite)

    jack = comp_mod.jackknife_sector(imputed)
    jack.to_csv(OUT_TABLES / "composite_jackknife.csv", index=False)
    log.info("Jackknife (rank-stability when dropping each sector):\n%s",
             jack.to_string(index=False))

    shocks = list(fits.keys())  # sector scores feed the LP as standardised shocks
    sector_df = composite[["iso3", "year"] + shocks].melt(
        id_vars=["iso3", "year"], var_name="indicator", value_name="value")
    sector_df["source"] = "sector_score"

    # LPs are estimated on log-levels of stock variables (GDP per capita) so
    # that y_{t+h} − y_{t-1} reads as cumulative log growth. Rate / ratio
    # variables (Gini, poverty headcount) stay in their natural units.
    gdppc = imputed[imputed["indicator"] == "NY.GDP.PCAP.CD"][
        ["iso3", "year", "value"]].copy()
    gdppc["value"] = np.log(gdppc["value"].clip(lower=1.0)) * 100.0
    gdppc["indicator"] = "log_gdppc_x100"
    gdppc["source"] = "derived"
    panel_with_shocks = pd.concat([imputed, sector_df, gdppc], ignore_index=True)

    lp_results = lp_mod.run_grid(
        panel_with_shocks,
        shocks=shocks,
        outcomes=["log_gdppc_x100", "SI.POV.DDAY", "SI.POV.GINI"],
        horizons=8,
        controls=["FP.CPI.TOTL.ZG"],
    )
    from scipy.stats import norm
    rows = []
    for (sh, out), res in lp_results.items():
        for h, b, se in zip(res.horizons, res.beta, res.se, strict=True):
            tstat = float(b / se) if se > 0 and np.isfinite(se) else float("nan")
            pvalue = float(2 * (1 - norm.cdf(abs(tstat)))) if np.isfinite(tstat) else float("nan")
            if not np.isfinite(pvalue):
                sig = ""
            elif pvalue < 0.01:
                sig = "***"
            elif pvalue < 0.05:
                sig = "**"
            elif pvalue < 0.10:
                sig = "*"
            else:
                sig = ""
            rows.append({"shock": sh, "outcome": out, "horizon": int(h),
                         "beta": float(b), "se": float(se),
                         "tstat": tstat, "pvalue": pvalue, "sig_flag": sig})
    pd.DataFrame(rows).to_csv(OUT_TABLES / "lp_irf.csv", index=False)
    log.info("Wrote outputs/tables/lp_irf.csv with t-stats and p-values")

    credit_out = credit_mod.fit_models(composite, imputed)

    log.info("Running Bayesian hierarchical credit model (NUTS, this takes ~1-3 min)…")
    bayes_out = bayes_mod.fit(composite, imputed)

    log.info("Running synthetic-control episodes (Rwanda 2000, Singapore 1990)…")
    syn_results = syn_mod.run_default_episodes(imputed)

    log.info("Running Dumitrescu-Hurlin panel Granger…")
    granger_df = granger_mod.run_grid(
        panel_with_shocks,
        shocks=shocks,
        outcomes=["log_gdppc_x100", "SI.POV.DDAY", "SI.POV.GINI"],
        p=2,
    )

    log.info("Running panel unit-root + cointegration tests (CIPS, Westerlund)…")
    coint_df = coint_mod.run_panel_tests(imputed, composite)

    log.info("Running Lasso composite weights…")
    lasso_out = lasso_mod.run_all(composite, imputed)

    log.info("Running survival analysis on time-to-IG…")
    surv_out = surv_mod.run_all(composite, imputed)

    log.info("Running rating-change prediction…")
    rc_out = rc_mod.fit_oos(composite, imputed)

    log.info("Running credit portfolio extension…")
    pc_out = pc_mod.run_all(composite)

    return {"composite": composite, "fits": fits, "lp": lp_results,
            "credit": credit_out, "bayes": bayes_out,
            "synthetic_control": syn_results}


def stage_portfolio(composite: pd.DataFrame) -> dict[str, object]:
    log.info("=== Stage: portfolio ===")
    try:
        yf_io.fetch()
    except Exception as exc:  # noqa: BLE001
        log.warning("yfinance fetch failed: %s — skipping portfolio stage.", exc)
        return {}
    return portfolio_mod.run(composite)


def stage_robustness(imputed: pd.DataFrame) -> dict[str, object]:
    log.info("=== Stage: robustness ===")
    return robust_mod.run_all(imputed)


def stage_figures(raw: pd.DataFrame, imputed: pd.DataFrame,
                  composite: pd.DataFrame, coverage: pd.DataFrame) -> None:
    log.info("=== Stage: figures ===")
    figures_paper.coverage_heatmap(coverage)
    figures_paper.composite_evolution(composite)
    figures_paper.composite_rank_journey(composite)
    figures_paper.improvement_vs_starting_point(composite)
    figures_paper.composite_dispersion(composite)
    figures_paper.sector_scores_small_multiples(composite)
    figures_paper.sector_change_heatmap(composite)
    figures_paper.gdp_slope_chart(imputed)
    figures_paper.macro_panel(imputed)
    figures_paper.pca_biplots(imputed)
    figures_paper.lp_irf_panel()
    figures_paper.rating_heatmap(raw)
    figures_paper.composite_vs_rating_scatter(composite, raw)
    figures_paper.energy_health_credit_map(composite, raw)
    figures_paper.credit_oos_calibration()
    figures_paper.credit_oos_scorecard()
    try:
        figures_paper.bayesian_loadings()
    except FileNotFoundError as exc:
        log.warning("Bayesian figure skipped (run analysis first): %s", exc)
    for tag, title, year in [
        ("rwanda_2000", "Rwanda — Vision 2020", 2000),
        ("singapore_1990", "Singapore counterfactual", 1990),
    ]:
        try:
            figures_paper.synthetic_control_plot(tag, title, year)
        except FileNotFoundError as exc:
            log.warning("Synthetic-control figure '%s' skipped: %s", tag, exc)
    try:
        figures_paper.portfolio_nav()
        figures_paper.portfolio_drawdown()
        figures_paper.drawdown_overlap(imputed)
        figures_paper.quintile_navs()
        figures_paper.risk_decomposition_bars()
        figures_paper.rolling_sharpe()
    except FileNotFoundError as exc:
        log.warning("Portfolio figures skipped (run pipeline portfolio first): %s", exc)
    for fn, label in [
        (figures_paper.bootstrap_sharpe_vs_random, "bootstrap-vs-random"),
        (figures_paper.auc_curve, "AUC curve"),
        (figures_paper.marginal_effects, "marginal effects"),
        (figures_paper.policy_timeline, "policy timeline"),
        (figures_paper.granger_heatmap, "Granger heatmap"),
        (figures_paper.survival_km, "survival KM"),
        (figures_paper.cox_hazards_forest, "Cox PH forest"),
        (figures_paper.rating_change_confusion, "rating-change confusion"),
        (figures_paper.selection_flowchart, "selection flowchart"),
    ]:
        try:
            fn()
        except FileNotFoundError as exc:
            log.warning("Figure %s skipped: %s", label, exc)
    try:
        figures_paper.country_dashboards(composite, imputed)
    except Exception as exc:  # noqa: BLE001
        log.warning("Country dashboards skipped: %s", exc)


def main() -> int:
    parser = argparse.ArgumentParser(prog="newperformers.pipeline")
    parser.add_argument("stage",
                        choices=["data", "analysis", "portfolio", "robustness",
                                  "figures", "all"],
                        default="all", nargs="?")
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-pull of cached data")
    args = parser.parse_args()
    seeds.set_global_seed()
    t0 = time.time()

    raw, imputed = stage_data(args.refresh) if args.stage in {"data", "all"} else merge_mod.load()

    composite = None
    if args.stage in {"analysis", "all"}:
        out = stage_analysis(imputed)
        composite = out["composite"]
    elif args.stage in {"portfolio", "figures"}:
        composite = pd.read_csv(OUT_TABLES / "composite_score.csv")

    if args.stage in {"portfolio", "all"}:
        stage_portfolio(composite)

    if args.stage in {"robustness", "all"}:
        stage_robustness(imputed)

    if args.stage in {"figures", "all"}:
        coverage = val_mod.coverage_matrix(raw, start=1990, end=2024)
        stage_figures(raw, imputed, composite, coverage)

    log.info("Done in %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
