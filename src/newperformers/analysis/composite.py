"""Composite KPI scoring.

For each sector ``s``, the score is built as follows:

    1. Each indicator is sign-aligned by its YAML ``direction`` (higher
       means better).
    2. Each indicator is z-scored cross-sectionally *within each year*
       (across countries).
    3. PCA is fit on the country-year matrix; the first principal component
       is the sector score, with sign flipped so that higher indicator
       values give a higher score.
    4. The composite score is the equal-weighted mean of the six sector
       scores.

Robustness:
    - jackknife by sector (Kendall's tau on rank vs full sample)
    - jackknife by country
    - bootstrap 95% CI on the final composite rank
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.decomposition import PCA

from ..utils.config import Indicator, kpi_indicators
from ..utils.logging import get_logger
from ..utils.paths import OUT_TABLES, ensure_dirs

log = get_logger(__name__)


@dataclass
class SectorFit:
    sector: str
    indicators: list[str]
    pc1_explained: float
    loadings: dict[str, float]
    score: pd.DataFrame  # (iso3, year, score)


def _sectors() -> dict[str, list[Indicator]]:
    out: dict[str, list[Indicator]] = {}
    for ind in kpi_indicators():
        if ind.role != "core":
            continue
        out.setdefault(ind.sector, []).append(ind)
    return out


def _apply_transforms(panel: pd.DataFrame, indicators: list[Indicator]) -> pd.DataFrame:
    """Apply per-indicator transforms (log1p, log_per_capita) in-place."""
    df = panel.copy()
    pop = (df[df["indicator"] == "SP.POP.TOTL"][["iso3", "year", "value"]]
           .rename(columns={"value": "population"}))
    for ind in indicators:
        if ind.transform == "identity":
            continue
        mask = df["indicator"] == ind.code
        rows = df.loc[mask]
        if ind.transform == "log1p":
            df.loc[mask, "value"] = np.log1p(rows["value"].clip(lower=0))
        elif ind.transform == "log_per_capita":
            merged = rows.merge(pop, on=["iso3", "year"], how="left")
            per_cap = merged["value"] / merged["population"]
            df.loc[mask, "value"] = np.log1p((per_cap * 1e6).clip(lower=0)).values
        else:
            raise ValueError(f"unknown transform: {ind.transform}")
    return df


def _sign_align(panel: pd.DataFrame, indicators: list[Indicator]) -> pd.DataFrame:
    df = panel.copy()
    for ind in indicators:
        if ind.direction == "lower_is_better":
            mask = df["indicator"] == ind.code
            df.loc[mask, "value"] = -df.loc[mask, "value"]
    return df


def _standardize(df_wide: pd.DataFrame) -> pd.DataFrame:
    """Standardise each column over the full panel (country-year).

    Panel-wide z-scoring keeps both cross-sectional and time-series variation,
    so countries that improve absolutely show an increase in their score.
    Year-by-year standardisation removes secular trends — useful for ranks,
    but not for evolution plots.
    """
    z = df_wide.copy()
    mean = z.mean()
    std = z.std(ddof=0).replace(0.0, np.nan)
    return (z - mean) / std


def _pivot(panel: pd.DataFrame, indicators: list[Indicator]) -> pd.DataFrame:
    sub = panel[panel["indicator"].isin([i.code for i in indicators])]
    wide = sub.pivot_table(index=["iso3", "year"], columns="indicator", values="value")
    return wide.reindex(columns=[i.code for i in indicators])


def _fit_sector(panel: pd.DataFrame, indicators: list[Indicator]) -> SectorFit:
    wide = _pivot(panel, indicators)
    wide = _standardize(wide)
    wide = wide.dropna(how="all", axis=1)
    wide_complete = wide.dropna()
    if wide_complete.empty:
        raise RuntimeError("No complete observations for sector.")
    pca = PCA(n_components=1, random_state=20260519)
    pca.fit(wide_complete.values)
    loadings = pd.Series(pca.components_[0], index=wide_complete.columns)
    if loadings.mean() < 0:
        loadings = -loadings
    # PC1 scores live on the standardised scale: aggregate with each row's
    # observed columns and rescale by the share of non-null entries to keep
    # countries with partial coverage on the same footing.
    sub = wide[loadings.index]
    weights = loadings.to_numpy()
    mat = sub.to_numpy()
    mask = ~np.isnan(mat)
    filled = np.where(mask, mat, 0.0)
    raw_scores = filled @ weights
    coverage = mask.astype(float) @ np.abs(weights)
    coverage[coverage == 0] = np.nan
    scores = raw_scores * (np.abs(weights).sum() / coverage)
    out = (sub.assign(score=scores).reset_index()[["iso3", "year", "score"]])
    return SectorFit(
        sector=indicators[0].sector,
        indicators=list(loadings.index),
        pc1_explained=float(pca.explained_variance_ratio_[0]),
        loadings=loadings.to_dict(),
        score=out,
    )


def sector_scores(panel: pd.DataFrame) -> dict[str, SectorFit]:
    transformed = _apply_transforms(panel, kpi_indicators())
    aligned = _sign_align(transformed, kpi_indicators())
    fits: dict[str, SectorFit] = {}
    for sector, inds in _sectors().items():
        fits[sector] = _fit_sector(aligned, inds)
        log.info("Sector '%s' PC1 explained variance = %.3f",
                 sector, fits[sector].pc1_explained)
    return fits


# AHP-style alternative weights motivated by the institutional-quality
# literature (Acemoglu-Robinson 2012, Kaufmann-Kraay-Mastruzzi 2010).
# Heavier on security and R&I as foundational growth drivers; lighter on
# housing and energy infrastructure as derived outputs. Sums to 1.
AHP_WEIGHTS: dict[str, float] = {
    "security_stability": 0.25,
    "research_innovation": 0.20,
    "education": 0.20,
    "health": 0.15,
    "energy": 0.10,
    "housing_living": 0.10,
}


def composite_score(fits: dict[str, SectorFit],
                     weights: dict[str, float] | None = None) -> pd.DataFrame:
    frames = []
    for sector, fit in fits.items():
        df = fit.score.copy()
        df["sector"] = sector
        frames.append(df)
    long = pd.concat(frames, ignore_index=True)
    wide = long.pivot_table(index=["iso3", "year"], columns="sector", values="score")
    if weights is None:
        composite = wide.mean(axis=1).rename("composite")
    else:
        w = pd.Series(weights).reindex(wide.columns).fillna(0.0)
        w = w / w.sum()
        composite = (wide * w).sum(axis=1).rename("composite")
    out = wide.join(composite).reset_index()
    return out


def ahp_rank_comparison(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute Kendall τ between equal-weight and AHP-weighted composite ranks.
    Returns one row per year with the τ over the cross-section.
    """
    from scipy.stats import kendalltau

    fits = sector_scores(panel)
    eq = composite_score(fits).set_index(["iso3", "year"])["composite"]
    ahp = composite_score(fits, weights=AHP_WEIGHTS).set_index(["iso3", "year"])["composite"]
    common = eq.index.intersection(ahp.index)
    rows = []
    for year in sorted(set(y for _, y in common)):
        sub_eq = eq.loc[(slice(None), year)].rank(ascending=False)
        sub_ahp = ahp.loc[(slice(None), year)].rank(ascending=False)
        idx = sub_eq.index.intersection(sub_ahp.index)
        if len(idx) < 5:
            continue
        tau, p = kendalltau(sub_eq.loc[idx], sub_ahp.loc[idx])
        rows.append({"year": int(year), "kendall_tau": float(tau),
                      "p_value": float(p), "n": int(len(idx))})
    return pd.DataFrame(rows)


def latest_ranking(composite: pd.DataFrame, year: int | None = None) -> pd.DataFrame:
    if year is None:
        year = composite["year"].max()
    snap = composite[composite["year"] == year].copy()
    snap["rank"] = snap["composite"].rank(ascending=False, method="min").astype(int)
    return snap.sort_values("rank").reset_index(drop=True)


def jackknife_sector(panel: pd.DataFrame) -> pd.DataFrame:
    """Drop each sector in turn; report Kendall tau on the latest-year rank."""
    fits = sector_scores(panel)
    full = composite_score(fits)
    base_rank = latest_ranking(full).set_index("iso3")["rank"]
    rows = []
    for drop in fits:
        subset = {k: v for k, v in fits.items() if k != drop}
        comp = composite_score(subset)
        rank = latest_ranking(comp).set_index("iso3")["rank"]
        common = base_rank.index.intersection(rank.index)
        tau, _ = kendalltau(base_rank.loc[common], rank.loc[common])
        rows.append({"dropped_sector": drop, "kendall_tau": float(tau)})
    return pd.DataFrame(rows).sort_values("kendall_tau", ascending=False)


def bootstrap_rank_ci(panel: pd.DataFrame, n: int = 500, seed: int = 20260519,
                      year: int | None = None) -> pd.DataFrame:
    """Bootstrap by resampling indicators within each sector (block bootstrap)."""
    rng = np.random.default_rng(seed)
    fits = sector_scores(panel)
    base = latest_ranking(composite_score(fits), year=year).set_index("iso3")
    if year is None:
        year = int(panel["year"].max())

    samples: list[pd.Series] = []
    sectors = _sectors()
    for _ in range(n):
        boot_fits = {}
        for sector, inds in sectors.items():
            codes = [i.code for i in inds]
            idx = rng.integers(0, len(codes), size=len(codes))
            boot_inds = [inds[i] for i in idx]
            if len({i.code for i in boot_inds}) < 2:
                boot_inds = inds
            boot_fits[sector] = _fit_sector(_sign_align(panel, boot_inds), boot_inds)
        comp = composite_score(boot_fits)
        rank = latest_ranking(comp, year=year).set_index("iso3")["rank"]
        samples.append(rank)
    boot = pd.concat(samples, axis=1)
    summary = pd.DataFrame({
        "rank_mean": boot.mean(axis=1),
        "rank_p05": boot.quantile(0.05, axis=1),
        "rank_p95": boot.quantile(0.95, axis=1),
    })
    return base.join(summary, how="left").reset_index()


def save_tables(fits: dict[str, SectorFit], composite: pd.DataFrame) -> None:
    ensure_dirs()
    loadings_rows = []
    for sector, fit in fits.items():
        for code, weight in fit.loadings.items():
            loadings_rows.append({
                "sector": sector,
                "indicator": code,
                "loading": float(weight),
                "pc1_explained": fit.pc1_explained,
            })
    pd.DataFrame(loadings_rows).to_csv(OUT_TABLES / "sector_loadings.csv", index=False)
    composite.to_csv(OUT_TABLES / "composite_score.csv", index=False)
    log.info("Saved sector loadings and composite scores to %s", OUT_TABLES)
