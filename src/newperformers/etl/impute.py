"""Imputation pipeline for the master panel.

Tiered, per-(country, indicator) series:
    1. Linear interpolation for interior gaps up to ``max_gap`` years.
    2. Forward-fill the last few years (handle one-year publication lag).
    3. MICE (sklearn IterativeImputer) across indicators within a sector,
       per country, when at least 30% of years are observed for that
       indicator in that country.
    4. Otherwise leave NaN; downstream models drop those observations.

Returns two parallel panels: the raw harmonized one and the imputed one,
so robustness checks can compare results across imputation regimes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

from . import schema
from ..utils.config import (
    all_iso3,
    kpi_indicators,
    macro_indicators,
    outcomes,
)
from ..utils.logging import get_logger

log = get_logger(__name__)


def _pivot(panel: pd.DataFrame, codes: list[str], iso3s: list[str], years: range) -> pd.DataFrame:
    sub = panel[panel["indicator"].isin(codes) & panel["iso3"].isin(iso3s)]
    wide = (sub.pivot_table(index=["iso3", "year"], columns="indicator", values="value")
              .reindex(index=pd.MultiIndex.from_product([iso3s, list(years)],
                                                        names=["iso3", "year"]))
              .reindex(columns=codes))
    return wide


def _unpivot(wide: pd.DataFrame, source: str) -> pd.DataFrame:
    long = wide.stack().rename("value").reset_index()
    long.columns = ["iso3", "year", "indicator", "value"]
    long["source"] = source
    return schema.coerce(long, source=source)


def _interpolate_and_ffill(wide: pd.DataFrame, max_gap: int) -> pd.DataFrame:
    out = wide.copy()
    for iso3, block in out.groupby(level="iso3"):
        block = block.sort_index(level="year")
        block = block.interpolate(method="linear", limit=max_gap, limit_area="inside")
        block = block.ffill(limit=3)
        out.loc[block.index] = block
    return out


def _mice_within_sector(wide: pd.DataFrame, min_coverage: float, seed: int) -> pd.DataFrame:
    """Apply iterative imputation to columns that meet the coverage threshold,
    country by country. Sparser columns are left untouched.
    """
    out = wide.copy()
    for iso3, block in out.groupby(level="iso3"):
        coverage = block.notna().mean(axis=0)
        keep = coverage[coverage >= min_coverage].index.tolist()
        if len(keep) < 2:
            continue
        mat = block[keep].to_numpy(dtype=float)
        if np.isnan(mat).sum() == 0:
            continue
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=20,
            random_state=seed,
            tol=1e-3,
            initial_strategy="median",
        )
        try:
            filled = imputer.fit_transform(mat)
        except Exception as exc:  # noqa: BLE001
            log.warning("MICE failed for %s (%s)", iso3, exc)
            continue
        out.loc[block.index, keep] = filled
    return out


def impute(panel: pd.DataFrame, *, start: int, end: int,
           max_gap: int = 3, min_coverage: float = 0.30,
           seed: int = 20260519) -> pd.DataFrame:
    """Produce the imputed long-format panel."""
    years = range(start, end + 1)
    iso3s = all_iso3()

    sector_groups: dict[str, list[str]] = {}
    for ind in kpi_indicators():
        sector_groups.setdefault(f"kpi_{ind.sector}", []).append(ind.code)
    for ind in macro_indicators():
        sector_groups.setdefault(f"macro_{ind.sector}", []).append(ind.code)
    sector_groups["outcomes"] = [o.code for o in outcomes()]

    imputed_frames: list[pd.DataFrame] = []
    grouped_codes = set()
    for sector, codes in sector_groups.items():
        codes = list(dict.fromkeys(codes))
        grouped_codes.update(codes)
        wide = _pivot(panel, codes, iso3s, years)
        wide = _interpolate_and_ffill(wide, max_gap=max_gap)
        wide = _mice_within_sector(wide, min_coverage=min_coverage, seed=seed)
        long = _unpivot(wide, source=f"imputed_{sector}")
        imputed_frames.append(long)
        log.info("Imputed %s: %d non-null cells", sector, long["value"].notna().sum())

    # Pass-through: any indicator not assigned to a sector group (e.g. WGI
    # series, S&P sovereign rating) is shipped through with light cleaning
    # but without imputation.
    other_codes = sorted(set(panel["indicator"].unique()) - grouped_codes)
    if other_codes:
        wide = _pivot(panel, other_codes, iso3s, years)
        wide = _interpolate_and_ffill(wide, max_gap=max_gap)
        imputed_frames.append(_unpivot(wide, source="passthrough"))
        log.info("Pass-through: %d additional indicators (WGI, ratings, ...)",
                 len(other_codes))

    out = pd.concat(imputed_frames, ignore_index=True)
    return schema.coerce(out, source="imputed")
