"""Final cleanup before the panel is consumed by analysis modules.

Operations:
    1. Trim to the canonical country universe.
    2. Trim to the configured year range.
    3. Coerce dtypes and drop duplicate (iso3, year, indicator) keys, keeping
       the latest non-null value.
    4. Drop rows with NaN values (those are reintroduced after imputation).
"""

from __future__ import annotations

import pandas as pd

from . import schema
from ..utils.config import all_iso3


def harmonize(panel: pd.DataFrame, *, start: int, end: int) -> pd.DataFrame:
    iso3s = set(all_iso3())
    out = panel[panel["iso3"].isin(iso3s)].copy()
    out = out[(out["year"] >= start) & (out["year"] <= end)]
    out = out.dropna(subset=["value"])
    out = (out
           .sort_values(["iso3", "indicator", "year"])
           .drop_duplicates(subset=["iso3", "year", "indicator"], keep="last"))
    return schema.coerce(out.reset_index(drop=True), source="harmonized")
