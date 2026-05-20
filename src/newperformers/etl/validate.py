"""Range checks and coverage diagnostics for the master panel."""

from __future__ import annotations

import pandas as pd

from ..utils.config import (
    all_iso3,
    kpi_indicators,
    macro_indicators,
    outcomes,
)
from ..utils.logging import get_logger

log = get_logger(__name__)

# Inclusive [min, max] ranges flagged when violated. Values *outside* the
# range are surfaced as warnings, not dropped, so the underlying issue
# (unit mismatch, scaling bug) can be investigated.
RANGES: dict[str, tuple[float, float]] = {
    "SP.DYN.LE00.IN": (25.0, 95.0),
    "SP.DYN.IMRT.IN": (0.0, 200.0),
    "SH.DYN.MORT": (0.0, 300.0),
    "SE.ADT.LITR.ZS": (0.0, 100.0),
    "SE.PRM.CMPT.ZS": (0.0, 200.0),
    "SE.TER.ENRR": (0.0, 200.0),
    "SH.IMM.MEAS": (0.0, 100.0),
    "EG.ELC.ACCS.ZS": (0.0, 100.0),
    "SI.POV.GINI": (15.0, 75.0),
    "SI.POV.DDAY": (0.0, 100.0),
    "FP.CPI.TOTL.ZG": (-50.0, 1500.0),
    "NY.GDP.MKTP.KD.ZG": (-30.0, 30.0),
    "WGI_PV": (-3.0, 3.0),
    "WGI_RL": (-3.0, 3.0),
    "WGI_CC": (-3.0, 3.0),
    "WGI_GE": (-3.0, 3.0),
    "WGI_VA": (-3.0, 3.0),
    "WGI_RQ": (-3.0, 3.0),
    "SP_RATING": (1.0, 22.0),
}


def out_of_range(panel: pd.DataFrame) -> pd.DataFrame:
    """Return rows whose value falls outside an indicator's expected range."""
    out: list[pd.DataFrame] = []
    for ind, (lo, hi) in RANGES.items():
        sl = panel[panel["indicator"] == ind]
        bad = sl[(sl["value"] < lo) | (sl["value"] > hi)]
        if not bad.empty:
            out.append(bad)
    return pd.concat(out, ignore_index=True) if out else panel.iloc[0:0]


def coverage_matrix(panel: pd.DataFrame, *, start: int, end: int) -> pd.DataFrame:
    """Country x indicator matrix of coverage ratio in [start, end]."""
    years = pd.Index(range(start, end + 1), name="year")
    indicators = sorted(panel["indicator"].unique())
    iso3s = all_iso3()
    rows = []
    for iso3 in iso3s:
        sub = panel[panel["iso3"] == iso3]
        row: dict[str, float | str] = {"iso3": iso3}
        for ind in indicators:
            vals = sub[sub["indicator"] == ind].set_index("year")["value"].reindex(years)
            row[ind] = float(vals.notna().mean())
        rows.append(row)
    return pd.DataFrame(rows).set_index("iso3")


def report(panel: pd.DataFrame, *, start: int, end: int) -> dict[str, pd.DataFrame]:
    """Run all diagnostics. Used by the pipeline to print a coverage summary."""
    bad = out_of_range(panel)
    cov = coverage_matrix(panel, start=start, end=end)
    expected = {ind.code for ind in (kpi_indicators() + macro_indicators() + outcomes())}
    present = set(panel["indicator"].unique())
    missing = sorted(expected - present)
    log.info("Validation: %d out-of-range rows, %d missing indicators", len(bad), len(missing))
    return {"out_of_range": bad, "coverage": cov, "missing_indicators": pd.Series(missing)}
