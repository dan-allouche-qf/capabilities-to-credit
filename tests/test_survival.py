"""Tests for analysis/survival.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from newperformers.analysis import survival as surv


def _toy_panel(n_countries: int = 8, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    iso3s = [f"C{i:02d}" for i in range(n_countries)]
    comp_rows = []
    panel_rows = []
    sectors = surv.SECTORS
    for k, iso3 in enumerate(iso3s):
        baseline = rng.normal(size=len(sectors))
        for year in range(1995, 2025):
            sec_scores = baseline + 0.02 * (year - 1995)
            row = {"iso3": iso3, "year": year, "composite": float(sec_scores.mean())}
            row.update({s: float(sec_scores[i]) for i, s in enumerate(sectors)})
            comp_rows.append(row)
            base_rating = 8.0 + 1.5 * sec_scores.mean() + 0.05 * (year - 1995)
            base_rating += 4.0 if k % 2 == 0 else 0.0
            panel_rows.append({"iso3": iso3, "year": year,
                                "indicator": "SP_RATING",
                                "value": float(np.clip(base_rating, 1, 22))})
    return pd.DataFrame(comp_rows), pd.DataFrame(panel_rows)


def test_event_table_columns():
    comp, panel = _toy_panel()
    ev = surv.event_table(comp, panel)
    expected = {"iso3", "duration", "event", "entry_year", "event_year"}
    assert expected.issubset(set(ev.columns))
    assert (ev["duration"] >= 1).all()
    assert ev["event"].isin([0, 1]).all()


def test_kaplan_meier_two_arms():
    comp, panel = _toy_panel()
    ev = surv.event_table(comp, panel)
    km = surv.kaplan_meier(ev, split_by="energy")
    assert set(km["arm"].unique()).issubset({"high", "low"})
    assert (km["survival"] <= 1.0).all()
    assert (km["survival"] >= 0.0).all()


def test_cox_ph_runs():
    comp, panel = _toy_panel(n_countries=12)
    ev = surv.event_table(comp, panel)
    cox = surv.cox_ph(ev)
    assert not cox.empty
    assert "hazard_ratio" in cox.columns
    assert (cox["hazard_ratio"] > 0).all()
