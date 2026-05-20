"""Tests for analysis/rating_change.py.

All tests isolate writes to a temporary directory so the production
``outputs/tables/`` artefacts are never overwritten by toy-data runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from newperformers.analysis import rating_change as rc


@pytest.fixture
def isolated_out_tables(tmp_path, monkeypatch):
    """Redirect OUT_TABLES to ``tmp_path`` for the duration of one test."""
    from newperformers.utils import paths as paths_mod
    monkeypatch.setattr(paths_mod, "OUT_TABLES", tmp_path)
    monkeypatch.setattr(rc, "OUT_TABLES", tmp_path)
    return tmp_path


def _toy_data(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    sectors = rc.SECTORS
    iso3s = [f"C{i:02d}" for i in range(10)]
    comp_rows = []
    panel_rows = []
    for iso3 in iso3s:
        rating = 10.0 + rng.normal()
        for year in range(2000, 2024):
            scores = rng.normal(size=6)
            row = {"iso3": iso3, "year": year}
            row.update({s: float(scores[i]) for i, s in enumerate(sectors)})
            comp_rows.append(row)
            move = rng.choice([-1, 0, 0, 0, 1], p=[0.05, 0.85, 0.0, 0.0, 0.10])
            rating = float(np.clip(rating + move, 1, 22))
            panel_rows.append({"iso3": iso3, "year": year,
                                "indicator": "SP_RATING",
                                "value": rating})
    return pd.DataFrame(comp_rows), pd.DataFrame(panel_rows)


def test_build_panel_returns_delta_class():
    comp, panel = _toy_data()
    df = rc._build_panel(comp, panel)
    assert "delta_class" in df.columns
    assert df["delta_class"].isin([-1, 0, 1]).all()


def test_oos_predictions_match_classes(isolated_out_tables):
    comp, panel = _toy_data()
    out = rc.fit_oos(comp, panel, start_year=2010, end_year=2015)
    if not out:
        return
    preds = pd.read_csv(isolated_out_tables / "rating_change_predictions.csv")
    assert preds["actual_class"].isin([-1, 0, 1]).all()
    assert preds["predicted_class"].isin([-1, 0, 1]).all()
