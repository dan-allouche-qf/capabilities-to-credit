"""Tests for analysis/composite_lasso.py.

All tests isolate writes to a temporary directory so the production
``outputs/tables/`` artefacts are never overwritten by toy-data runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from newperformers.analysis import composite_lasso as cl


@pytest.fixture
def isolated_out_tables(tmp_path, monkeypatch):
    """Redirect OUT_TABLES to ``tmp_path`` for the duration of one test."""
    from newperformers.utils import paths as paths_mod
    monkeypatch.setattr(paths_mod, "OUT_TABLES", tmp_path)
    monkeypatch.setattr(cl, "OUT_TABLES", tmp_path)
    return tmp_path


def _toy_data(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    sectors = cl.SECTORS
    iso3s = [f"C{i:02d}" for i in range(12)]
    comp_rows = []
    panel_rows = []
    true_w = np.array([0.5, 1.2, 0.2, -0.8, 0.1, 0.3])
    for iso3 in iso3s:
        for year in range(1995, 2025):
            scores = rng.normal(size=6)
            row = {"iso3": iso3, "year": year}
            row.update({s: float(scores[i]) for i, s in enumerate(sectors)})
            comp_rows.append(row)
            rating = 10.0 + scores @ true_w + 0.5 * rng.standard_normal()
            panel_rows.append({"iso3": iso3, "year": year,
                                "indicator": "SP_RATING",
                                "value": float(np.clip(rating, 1, 22))})
    return pd.DataFrame(comp_rows), pd.DataFrame(panel_rows)


def test_fit_lasso_returns_six_coefs(isolated_out_tables):
    comp, panel = _toy_data()
    coefs = cl.fit_lasso(comp, panel)
    assert set(coefs.keys()) == set(cl.SECTORS)
    assert all(np.isfinite(v) for v in coefs.values())


def test_selection_stability_freq_in_unit_interval(isolated_out_tables):
    comp, panel = _toy_data()
    stab = cl.selection_stability(comp, panel, n_boot=30)
    assert (stab["selection_freq"] >= 0.0).all()
    assert (stab["selection_freq"] <= 1.0).all()


def test_compare_weight_schemes_tau_in_minus_one_one(isolated_out_tables):
    comp, panel = _toy_data()
    coefs = cl.fit_lasso(comp, panel)
    out = cl.compare_weight_schemes(comp, coefs)
    assert ((out["kendall_tau"] >= -1.0) & (out["kendall_tau"] <= 1.0)).all()
    assert len(out) == 3
