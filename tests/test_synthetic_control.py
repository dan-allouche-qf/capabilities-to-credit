"""Tests for analysis/synthetic_control.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from newperformers.analysis import synthetic_control as syn


def test_fit_returns_pvalues():
    """When given enough donor pre/post coverage the fit should return
    finite p-values."""
    rng = np.random.default_rng(0)
    rows = []
    for c in ["TR1", "DON1", "DON2", "DON3", "DON4", "DON5"]:
        x = 100.0
        for t in range(1990, 2025):
            x = x + 1 + rng.standard_normal() * 0.3
            v = x + (5.0 if c == "TR1" and t >= 2010 else 0.0)
            rows.append({"iso3": c, "year": t,
                          "indicator": "OUTC", "value": float(v)})
    panel = pd.DataFrame(rows)
    res = syn.fit(panel, treated="TR1", treatment_year=2010, outcome="OUTC")
    assert res is not None
    assert np.isfinite(res.pvalue_post), "post p-value should be finite"
    assert np.isfinite(res.pvalue_rmse_ratio), "RMSPE-ratio p-value should be finite"
    assert 0.0 <= res.pvalue_post <= 1.0
    assert 0.0 <= res.pvalue_rmse_ratio <= 1.0


def test_weights_sum_to_one():
    rng = np.random.default_rng(1)
    rows = []
    for c in ["TR1", "DON1", "DON2", "DON3"]:
        base = rng.normal(100, 10)
        for t in range(1990, 2025):
            rows.append({"iso3": c, "year": t, "indicator": "OUTC",
                          "value": float(base + 0.5 * (t - 1990) + rng.standard_normal())})
    panel = pd.DataFrame(rows)
    res = syn.fit(panel, treated="TR1", treatment_year=2010, outcome="OUTC")
    assert res is not None
    total = float(res.donor_weights.sum())
    assert 0.99 <= total <= 1.01, f"weights should sum to ~1, got {total:.3f}"
