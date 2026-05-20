"""Tests for analysis/cointegration.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from newperformers.analysis import cointegration as coint


def _make_panel(stationary: bool, n_country: int = 10, n_year: int = 30,
                 seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_country):
        iso = f"X{c:02d}"
        x = rng.standard_normal()
        for t in range(n_year):
            if stationary:
                x = 0.5 * x + rng.standard_normal()
            else:
                x = x + rng.standard_normal()  # random walk
            rows.append({"iso3": iso, "year": 1990 + t,
                          "indicator": "Z", "value": float(x)})
    return pd.DataFrame(rows)


def test_cips_rejects_stationary():
    panel = _make_panel(stationary=True)
    res = coint.cips(panel, "Z", lag=1)
    assert res["cips_stat"] < res["p_005_critical"], (
        f"CIPS should reject stationarity-on-AR(0.5) data, got {res['cips_stat']:.2f}")


def test_cips_fails_to_reject_random_walk():
    panel = _make_panel(stationary=False)
    res = coint.cips(panel, "Z", lag=1)
    assert res["cips_stat"] > res["p_005_critical"], (
        f"CIPS should fail to reject on random walk, got {res['cips_stat']:.2f}")


def test_westerlund_rejects_cointegrated():
    rng = np.random.default_rng(7)
    rows = []
    for c in range(15):
        iso = f"X{c:02d}"
        x = 0.0
        for t in range(35):
            x = x + rng.standard_normal()
            y = x + 0.3 * rng.standard_normal()  # y is cointegrated with x
            rows.append({"iso3": iso, "year": 1990 + t,
                          "indicator": "Y", "value": float(y)})
            rows.append({"iso3": iso, "year": 1990 + t,
                          "indicator": "X", "value": float(x)})
    panel = pd.DataFrame(rows)
    res = coint.westerlund_group_t(panel, "Y", "X", lag=1)
    assert res["pvalue_normal"] < 0.05, (
        f"Westerlund should reject no-cointegration on cointegrated data, got p={res['pvalue_normal']:.3f}")
