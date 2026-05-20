"""Convergence diagnostic test — reads the persisted Bayesian loadings CSV
and asserts that the build is in the publishable convergence regime.

This test only runs after at least one full Bayesian fit; if the CSV is
absent, the test skips. The pytest hook never fabricates diagnostics
from synthetic data — it checks the real posterior the paper cites.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from newperformers.utils.paths import OUT_TABLES


@pytest.fixture(scope="module")
def loadings() -> pd.DataFrame:
    path = OUT_TABLES / "credit_bayesian_loadings.csv"
    if not path.exists():
        pytest.skip("credit_bayesian_loadings.csv missing — run analysis first")
    return pd.read_csv(path)


def test_rhat_below_threshold(loadings: pd.DataFrame):
    assert loadings["r_hat"].max() < 1.05, (
        f"max R̂ = {loadings['r_hat'].max():.3f} exceeds the 1.05 publishability threshold")


def test_ess_above_threshold(loadings: pd.DataFrame):
    assert loadings["ess_bulk"].min() > 200, (
        f"min ESS = {loadings['ess_bulk'].min():.0f} below the 200-sample threshold")


def test_loadings_columns(loadings: pd.DataFrame):
    required = {"sector", "posterior_mean", "hdi_5%", "hdi_95%", "r_hat", "ess_bulk"}
    assert required.issubset(loadings.columns)
