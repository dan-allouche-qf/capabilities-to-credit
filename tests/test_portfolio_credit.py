"""Tests for analysis/portfolio_credit.py.

The credit portfolio module relies on yfinance cache + composite data. We
test the helpers directly with synthetic frames so the test suite stays
hermetic and does not hit the network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from newperformers.analysis import portfolio_credit as pc


def test_monthly_returns_resamples_correctly():
    dates = pd.bdate_range("2010-01-01", "2010-12-31")
    df = pd.DataFrame({
        "date": np.tile(dates, 2),
        "ticker": ["X"] * len(dates) + ["Y"] * len(dates),
        "adj_close": np.concatenate(
            [100 * (1 + 0.001 * np.arange(len(dates))),
             50 * (1 + 0.002 * np.arange(len(dates)))]
        ),
    })
    monthly = pc._monthly_returns(df, ["X", "Y"])
    assert "X" in monthly.columns and "Y" in monthly.columns
    assert len(monthly) == 12
    assert monthly.dropna().shape[0] >= 11


def test_aggregate_signal_handles_empty_emb(tmp_path, monkeypatch):
    from newperformers.io import yfinance as yf_io
    monkeypatch.setattr(yf_io, "fetch", lambda: pd.DataFrame())
    composite = pd.DataFrame({"iso3": ["AAA"] * 10,
                                "year": list(range(2010, 2020)),
                                "composite": np.linspace(-1, 1, 10)})
    out = pc.aggregate_signal_vs_emb(composite)
    assert out == {} or "n_months" in out


def test_overlay_handles_no_tickers(tmp_path, monkeypatch):
    from newperformers.io import yfinance as yf_io
    monkeypatch.setattr(yf_io, "fetch", lambda: pd.DataFrame())
    composite = pd.DataFrame({"iso3": ["AAA"] * 5,
                                "year": list(range(2010, 2015)),
                                "composite": np.linspace(-1, 1, 5)})
    out = pc.credit_signal_overlay(composite)
    assert out == {} or "sharpe" in out
