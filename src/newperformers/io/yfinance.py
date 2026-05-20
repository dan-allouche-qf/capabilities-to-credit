"""Country-ETF and benchmark price ingestion via the yfinance package.

The Part III factor portfolio uses daily total-return ETF prices (adjusted
close, dividend-reinvested) for the 17 tradable countries plus benchmarks
(MSCI EM via EEM, MSCI World via URTH, U.S. Dollar Index via UUP, and the
WTI front-month future via USO as a free proxy).

The frame is stored as ``data/raw/yfinance.parquet`` keyed on
{ticker, date, field=adj_close}.
"""

from __future__ import annotations

import pandas as pd

from ..utils.config import countries
from ..utils.logging import get_logger
from ..utils.paths import DATA_RAW, ensure_dirs

log = get_logger(__name__)

SOURCE = "yfinance"
CACHE_PATH = DATA_RAW / f"{SOURCE}.parquet"

BENCHMARKS: dict[str, str] = {
    "EEM":  "MSCI Emerging Markets",
    "URTH": "MSCI World",
    "UUP":  "U.S. Dollar Index",
    "USO":  "WTI crude oil",
    "MOVE": "MOVE bond-vol index",
    # EM sovereign credit instruments — the Part-III credit extension.
    "EMB":  "iShares JPM USD EM Sovereign Bond",
    "EMLC": "VanEck JPM EM Local Currency Bond",
    "PCY":  "Invesco EM Sovereign Debt",
}


def _tickers() -> dict[str, str]:
    """All tickers we need: tradable country ETFs + benchmarks."""
    cs = countries()
    out: dict[str, str] = {
        c.etf_ticker: c.iso3 for c in cs.values() if c.tradable and c.etf_ticker
    }
    for tkr in BENCHMARKS:
        out[tkr] = tkr
    return out


def fetch(*, start: str = "1995-01-01", refresh: bool = False) -> pd.DataFrame:
    """Return tidy long frame: (ticker, date, adj_close)."""
    ensure_dirs()
    if not refresh and CACHE_PATH.exists():
        return pd.read_parquet(CACHE_PATH)

    import yfinance as yf

    tickers = list(_tickers().keys())
    log.info("yfinance: downloading %d tickers from %s", len(tickers), start)
    raw = yf.download(
        tickers=tickers,
        start=start,
        progress=False,
        auto_adjust=True,
        actions=False,
        group_by="ticker",
        threads=True,
    )
    rows: list[pd.DataFrame] = []
    if isinstance(raw.columns, pd.MultiIndex):
        for tkr in tickers:
            if tkr not in raw.columns.get_level_values(0):
                continue
            df = raw[tkr][["Close"]].dropna().rename(columns={"Close": "adj_close"})
            df["ticker"] = tkr
            df["date"] = df.index
            rows.append(df.reset_index(drop=True)[["ticker", "date", "adj_close"]])
    else:
        df = raw[["Close"]].dropna().rename(columns={"Close": "adj_close"})
        df["ticker"] = tickers[0]
        df["date"] = df.index
        rows.append(df.reset_index(drop=True)[["ticker", "date", "adj_close"]])

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["ticker", "date", "adj_close"])
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    out.to_parquet(CACHE_PATH, index=False)
    log.info("yfinance: %d rows cached -> %s", len(out), CACHE_PATH.name)
    return out


def sanity_check() -> None:
    df = fetch()
    if df.empty:
        raise RuntimeError("yfinance sanity check: empty frame")
    for tkr in ("EWZ", "EEM"):
        if tkr not in df["ticker"].unique():
            raise RuntimeError(f"{tkr} missing from yfinance cache")
