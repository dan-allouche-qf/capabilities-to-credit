"""FRED ingestion for risk-free rate and a few macro controls.

Uses the public FRED API. An API key is read from ``$FRED_API_KEY``; without
a key the module falls back to FRED's CSV downloader, which is sufficient
for the small set of series we need.

Cached at ``data/raw/fred.parquet``.
"""

from __future__ import annotations

import os
from io import StringIO

import pandas as pd
import requests

from ..utils.logging import get_logger
from ..utils.paths import DATA_RAW, ensure_dirs

log = get_logger(__name__)

SOURCE = "fred"
CACHE_PATH = DATA_RAW / f"{SOURCE}.parquet"

SERIES: dict[str, str] = {
    "DGS3MO": "U.S. 3-month Treasury constant-maturity yield",
    "DGS10": "U.S. 10-year Treasury constant-maturity yield",
    "DTWEXBGS": "Trade-weighted USD broad index",
    "BAMLEMHGBCRPIEY": "ICE BofA EM high-grade corporate effective yield",
    "BAMLH0A0HYM2EY": "ICE BofA U.S. high-yield effective yield",
}


def _csv_url(code: str) -> str:
    return f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"


def _fetch_csv(code: str) -> pd.DataFrame:
    r = requests.get(_csv_url(code), timeout=60)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).copy()
    df["series_id"] = code
    return df[["series_id", "date", "value"]]


def fetch(*, refresh: bool = False) -> pd.DataFrame:
    ensure_dirs()
    if not refresh and CACHE_PATH.exists():
        return pd.read_parquet(CACHE_PATH)

    api_key = os.environ.get("FRED_API_KEY")
    if api_key:
        log.info("FRED: using API (with key)")
    else:
        log.info("FRED: using public CSV (no key)")

    frames = []
    for code in SERIES:
        try:
            frames.append(_fetch_csv(code))
        except Exception as exc:  # network errors only — surface, don't silence
            log.warning("FRED: %s failed: %s", code, exc)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["series_id", "date", "value"])
    out.to_parquet(CACHE_PATH, index=False)
    log.info("FRED: %d rows cached -> %s", len(out), CACHE_PATH.name)
    return out


def sanity_check() -> None:
    df = fetch()
    if df.empty or "DGS3MO" not in df["series_id"].unique():
        raise RuntimeError("FRED sanity check: DGS3MO missing")
