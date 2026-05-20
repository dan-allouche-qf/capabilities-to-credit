"""World Bank Open Data fetcher.

Uses the public REST API directly (no auth, no wbdata dependency on the hot
path) and caches the resulting long-format frame to ``data/raw/worldbank.parquet``.
A single call pulls a batch of indicators for the full country universe over
the configured year range.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
import requests
from tqdm import tqdm

from ..etl import schema
from ..utils.config import all_iso3
from ..utils.logging import get_logger
from ..utils.paths import DATA_RAW, ensure_dirs

log = get_logger(__name__)

API = "https://api.worldbank.org/v2"
SOURCE = "worldbank"
DEFAULT_START = 1990
DEFAULT_END = 2024
CACHE_PATH = DATA_RAW / f"{SOURCE}.parquet"
TIMEOUT = 60


def _fetch_one(indicator: str, iso3s: list[str], start: int, end: int) -> pd.DataFrame:
    """Fetch one indicator for a batch of countries in one or several pages."""
    rows: list[dict] = []
    country_param = ";".join(iso3s)
    page = 1
    while True:
        params = {
            "date": f"{start}:{end}",
            "format": "json",
            "per_page": 20000,
            "page": page,
        }
        url = f"{API}/country/{country_param}/indicator/{indicator}"
        resp = requests.get(url, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
            break
        meta, records = payload[0], payload[1]
        for r in records:
            iso3 = r.get("countryiso3code") or r.get("country", {}).get("id")
            value = r.get("value")
            year = int(r["date"]) if r.get("date") else None
            if iso3 and year is not None and value is not None:
                rows.append({
                    "iso3": iso3,
                    "year": year,
                    "indicator": indicator,
                    "value": float(value),
                    "source": SOURCE,
                })
        if meta is None or page >= int(meta.get("pages", 1) or 1):
            break
        page += 1
    return pd.DataFrame(rows, columns=schema.COLUMNS)


def fetch(
    indicators: Iterable[str],
    *,
    iso3s: list[str] | None = None,
    start: int = DEFAULT_START,
    end: int = DEFAULT_END,
    use_cache: bool = True,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch the requested indicators for the configured country panel.

    The cache is the *union* of everything ever fetched; a refresh forces a
    full re-pull of the requested indicators only.
    """
    ensure_dirs()
    iso3s = iso3s or all_iso3()
    indicators = list(dict.fromkeys(indicators))  # dedupe, preserve order

    cached = pd.DataFrame()
    if use_cache and CACHE_PATH.exists():
        cached = pd.read_parquet(CACHE_PATH)

    if refresh or cached.empty:
        missing = indicators
    else:
        have = set(cached["indicator"].unique())
        missing = [i for i in indicators if i not in have]

    if missing:
        log.info("World Bank: fetching %d indicator(s) for %d countries (%d-%d)",
                 len(missing), len(iso3s), start, end)
        new_frames: list[pd.DataFrame] = []
        for ind in tqdm(missing, desc="WB indicators"):
            df = _fetch_one(ind, iso3s, start, end)
            new_frames.append(df)
        new_part = schema.stack(new_frames) if new_frames else schema.empty()
        new_part = schema.coerce(new_part, source=SOURCE)
        if not cached.empty:
            cached = cached[~cached["indicator"].isin(missing)]
        cached = pd.concat([cached, new_part], ignore_index=True)
        cached = schema.coerce(cached, source=SOURCE)
        cached.to_parquet(CACHE_PATH, index=False)
    else:
        log.info("World Bank: cache hit for all %d indicator(s)", len(indicators))

    out = cached[
        cached["indicator"].isin(indicators) & cached["iso3"].isin(iso3s)
    ].copy()
    return out.reset_index(drop=True)


def sanity_check() -> None:
    """Run a quick range check on a small reference indicator."""
    df = fetch(["NY.GDP.MKTP.CD"], iso3s=["IND"], start=2023, end=2024)
    if df.empty:
        raise RuntimeError("World Bank sanity check: empty frame")
    val = df.loc[df["year"] == 2023, "value"].max()
    log.info("India 2023 GDP = $%.2f T (expected ~3.5 T)", val / 1e12)
    if not (2.5e12 < val < 5.0e12):
        raise RuntimeError(f"India 2023 GDP unexpected: {val:.3e}")
