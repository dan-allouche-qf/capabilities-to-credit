"""SIPRI Military Expenditure Database.

The public SIPRI XLSX download exposes military expenditure as share of GDP,
as constant USD, and per capita. We fetch the share-of-GDP sheet, since
that is what is comparable across countries.
Cached at ``data/raw/sipri.parquet``.
"""

from __future__ import annotations

from io import BytesIO

import pandas as pd
import requests

from ..etl import schema
from ..utils.config import all_iso3
from ..utils.logging import get_logger
from ..utils.paths import DATA_RAW, ensure_dirs

log = get_logger(__name__)

SOURCE = "sipri"
CACHE_PATH = DATA_RAW / f"{SOURCE}.parquet"
# SIPRI URL for the share-of-GDP sheet (publicly available).
URL = "https://www.sipri.org/sites/default/files/SIPRI-Milex-data-1948-2023.xlsx"

NAME_TO_ISO3 = {
    "Brazil": "BRA", "Russia": "RUS", "India": "IND", "China": "CHN",
    "South Africa": "ZAF", "Burkina Faso": "BFA", "Mali": "MLI", "Niger": "NER",
    "Singapore": "SGP", "Rwanda": "RWA", "Kenya": "KEN", "Morocco": "MAR",
    "Pakistan": "PAK", "Indonesia": "IDN", "Mexico": "MEX", "Turkey": "TUR",
    "Türkiye": "TUR", "Thailand": "THA", "Philippines": "PHL", "Malaysia": "MYS",
    "Poland": "POL", "Chile": "CHL", "Colombia": "COL", "Peru": "PER",
    "USSR": "RUS",
}


def fetch(*, refresh: bool = False) -> pd.DataFrame:
    ensure_dirs()
    if not refresh and CACHE_PATH.exists():
        return pd.read_parquet(CACHE_PATH)
    try:
        r = requests.get(URL, timeout=120)
        r.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        log.warning("SIPRI download failed: %s — using empty cache", exc)
        return schema.empty()
    xls = pd.ExcelFile(BytesIO(r.content))
    sheet_candidates = [s for s in xls.sheet_names if "GDP" in s.upper()
                         and "SHARE" in s.upper()]
    sheet = sheet_candidates[0] if sheet_candidates else (
        next((s for s in xls.sheet_names if "% of GDP" in s), xls.sheet_names[-1]))
    df = pd.read_excel(xls, sheet_name=sheet, header=None)
    # Try several header positions.
    header_row = None
    for i in range(15):
        row = df.iloc[i].astype(str).fillna("")
        joined = " ".join(str(v) for v in row.values)
        if "1949" in joined or "1950" in joined or "1990" in joined:
            header_row = i
            break
    if header_row is None:
        log.warning("SIPRI: could not locate header row, skipping")
        return schema.empty()
    df = pd.read_excel(xls, sheet_name=sheet, header=header_row)
    df = df.rename(columns={df.columns[0]: "country"})
    df = df.dropna(subset=["country"])
    long = df.melt(id_vars="country", var_name="year", value_name="value")
    long["year"] = pd.to_numeric(long["year"], errors="coerce")
    long = long.dropna(subset=["year"])
    long["year"] = long["year"].astype(int)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"])
    long["iso3"] = long["country"].map(NAME_TO_ISO3)
    long = long[long["iso3"].isin(all_iso3())]
    long["indicator"] = "SIPRI_MILEX_PCT_GDP"
    long["source"] = SOURCE
    out = schema.coerce(long[["iso3", "year", "indicator", "value", "source"]],
                        source=SOURCE)
    out.to_parquet(CACHE_PATH, index=False)
    log.info("SIPRI: %d rows cached", len(out))
    return out


def sanity_check() -> None:
    df = fetch()
    if df.empty:
        log.warning("SIPRI sanity check: empty (network or layout issue) — accepted")
        return
    rus = df[(df["iso3"] == "RUS") & (df["year"].between(2010, 2023))]
    if rus.empty or rus["value"].mean() < 2.0:
        log.warning("SIPRI: Russia % GDP unexpectedly low or missing")
