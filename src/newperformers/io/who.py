"""WHO Global Health Observatory (OData) ingestion.

We pull a small set of UHC- and maternal-mortality-related indicators from
the public WHO GHO OData feed (no API key needed):
    UHC_INDEX_REPORTED  — Universal Health Coverage service-coverage index
    MDG_0000000026      — Maternal mortality ratio (per 100 000 live births)
    NCD_BMI_30A         — Prevalence of obesity among adults
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

SOURCE = "who_gho"
CACHE_PATH = DATA_RAW / f"{SOURCE}.parquet"

INDICATORS: dict[str, str] = {
    "UHC_INDEX_REPORTED": "WHO_UHC_INDEX",
    "MDG_0000000026":     "WHO_MAT_MORT",
    "NCD_BMI_30A":        "WHO_OBESITY",
}

BASE = "https://ghoapi.azureedge.net/api/"


def _fetch_one(code: str) -> pd.DataFrame:
    r = requests.get(BASE + code, timeout=60)
    r.raise_for_status()
    rows = r.json().get("value", [])
    out = []
    for row in rows:
        out.append({
            "iso3": row.get("SpatialDim"),
            "year": int(row["TimeDim"]) if row.get("TimeDim") else None,
            "value": row.get("NumericValue"),
            "dim1": row.get("Dim1"),
        })
    df = pd.DataFrame(out)
    df = df.dropna(subset=["iso3", "year", "value"])
    df = df[df["dim1"].isna() | (df["dim1"] == "SEX_BTSX")]
    df = df.drop(columns=["dim1"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["value"])


def fetch(*, refresh: bool = False) -> pd.DataFrame:
    ensure_dirs()
    if not refresh and CACHE_PATH.exists():
        return pd.read_parquet(CACHE_PATH)
    iso3s = set(all_iso3())
    frames: list[pd.DataFrame] = []
    for code, label in INDICATORS.items():
        try:
            df = _fetch_one(code)
        except Exception as exc:  # noqa: BLE001
            log.warning("WHO %s failed: %s", code, exc)
            continue
        df = df[df["iso3"].isin(iso3s)].copy()
        df["indicator"] = label
        df["source"] = SOURCE
        frames.append(df[["iso3", "year", "indicator", "value", "source"]])
        log.info("WHO %s: %d rows", label, len(df))
    if not frames:
        return schema.empty()
    out = schema.coerce(pd.concat(frames, ignore_index=True), source=SOURCE)
    out.to_parquet(CACHE_PATH, index=False)
    return out


def sanity_check() -> None:
    df = fetch()
    if df.empty:
        raise RuntimeError("WHO GHO sanity check: empty frame")
    sgp = df[(df["iso3"] == "SGP") & (df["indicator"] == "WHO_UHC_INDEX")]
    if sgp.empty or sgp["value"].max() < 60:
        raise RuntimeError("Singapore UHC index unexpectedly missing or low")
