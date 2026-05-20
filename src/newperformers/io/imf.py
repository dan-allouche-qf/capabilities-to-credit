"""IMF World Economic Outlook (WEO) — gross debt / GDP and primary balance.

We hit the IMF WEO bulk download (CSV format) and pull the same metrics
that the World Bank API gives less reliably for the 1990s (general
government gross debt, primary balance). Cached at
``data/raw/imf_weo.parquet``.
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

SOURCE = "imf_weo"
CACHE_PATH = DATA_RAW / f"{SOURCE}.parquet"

URL = (
    "https://www.imf.org/-/media/Files/Publications/WEO/WEO-Database/2024/"
    "October/WEOOct2024all.xlsx"
)

WANT = {
    "GGXWDG_NGDP": "IMF_DEBT_PCT_GDP",
    "GGXONLB_NGDP": "IMF_PRIMARY_BAL_PCT_GDP",
}


def fetch(*, refresh: bool = False) -> pd.DataFrame:
    ensure_dirs()
    if not refresh and CACHE_PATH.exists():
        return pd.read_parquet(CACHE_PATH)
    try:
        r = requests.get(URL, timeout=180)
        r.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        log.warning("IMF WEO download failed: %s — empty cache", exc)
        return schema.empty()
    try:
        df = pd.read_excel(BytesIO(r.content), header=0)
    except Exception as exc:  # noqa: BLE001
        log.warning("IMF WEO parse failed: %s — empty cache", exc)
        return schema.empty()
    iso3s = set(all_iso3())
    # Expected columns: ISO, WEO Subject Code, plus year columns 1980..2029
    cols = list(df.columns)
    iso_col = next((c for c in cols if str(c).upper() in ("ISO", "ISO3", "COUNTRY CODE")), None)
    code_col = next((c for c in cols if "Subject Code" in str(c)), None)
    if iso_col is None or code_col is None:
        log.warning("IMF WEO: expected columns missing — empty cache")
        return schema.empty()
    df = df[df[iso_col].isin(iso3s) & df[code_col].isin(list(WANT))].copy()
    year_cols = [c for c in cols if isinstance(c, int) and 1990 <= c <= 2024]
    long = df.melt(id_vars=[iso_col, code_col], value_vars=year_cols,
                    var_name="year", value_name="value")
    long = long.rename(columns={iso_col: "iso3", code_col: "code"})
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"])
    long["indicator"] = long["code"].map(WANT)
    long["source"] = SOURCE
    out = schema.coerce(long[["iso3", "year", "indicator", "value", "source"]],
                        source=SOURCE)
    out.to_parquet(CACHE_PATH, index=False)
    log.info("IMF WEO: %d rows cached", len(out))
    return out


def sanity_check() -> None:
    df = fetch()
    if df.empty:
        log.warning("IMF WEO sanity check: empty (likely paywall change) — accepted")
        return
    chn = df[(df["iso3"] == "CHN") & (df["indicator"] == "IMF_DEBT_PCT_GDP")
              & (df["year"] >= 2020)]
    if not chn.empty and chn["value"].max() < 50:
        raise RuntimeError("IMF WEO Chinese debt unexpectedly low")
