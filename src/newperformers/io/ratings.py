"""Sovereign credit ratings.

The hand-compiled CSV at ``config/ratings.csv`` carries the year-end S&P
foreign-currency long-term sovereign rating for the 20-country universe,
1995–2024.

Mapping to the 22-notch numeric scale (S&P main scale):

    AAA  22  | AA+  21  | AA   20  | AA-  19
    A+   18  | A    17  | A-   16
    BBB+ 15  | BBB  14  | BBB- 13   (investment-grade threshold)
    BB+  12  | BB   11  | BB-  10
    B+    9  | B     8  | B-    7
    CCC+  6  | CCC   5  | CCC-  4
    CC    3  | C     2  | SD/D  1
"""

from __future__ import annotations

import pandas as pd

from ..etl import schema
from ..utils.paths import CONFIG

SOURCE = "ratings"
PATH = CONFIG / "ratings.csv"

NOTCH_TO_NUM: dict[str, int] = {
    "AAA": 22, "AA+": 21, "AA": 20, "AA-": 19,
    "A+": 18, "A": 17, "A-": 16,
    "BBB+": 15, "BBB": 14, "BBB-": 13,
    "BB+": 12, "BB": 11, "BB-": 10,
    "B+": 9, "B": 8, "B-": 7,
    "CCC+": 6, "CCC": 5, "CCC-": 4,
    "CC": 3, "C": 2,
    "SD": 1, "D": 1,
    "NR": pd.NA,
}


def numeric(notch: str) -> int | float:
    return NOTCH_TO_NUM.get(notch.strip(), pd.NA)


def fetch() -> pd.DataFrame:
    if not PATH.exists():
        raise FileNotFoundError(
            f"Hand-compiled ratings CSV not found at {PATH}. "
            "See config/ratings.csv for the expected format.")
    df = pd.read_csv(PATH, comment="#")
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"iso3", "year", "rating"}
    if not required.issubset(df.columns):
        raise ValueError(f"ratings.csv must include columns {required}")
    df = df[df["rating"].astype(str).str.strip() != "NR"].copy()
    df["value"] = df["rating"].astype(str).str.strip().map(NOTCH_TO_NUM)
    df = df.dropna(subset=["value"]).copy()
    df["indicator"] = "SP_RATING"
    df["source"] = SOURCE
    return schema.coerce(df[["iso3", "year", "indicator", "value", "source"]], source=SOURCE)


def sanity_check() -> None:
    df = fetch()
    sgp = df[(df["iso3"] == "SGP") & (df["year"].between(2015, 2024))]
    if sgp.empty or sgp["value"].mean() < 21.5:
        raise RuntimeError("Singapore should be AAA in the 2015–2024 window")
