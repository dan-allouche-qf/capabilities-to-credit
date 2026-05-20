"""Canonical long-format schema and helpers."""

from __future__ import annotations

import pandas as pd

COLUMNS: list[str] = ["iso3", "year", "indicator", "value", "source"]
DTYPES: dict[str, str] = {
    "iso3": "string",
    "year": "Int32",
    "indicator": "string",
    "value": "float64",
    "source": "string",
}


def empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=DTYPES[c]) for c in COLUMNS})


def coerce(df: pd.DataFrame, *, source: str) -> pd.DataFrame:
    """Coerce an incoming DataFrame to the canonical long-format schema."""
    missing = [c for c in COLUMNS if c not in df.columns and c != "source"]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    out = df.copy()
    if "source" not in out.columns:
        out["source"] = source
    out = out[COLUMNS]
    out = out.astype(DTYPES)
    out = out.dropna(subset=["iso3", "year", "indicator"])
    out = out.drop_duplicates(subset=["iso3", "year", "indicator"], keep="last")
    return out.reset_index(drop=True)


def stack(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return empty()
    out = pd.concat(frames, ignore_index=True)
    return out.astype(DTYPES)
