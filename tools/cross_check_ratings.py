"""Cross-check `config/ratings.csv` against the major transition dates
documented in `config/ratings_audit_trail.md`.

This is the programmatic complement to `tests/test_ratings_sanity.py`:
for each documented transition, the script asserts that the rating
moves between the *correct* notches in the *correct* year, and writes
the pass/fail status to `outputs/tables/ratings_cross_check.csv`.

Exits with code 0 on all-pass, 1 on any failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from newperformers.io import ratings as r_io  # noqa: E402


# Documented major transitions: (iso3, year_pre, year_post, notch_pre, notch_post).
TRANSITIONS = [
    ("BRA", 2007, 2008, "BB+", "BBB-"),
    ("BRA", 2014, 2015, "BBB-", "BB+"),
    ("IND", 2006, 2007, "BB+", "BBB-"),
    ("RUS", 1997, 1998, "BB-", "CCC"),
    ("RUS", 2013, 2014, "BBB", "BB+"),
    ("RUS", 2017, 2018, "BB+", "BBB-"),
    ("RUS", 2021, 2022, "BBB-", "SD"),
    ("ZAF", 1999, 2000, "BB+", "BBB-"),
    ("ZAF", 2016, 2017, "BBB-", "BB+"),
    ("CHN", 2016, 2017, "AA-", "A+"),
    ("MEX", 2001, 2002, "BB+", "BBB-"),
    ("IDN", 2016, 2017, "BB+", "BBB-"),
    ("PHL", 2012, 2013, "BB+", "BBB-"),
    ("PER", 2007, 2008, "BB+", "BBB-"),
    ("COL", 2020, 2021, "BBB-", "BB+"),
    ("MAR", 2009, 2010, "BB+", "BBB-"),
    ("MAR", 2020, 2021, "BBB-", "BB+"),
]


def _value(df: pd.DataFrame, iso3: str, year: int) -> int | None:
    sub = df[(df["iso3"] == iso3) & (df["year"] == year)]
    if sub.empty:
        return None
    return int(sub["value"].iloc[0])


def main() -> int:
    df = r_io.fetch()
    rows = []
    fails = 0
    for iso3, y0, y1, notch0, notch1 in TRANSITIONS:
        expected_pre = r_io.NOTCH_TO_NUM[notch0]
        expected_post = r_io.NOTCH_TO_NUM[notch1]
        got_pre = _value(df, iso3, y0)
        got_post = _value(df, iso3, y1)
        match_pre = got_pre == expected_pre if got_pre is not None else False
        match_post = got_post == expected_post if got_post is not None else False
        ok = match_pre and match_post
        rows.append({
            "iso3": iso3, "year_pre": y0, "year_post": y1,
            "expected_pre": notch0, "expected_post": notch1,
            "got_pre": got_pre, "got_post": got_post,
            "pass": ok,
        })
        if not ok:
            fails += 1
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "outputs" / "tables" / "ratings_cross_check.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nCross-check: {len(rows) - fails}/{len(rows)} passed")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
