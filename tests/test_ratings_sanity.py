"""Sanity-check the hand-compiled S&P ratings against widely-documented
rating-action dates from the S&P sovereign archive. The thresholds below
are the public, well-attested major transitions; if any of them moves we
want to know.
"""

from __future__ import annotations

import pytest

from newperformers.io import ratings

NUM = ratings.NOTCH_TO_NUM


def _at(df, iso3, year):
    sub = df[(df["iso3"] == iso3) & (df["year"] == year)]
    if sub.empty:
        return None
    return int(sub["value"].iloc[0])


@pytest.fixture(scope="module")
def panel():
    return ratings.fetch()


def test_singapore_aaa_continuous(panel):
    for y in range(1995, 2025):
        v = _at(panel, "SGP", y)
        assert v is not None, f"Singapore {y} missing"
        assert v == NUM["AAA"], f"Singapore {y} not AAA: numeric={v}"


def test_brazil_ig_2008_2014(panel):
    # S&P upgraded BRA to BBB- in April 2008, kept IG through 2014.
    for y in (2008, 2010, 2013, 2014):
        v = _at(panel, "BRA", y)
        assert v is not None and v >= NUM["BBB-"], (
            f"Brazil {y} should be investment-grade, got {v}")
    # And lost IG in September 2015.
    v2015 = _at(panel, "BRA", 2015)
    assert v2015 is not None and v2015 < NUM["BBB-"], (
        f"Brazil 2015 should be sub-IG, got {v2015}")


def test_india_ig_since_2007(panel):
    # S&P upgraded India to BBB- in January 2007 and has held it since.
    for y in (2007, 2010, 2015, 2020, 2024):
        v = _at(panel, "IND", y)
        assert v is not None and v >= NUM["BBB-"], (
            f"India {y} should be at least BBB-, got {v}")


def test_russia_arc(panel):
    # IG to BB+ in April 2014 (Crimea), back to IG in February 2018,
    # SD in March 2022.
    for y in (2008, 2010, 2013):
        v = _at(panel, "RUS", y)
        assert v is not None and v >= NUM["BBB-"], f"Russia {y} should be IG"
    for y in (2014, 2015, 2016, 2017):
        v = _at(panel, "RUS", y)
        assert v is not None and v < NUM["BBB-"] and v >= NUM["BB+"], (
            f"Russia {y} should be BB+ region")
    v2022 = _at(panel, "RUS", 2022)
    assert v2022 is not None and v2022 <= NUM["CCC-"], (
        f"Russia 2022 should be SD/CCC region, got {v2022}")


def test_south_africa_arc(panel):
    # IG from 2000 to 2016, lost IG in April 2017.
    for y in (2005, 2010, 2014, 2016):
        v = _at(panel, "ZAF", y)
        assert v is not None and v >= NUM["BBB-"], (
            f"ZAF {y} should be IG, got {v}")
    for y in (2017, 2020, 2024):
        v = _at(panel, "ZAF", y)
        assert v is not None and v < NUM["BBB-"], (
            f"ZAF {y} should be sub-IG, got {v}")


def test_china_post2017_downgrade(panel):
    # S&P AA- through 2016, A+ from September 2017 onwards.
    v2016 = _at(panel, "CHN", 2016)
    assert v2016 == NUM["AA-"], f"China 2016 should be AA-, got {v2016}"
    v2018 = _at(panel, "CHN", 2018)
    assert v2018 == NUM["A+"], f"China 2018 should be A+, got {v2018}"


def test_morocco_ig_window(panel):
    # IG attained March 2010, lost April 2021.
    for y in (2010, 2015, 2020):
        v = _at(panel, "MAR", y)
        assert v is not None and v >= NUM["BBB-"], (
            f"Morocco {y} should be IG, got {v}")
    for y in (2021, 2024):
        v = _at(panel, "MAR", y)
        assert v is not None and v < NUM["BBB-"], (
            f"Morocco {y} should be sub-IG, got {v}")


def test_aes_sahel_unrated(panel):
    # Mali and Niger are NR. They should not appear in the numeric panel at all.
    iso3s = panel["iso3"].unique()
    assert "MLI" not in iso3s, "Mali appears unexpectedly in the rated panel"
    assert "NER" not in iso3s, "Niger appears unexpectedly in the rated panel"


def test_mexico_ig_since_2002(panel):
    for y in (2002, 2005, 2010, 2015, 2020, 2024):
        v = _at(panel, "MEX", y)
        assert v is not None and v >= NUM["BBB-"], (
            f"Mexico {y} should be IG, got {v}")


def test_indonesia_ig_since_2017(panel):
    for y in (2017, 2020, 2024):
        v = _at(panel, "IDN", y)
        assert v is not None and v >= NUM["BBB-"], (
            f"Indonesia {y} should be IG, got {v}")


def test_philippines_ig_since_2013(panel):
    for y in (2013, 2015, 2020, 2024):
        v = _at(panel, "PHL", y)
        assert v is not None and v >= NUM["BBB-"], (
            f"Philippines {y} should be IG, got {v}")


def test_peru_ig_since_2008(panel):
    for y in (2008, 2010, 2015, 2020, 2024):
        v = _at(panel, "PER", y)
        assert v is not None and v >= NUM["BBB-"], (
            f"Peru {y} should be IG, got {v}")


def test_colombia_lost_ig_2021(panel):
    for y in (2017, 2018, 2019, 2020):
        v = _at(panel, "COL", y)
        assert v is not None and v >= NUM["BBB-"], (
            f"Colombia {y} should be IG, got {v}")
    for y in (2021, 2022, 2024):
        v = _at(panel, "COL", y)
        assert v is not None and v < NUM["BBB-"], (
            f"Colombia {y} should be sub-IG, got {v}")


def test_pakistan_distressed_recent(panel):
    # Pakistan dropped to CCC+ in 2022 after multiple downgrades.
    for y in (2022, 2023, 2024):
        v = _at(panel, "PAK", y)
        assert v is not None and v <= NUM["B-"], (
            f"Pakistan {y} should be B-/CCC, got {v}")
