"""Smoke tests on the canonical data shape and core invariants."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from newperformers.etl import schema
from newperformers.io import ratings
from newperformers.utils.config import all_iso3, countries, kpi_indicators


def test_country_universe_has_focal_thirteen():
    iso3s = set(all_iso3())
    focal = {"BRA", "RUS", "IND", "CHN", "ZAF", "BFA", "MLI", "NER",
             "SGP", "RWA", "KEN", "MAR", "PAK"}
    assert focal.issubset(iso3s), f"missing focal countries: {focal - iso3s}"


def test_extension_countries_are_tradable():
    cs = countries()
    extensions = [c for c in cs.values() if "Extension" in c.groups]
    assert len(extensions) >= 10
    assert all(c.tradable for c in extensions)


def test_kpi_indicators_have_valid_directions():
    for ind in kpi_indicators():
        assert ind.direction in ("higher_is_better", "lower_is_better")
        assert ind.role in ("core", "supplementary")


def test_ratings_csv_loads_and_singapore_is_aaa_recent():
    df = ratings.fetch()
    sgp = df[(df["iso3"] == "SGP") & (df["year"].between(2015, 2024))]
    assert not sgp.empty
    assert sgp["value"].mean() >= 21.5


def test_schema_round_trip():
    df = pd.DataFrame({
        "iso3": ["SGP", "CHN"], "year": [2020, 2020],
        "indicator": ["X", "X"], "value": [1.0, 2.0], "source": ["x", "x"],
    })
    out = schema.coerce(df, source="x")
    assert list(out.columns) == schema.COLUMNS
    assert out["year"].dtype.name in ("Int32", "int32")


def test_schema_drops_duplicates_keep_last():
    df = pd.DataFrame({
        "iso3": ["SGP", "SGP"], "year": [2020, 2020],
        "indicator": ["X", "X"], "value": [1.0, 99.0], "source": ["x", "x"],
    })
    out = schema.coerce(df, source="x")
    assert len(out) == 1
    assert out["value"].iloc[0] == 99.0
