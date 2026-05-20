"""YAML config loaders. Read once, cache for the process lifetime."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import yaml

from .paths import CONFIG


def _load(name: str) -> Any:
    with (CONFIG / name).open() as fh:
        return yaml.safe_load(fh)


@dataclass(frozen=True)
class Country:
    iso3: str
    name: str
    groups: tuple[str, ...]
    region: str
    tradable: bool
    etf_ticker: str | None
    etf_inception: str | None
    note: str | None = None


@dataclass(frozen=True)
class Indicator:
    code: str
    source: str
    label: str
    units: str
    direction: str
    role: str = "core"
    sector: str = ""
    transform: str = "identity"  # identity | log1p | log_per_capita


@lru_cache(maxsize=1)
def countries() -> dict[str, Country]:
    raw = _load("countries.yaml")
    out: dict[str, Country] = {}
    for entry in raw.get("focal", []) + raw.get("extensions", []):
        out[entry["iso3"]] = Country(
            iso3=entry["iso3"],
            name=entry["name"],
            groups=tuple(entry["groups"]),
            region=entry["region"],
            tradable=entry["tradable"],
            etf_ticker=entry.get("etf_ticker"),
            etf_inception=entry.get("etf_inception"),
            note=entry.get("note"),
        )
    return out


@lru_cache(maxsize=1)
def kpi_indicators() -> list[Indicator]:
    raw = _load("kpi_indicators.yaml")
    out: list[Indicator] = []
    for sector, items in raw.items():
        for it in items:
            out.append(Indicator(sector=sector, **it))
    return out


@lru_cache(maxsize=1)
def macro_indicators() -> list[Indicator]:
    raw = _load("macro_indicators.yaml")
    out: list[Indicator] = []
    for dimension, items in raw.items():
        for it in items:
            out.append(Indicator(sector=dimension, role="core", **it))
    return out


@lru_cache(maxsize=1)
def outcomes() -> list[Indicator]:
    raw = _load("outcomes.yaml")
    return [
        Indicator(sector="outcome", role="core", direction="higher_is_better", **it)
        for it in raw["outcomes"]
    ]


@lru_cache(maxsize=1)
def policy_events() -> dict[str, list[dict[str, Any]]]:
    return _load("policy_events.yaml") or {}


@lru_cache(maxsize=1)
def palette() -> dict[str, Any]:
    return _load("palette.yaml")


def focal_iso3() -> list[str]:
    """The 13 focal-group countries (everything that isn't an Extension EM)."""
    return [c.iso3 for c in countries().values() if "Extension" not in c.groups]


# Backwards-compatible alias.
brief_iso3 = focal_iso3


def extension_iso3() -> list[str]:
    return [c.iso3 for c in countries().values() if "Extension" in c.groups]


def all_iso3() -> list[str]:
    return list(countries().keys())


def tradable_iso3() -> list[str]:
    return [c.iso3 for c in countries().values() if c.tradable]
