"""Palette helpers. Backs both matplotlib and plotly themes."""

from __future__ import annotations

from typing import Iterable

from ..utils.config import palette as _palette


def hex(name: str) -> str:
    return _palette()[name]


def cycle() -> list[str]:
    return list(_palette()["cycle"])


def country_color(iso3: str, ordering: Iterable[str]) -> str:
    seq = cycle()
    for i, code in enumerate(ordering):
        if code == iso3:
            return seq[i % len(seq)]
    return _palette()["neutral"]


def diverging_anchors() -> tuple[str, str, str]:
    d = _palette()["diverging"]
    return d["negative"], d["midpoint"], d["positive"]


def sequential_anchors() -> tuple[str, str]:
    s = _palette()["sequential"]
    return s["low"], s["high"]
