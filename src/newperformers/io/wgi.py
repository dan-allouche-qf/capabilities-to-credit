"""Worldwide Governance Indicators (Kaufmann–Kraay–Mastruzzi).

The WGI series are now also exposed by the World Bank API under the dotted
codes ``PV.EST``, ``RL.EST``, ``CC.EST``, ``GE.EST``, ``VA.EST`` and
``RQ.EST``. We delegate to the World Bank loader and re-label to the WGI_*
codes used throughout the analysis.
"""

from __future__ import annotations

import pandas as pd

from . import worldbank
from ..etl import schema

SOURCE = "wgi"

CODE_MAP: dict[str, str] = {
    "PV.EST": "WGI_PV",
    "RL.EST": "WGI_RL",
    "CC.EST": "WGI_CC",
    "GE.EST": "WGI_GE",
    "VA.EST": "WGI_VA",
    "RQ.EST": "WGI_RQ",
}


def fetch(*, refresh: bool = False) -> pd.DataFrame:
    raw = worldbank.fetch(list(CODE_MAP.keys()), refresh=refresh)
    raw = raw.copy()
    raw["indicator"] = raw["indicator"].map(CODE_MAP)
    raw["source"] = SOURCE
    return schema.coerce(raw, source=SOURCE)


def sanity_check() -> None:
    df = fetch()
    if df.empty:
        raise RuntimeError("WGI sanity check: empty frame")
    # Singapore's rule-of-law estimate is consistently above 1.5 since the 2000s.
    sgp_rl = df[(df["iso3"] == "SGP") & (df["indicator"] == "WGI_RL")]
    if not (sgp_rl["value"].mean() > 1.0):
        raise RuntimeError("Singapore rule-of-law unexpectedly low")
