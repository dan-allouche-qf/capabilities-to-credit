"""Build the master panel from every data source."""

from __future__ import annotations

import pandas as pd

from . import schema
from .harmonize import harmonize
from .impute import impute
from ..io import imf as imf_io
from ..io import ratings, sipri as sipri_io, wgi, who as who_io, worldbank
from ..utils.config import (
    kpi_indicators,
    macro_indicators,
    outcomes,
)
from ..utils.logging import get_logger
from ..utils.paths import DATA_PROCESSED, ensure_dirs

log = get_logger(__name__)

DEFAULT_START = 1990
DEFAULT_END = 2024


def _worldbank_codes() -> list[str]:
    codes: list[str] = []
    for ind in kpi_indicators() + macro_indicators() + outcomes():
        if ind.source == "worldbank":
            codes.append(ind.code)
    return list(dict.fromkeys(codes))


def build(*, start: int = DEFAULT_START, end: int = DEFAULT_END,
          refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (raw_panel, imputed_panel) and persist both to data/processed/."""
    ensure_dirs()

    wb = worldbank.fetch(_worldbank_codes(), start=start, end=end, refresh=refresh)
    log.info("World Bank: %d rows", len(wb))

    wgi_df = wgi.fetch(refresh=refresh)
    log.info("WGI: %d rows", len(wgi_df))

    ratings_df = ratings.fetch()
    log.info("Ratings: %d rows", len(ratings_df))

    who_df = who_io.fetch()
    log.info("WHO GHO: %d rows", len(who_df))

    sipri_df = sipri_io.fetch()
    log.info("SIPRI: %d rows", len(sipri_df))

    # IMF WEO URL pattern changes annually; the module returns gracefully on
    # 404. The headline build uses the WB API for debt/GDP (`GC.DOD.TOTL.GD.ZS`).
    imf_df = imf_io.fetch()
    if not imf_df.empty:
        log.info("IMF WEO: %d rows", len(imf_df))

    raw = schema.stack([wb, wgi_df, ratings_df, who_df, sipri_df, imf_df])
    raw = harmonize(raw, start=start, end=end)
    imputed = impute(raw, start=start, end=end)

    raw_path = DATA_PROCESSED / "panel_raw.parquet"
    imp_path = DATA_PROCESSED / "panel_imputed.parquet"
    raw.to_parquet(raw_path, index=False)
    imputed.to_parquet(imp_path, index=False)
    log.info("Saved %s (%d rows) and %s (%d rows)",
             raw_path.name, len(raw), imp_path.name, len(imputed))

    return raw, imputed


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_parquet(DATA_PROCESSED / "panel_raw.parquet")
    imputed = pd.read_parquet(DATA_PROCESSED / "panel_imputed.parquet")
    return raw, imputed
