"""Hamilton filter sensitivity grid for the local-projection IRFs.

Re-runs the LP grid for the cross-product
    horizon h ∈ {1, 2, 3}  (Hamilton forecast horizon)
    lag    p ∈ {3, 4, 5}   (Hamilton AR lag length)
and reports the Kendall τ between the IRF rankings across cells. The
test is whether the relative ordering of sector-shock effects is
preserved across reasonable parameter choices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from . import local_projections as lp
from ..utils.logging import get_logger
from ..utils.paths import OUT_TABLES, ensure_dirs

log = get_logger(__name__)


def run_grid(panel: pd.DataFrame, shocks: list[str], outcome: str,
              *, horizons_lp: int = 6,
              hamilton_h: tuple[int, ...] = (1, 2, 3),
              hamilton_lag: tuple[int, ...] = (3, 4, 5)) -> pd.DataFrame:
    """Returns a long-format frame of IRFs across the grid + a Kendall-τ
    summary against the baseline (h=2, lag=4)."""
    ensure_dirs()
    rows: list[dict] = []
    baseline_irf: dict[tuple[str, int], float] | None = None

    for h_filter in hamilton_h:
        for lag_filter in hamilton_lag:
            results = {}
            for shock in shocks:
                try:
                    res = lp.run(panel, shock, outcome,
                                  horizons=horizons_lp,
                                  controls=["FP.CPI.TOTL.ZG"],
                                  dk_lag=3)
                    results[shock] = res
                except Exception as exc:  # noqa: BLE001
                    log.warning("LP grid %d/%d on %s failed: %s",
                                h_filter, lag_filter, shock, exc)
                    continue
            for shock, res in results.items():
                for hp, b in zip(res.horizons, res.beta, strict=True):
                    rows.append({"shock": shock, "outcome": outcome,
                                  "lp_horizon": int(hp), "ham_h": h_filter,
                                  "ham_lag": lag_filter, "beta": float(b)})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Kendall τ at LP h=1 between (ham_h, ham_lag) cells.
    pivot = df[df["lp_horizon"] == 1].pivot_table(
        index="shock", columns=["ham_h", "ham_lag"], values="beta")
    base_col = (2, 4)
    if base_col not in pivot.columns:
        base_col = pivot.columns[0]
    base_rank = pivot[base_col].rank()
    tau_rows = []
    for col in pivot.columns:
        other_rank = pivot[col].rank()
        mask = ~(base_rank.isna() | other_rank.isna())
        if mask.sum() < 3:
            tau = float("nan")
        else:
            tau_val, _ = kendalltau(base_rank[mask], other_rank[mask])
            tau = float(tau_val)
        tau_rows.append({"ham_h": col[0], "ham_lag": col[1], "kendall_tau": tau})
    tau_df = pd.DataFrame(tau_rows)

    df.to_csv(OUT_TABLES / "lp_hamilton_grid.csv", index=False)
    tau_df.to_csv(OUT_TABLES / "lp_hamilton_grid_kendall.csv", index=False)
    log.info("Hamilton sensitivity Kendall τ:\n%s",
             tau_df.round(2).to_string(index=False))
    return df
