"""Synthetic control (Abadie, Diamond, Hainmueller 2010) — minimal in-house
implementation that does not require ``pysyncon``. The weights minimise the
mean-squared pre-treatment gap between the treated unit's outcome and the
convex combination of the donors' outcomes, subject to non-negative weights
summing to one.

Two cases:

    * Rwanda 2000 (Vision 2020 launch) — donor pool: the other 22 panel
      members that have GDP-per-capita coverage from 1990 onwards.
    * Singapore as a forever-treated counterfactual — donor pool: the
      same minus Singapore.

Inference: in-space placebo permutation — re-run synthetic control on each
donor as if it were the treated unit, compute the gap, and rank the actual
treated gap against the distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..utils.logging import get_logger
from ..utils.paths import OUT_TABLES, ensure_dirs

log = get_logger(__name__)


@dataclass
class SyntheticControlResult:
    treated: str
    treatment_year: int
    donor_weights: pd.Series
    actual: pd.Series
    synthetic: pd.Series
    placebo_gaps: pd.DataFrame  # date × donor placebo gap
    rmse_pre: float
    pvalue_post: float  # rank-based two-sided placebo p-value (post-treatment)
    pvalue_rmse_ratio: float  # Abadie-Diamond-Hainmueller RMSPE-ratio p-value


def _placebo_pvalues(actual: pd.Series, synthetic: pd.Series,
                      placebo_gaps: pd.DataFrame, treatment_year: int,
                      post_window: tuple[int, int] | None = None,
                      ) -> tuple[float, float]:
    """Compute two placebo-based p-values.

    1. Mean-absolute-gap rank: rank of the treated unit's average post-
       treatment |gap| among the placebo distribution; the two-sided p-value
       is rank / (n_placebos + 1).
    2. RMSPE-ratio rank (Abadie-Diamond-Hainmueller): post/pre RMSPE ratio
       compared to placebos. This penalises units whose synthetic control
       fit poorly pre-treatment.
    """
    treated_gap = actual - synthetic
    years = treated_gap.index.astype(int)
    pre_mask = years <= treatment_year
    post_mask = years > treatment_year
    if post_window is not None:
        post_mask = (years > treatment_year) & (years >= post_window[0]) & (years <= post_window[1])

    treated_post_abs = float(np.abs(treated_gap[post_mask]).mean())
    treated_pre_rmse = float(np.sqrt((treated_gap[pre_mask] ** 2).mean()))
    treated_post_rmse = float(np.sqrt((treated_gap[post_mask] ** 2).mean()))
    treated_ratio = treated_post_rmse / max(treated_pre_rmse, 1e-9)

    placebo_post_abs: list[float] = []
    placebo_ratios: list[float] = []
    placebo_idx_int = placebo_gaps.index.astype(int)
    for col in placebo_gaps.columns:
        gap = placebo_gaps[col].dropna()
        if gap.empty:
            continue
        idx = gap.index.astype(int)
        pre = idx <= treatment_year
        post = idx > treatment_year
        if post_window is not None:
            post = (idx > treatment_year) & (idx >= post_window[0]) & (idx <= post_window[1])
        if pre.sum() < 3 or post.sum() < 3:
            continue
        pre_rmse = float(np.sqrt((gap[pre] ** 2).mean()))
        post_rmse = float(np.sqrt((gap[post] ** 2).mean()))
        post_abs = float(np.abs(gap[post]).mean())
        placebo_post_abs.append(post_abs)
        placebo_ratios.append(post_rmse / max(pre_rmse, 1e-9))

    n = len(placebo_post_abs)
    if n == 0:
        return float("nan"), float("nan")

    rank_abs = 1 + sum(1 for v in placebo_post_abs if v >= treated_post_abs)
    pvalue_post = rank_abs / (n + 1)

    rank_ratio = 1 + sum(1 for v in placebo_ratios if v >= treated_ratio)
    pvalue_ratio = rank_ratio / (n + 1)

    return float(pvalue_post), float(pvalue_ratio)


def _solve_weights(treated_pre: np.ndarray, donors_pre: np.ndarray) -> np.ndarray:
    n_donors = donors_pre.shape[1]
    x0 = np.full(n_donors, 1.0 / n_donors)

    def obj(w):
        return float(np.mean((treated_pre - donors_pre @ w) ** 2))

    res = minimize(
        obj, x0, method="SLSQP",
        bounds=[(0.0, 1.0)] * n_donors,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
        options={"maxiter": 400, "ftol": 1e-9},
    )
    return res.x


def fit(panel: pd.DataFrame, *, treated: str, treatment_year: int,
        outcome: str = "NY.GDP.PCAP.PP.KD", donors: list[str] | None = None,
        pre_start: int = 1990) -> SyntheticControlResult | None:
    wide = (panel[panel["indicator"] == outcome]
            .pivot_table(index="year", columns="iso3", values="value")
            .sort_index())
    wide = wide.loc[(wide.index >= pre_start) & (wide.index <= 2024)]
    if treated not in wide.columns:
        log.warning("Synthetic control: %s not in panel for %s", treated, outcome)
        return None
    candidates = donors if donors is not None else list(wide.columns)
    candidates = [c for c in candidates if c != treated]
    sub = wide[[treated] + candidates].dropna(how="any")
    if sub.empty or treatment_year not in sub.index:
        log.warning("Synthetic control: insufficient coverage for %s", treated)
        return None

    pre = sub[sub.index <= treatment_year]
    treated_pre = pre[treated].to_numpy(dtype=float)
    donors_pre = pre[candidates].to_numpy(dtype=float)
    scale = pre[treated].std()
    weights = _solve_weights(treated_pre / scale, donors_pre / scale)
    weights = pd.Series(weights, index=candidates)
    synthetic = sub[candidates].dot(weights)
    rmse_pre = float(np.sqrt(((sub[treated].loc[pre.index]
                                - synthetic.loc[pre.index]) ** 2).mean()))

    placebos: dict[str, pd.Series] = {}
    for donor in candidates:
        other_donors = [c for c in candidates if c != donor] + [treated]
        treated_d = sub[donor].to_numpy(dtype=float)
        pool = sub[other_donors].to_numpy(dtype=float)
        treated_pre_d = treated_d[: len(treated_pre)]
        pool_pre = pool[: len(treated_pre), :]
        try:
            w = _solve_weights(treated_pre_d / scale, pool_pre / scale)
            synth_d = pool @ w
            placebos[donor] = pd.Series(treated_d - synth_d, index=sub.index)
        except Exception:  # noqa: BLE001
            continue

    placebos_df = pd.DataFrame(placebos)

    p_post, p_ratio = _placebo_pvalues(sub[treated], synthetic,
                                         placebos_df, treatment_year)

    res = SyntheticControlResult(
        treated=treated,
        treatment_year=treatment_year,
        donor_weights=weights[weights > 1e-4].sort_values(ascending=False),
        actual=sub[treated],
        synthetic=synthetic,
        placebo_gaps=placebos_df,
        rmse_pre=rmse_pre,
        pvalue_post=p_post,
        pvalue_rmse_ratio=p_ratio,
    )
    return res


def run_default_episodes(panel: pd.DataFrame) -> dict[str, SyntheticControlResult]:
    ensure_dirs()
    episodes = {
        "rwanda_2000": ("RWA", 2000),
        "singapore_1990": ("SGP", 1990),
    }
    results: dict[str, SyntheticControlResult] = {}
    for tag, (iso3, year) in episodes.items():
        res = fit(panel, treated=iso3, treatment_year=year)
        if res is None:
            continue
        results[tag] = res
        log.info("Synthetic control [%s]: pre-RMSE = %.3f, p(|gap|) = %.3f, "
                 "p(RMSPE ratio) = %.3f, top donors = %s",
                 tag, res.rmse_pre, res.pvalue_post, res.pvalue_rmse_ratio,
                 list(res.donor_weights.head(4).index))

        wide = pd.DataFrame({"year": res.actual.index,
                             "actual": res.actual.values,
                             "synthetic": res.synthetic.reindex(res.actual.index).values})
        wide.to_csv(OUT_TABLES / f"synthcontrol_{tag}.csv", index=False)
        res.donor_weights.to_csv(OUT_TABLES / f"synthcontrol_{tag}_weights.csv")
        res.placebo_gaps.to_csv(OUT_TABLES / f"synthcontrol_{tag}_placebos.csv")
        pd.DataFrame([{
            "treated": res.treated,
            "treatment_year": res.treatment_year,
            "rmse_pre": res.rmse_pre,
            "pvalue_post": res.pvalue_post,
            "pvalue_rmse_ratio": res.pvalue_rmse_ratio,
            "n_placebos": int(res.placebo_gaps.shape[1]),
        }]).to_csv(OUT_TABLES / f"synthcontrol_{tag}_pvalue.csv", index=False)
    return results
