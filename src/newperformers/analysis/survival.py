"""Survival analysis on time-to-investment-grade.

We treat the event ``country reaches BBB- for the first time`` as a
duration with right-censoring at 2024 for countries that have never been
upgraded into investment grade by S&P. The covariates are the six sector
scores at the start of each country's panel coverage (i.e. baseline
KPIs). The question is whether higher KPI scores at baseline shorten the
time to IG attainment.

Outputs:
    outputs/tables/survival_events.csv         — per-country event/censor
    outputs/tables/survival_km.csv             — KM survival function
    outputs/tables/survival_cox.csv            — Cox PH coefficients
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..utils.logging import get_logger
from ..utils.paths import OUT_TABLES, ensure_dirs

log = get_logger(__name__)

SECTORS = ["education", "energy", "research_innovation",
           "health", "housing_living", "security_stability"]
IG_THRESHOLD = 13  # BBB- numeric


def _baseline_kpis(composite: pd.DataFrame) -> pd.DataFrame:
    base = (composite.sort_values("year").groupby("iso3").head(1)
            [["iso3"] + SECTORS])
    return base.set_index("iso3")


def event_table(composite: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Build the per-country (T, E) table with sector covariates at baseline."""
    ratings = panel[panel["indicator"] == "SP_RATING"][
        ["iso3", "year", "value"]].sort_values(["iso3", "year"])
    base_year = int(composite["year"].min())
    end_year = int(composite["year"].max())
    rows = []
    base_kpi = _baseline_kpis(composite)
    for iso3, sub in ratings.groupby("iso3"):
        first_ig = sub[sub["value"] >= IG_THRESHOLD]
        if first_ig.empty:
            event_year = end_year
            event = 0
            entry_year = int(sub["year"].min())
        else:
            event_year = int(first_ig["year"].min())
            event = 1
            entry_year = int(sub["year"].min())
        # Duration: years between first rating and IG (or censoring).
        duration = max(event_year - entry_year, 1)
        row = {"iso3": iso3, "duration": duration, "event": event,
               "entry_year": entry_year, "event_year": event_year}
        if iso3 in base_kpi.index:
            for s in SECTORS:
                row[s] = float(base_kpi.loc[iso3, s])
        else:
            for s in SECTORS:
                row[s] = float("nan")
        rows.append(row)
    out = pd.DataFrame(rows).dropna(subset=SECTORS)
    return out


def kaplan_meier(events: pd.DataFrame, split_by: str = "energy") -> pd.DataFrame:
    """KM survival split into high vs low by the median of ``split_by``."""
    from lifelines import KaplanMeierFitter

    median = events[split_by].median()
    rows = []
    for label, mask in (("high", events[split_by] > median),
                         ("low", events[split_by] <= median)):
        sub = events[mask]
        if sub.empty:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(sub["duration"], sub["event"], label=label)
        sf = kmf.survival_function_.reset_index()
        sf.columns = ["duration", "survival"]
        sf["arm"] = label
        sf["split_by"] = split_by
        sf["n"] = len(sub)
        rows.append(sf)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def cox_ph(events: pd.DataFrame) -> pd.DataFrame:
    """Cox proportional-hazards on baseline sector scores."""
    from lifelines import CoxPHFitter

    cph = CoxPHFitter(penalizer=0.05)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            cph.fit(events[["duration", "event"] + SECTORS],
                    duration_col="duration", event_col="event",
                    show_progress=False)
        except Exception as exc:  # noqa: BLE001
            log.warning("Cox PH failed: %s", exc)
            return pd.DataFrame()
    summary = cph.summary[["coef", "exp(coef)", "se(coef)", "p",
                            "exp(coef) lower 95%", "exp(coef) upper 95%"]]
    summary = summary.reset_index().rename(columns={
        "covariate": "sector",
        "exp(coef)": "hazard_ratio",
        "exp(coef) lower 95%": "hr_lower",
        "exp(coef) upper 95%": "hr_upper",
        "p": "p_value",
        "se(coef)": "se",
    })
    return summary


def run_all(composite: pd.DataFrame, panel: pd.DataFrame) -> dict[str, object]:
    ensure_dirs()
    events = event_table(composite, panel)
    events.to_csv(OUT_TABLES / "survival_events.csv", index=False)
    log.info("Survival events: %d rated countries, %d achieved IG, %d censored",
             len(events), int(events["event"].sum()),
             int(len(events) - events["event"].sum()))
    km_energy = kaplan_meier(events, split_by="energy")
    km_health = kaplan_meier(events, split_by="health")
    km = pd.concat([km_energy, km_health], ignore_index=True)
    km.to_csv(OUT_TABLES / "survival_km.csv", index=False)

    cox = cox_ph(events)
    if not cox.empty:
        cox.to_csv(OUT_TABLES / "survival_cox.csv", index=False)
        log.info("Cox PH hazard ratios:\n%s",
                 cox[["sector", "hazard_ratio", "p_value"]].round(2)
                    .to_string(index=False))
    return {"events": events, "km": km, "cox": cox}
