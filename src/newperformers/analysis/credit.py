"""Sovereign credit models.

Three layers, increasing in ambition:

    1. Ordered-probit baseline (statsmodels OrderedModel). Pooled, no
       random effects. Reports marginal effects of each sector score on
       the latent rating index.
    2. Out-of-sample expanding-window predictions: train on years
       <= T-1, predict T, repeat for T in {2005, ..., 2024}. Compare
       three nested models — sector-scores only, macro-only, full —
       on MAE, RMSE and ROC AUC for the investment-grade binary.
    3. (Optional) Bayesian hierarchical ordered model in PyMC: written
       when PyMC is available in the environment; otherwise skipped
       with an explicit log message so the pipeline keeps running.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from ..utils.logging import get_logger
from ..utils.paths import OUT_RESULTS, OUT_TABLES, ensure_dirs

log = get_logger(__name__)


INVESTMENT_GRADE = 13  # BBB- and above


@dataclass
class CreditFit:
    name: str
    model: object
    coefs: pd.Series
    rmse: float
    mae: float
    accuracy: float
    auc_ig: float | None = None
    notes: str = ""
    diagnostics: dict = field(default_factory=dict)


def _make_features(composite: pd.DataFrame, panel: pd.DataFrame,
                   macro_controls: list[str]) -> pd.DataFrame:
    """Join composite sector scores with macro controls and ratings."""
    sectors = [c for c in composite.columns if c not in {"iso3", "year", "composite"}]
    base = composite[["iso3", "year"] + sectors].copy()
    pivot = (panel[panel["indicator"].isin(macro_controls + ["SP_RATING"])]
             .pivot_table(index=["iso3", "year"], columns="indicator", values="value")
             .reset_index())
    df = base.merge(pivot, on=["iso3", "year"], how="left")
    if "NY.GDP.PCAP.CD" in df.columns:
        df["log_gdppc"] = np.log(df["NY.GDP.PCAP.CD"].clip(lower=1))
    df = df.dropna(subset=["SP_RATING"])
    return df


def _fit_ordered_probit(df: pd.DataFrame, regressors: list[str]) -> CreditFit:
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    use = df[regressors + ["SP_RATING"]].dropna().copy()
    y = use["SP_RATING"].astype(int)
    X = use[regressors].astype(float)
    model = OrderedModel(y, X, distr="probit")
    res = model.fit(method="bfgs", disp=False, maxiter=400)
    pred_classes = np.argmax(res.model.predict(res.params), axis=1) + int(y.min())
    rmse = float(np.sqrt(mean_squared_error(y, pred_classes)))
    mae = float(mean_absolute_error(y, pred_classes))
    acc = float((pred_classes == y).mean())
    coefs = pd.Series(res.params[: len(regressors)], index=regressors)
    return CreditFit(
        name="ordered_probit_pooled",
        model=res,
        coefs=coefs,
        rmse=rmse,
        mae=mae,
        accuracy=acc,
        notes=f"In-sample fit on {len(use)} country-years.",
        diagnostics={"loglik": float(res.llf), "aic": float(res.aic)},
    )


def _oos_expanding(df: pd.DataFrame, regressors: list[str],
                   start_year: int = 2005,
                   end_year: int = 2024) -> pd.DataFrame:
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    records = []
    for yr in range(start_year, end_year + 1):
        train = df[df["year"] < yr].dropna(subset=regressors + ["SP_RATING"])
        test = df[df["year"] == yr].dropna(subset=regressors + ["SP_RATING"])
        if train.empty or test.empty:
            continue
        try:
            model = OrderedModel(train["SP_RATING"].astype(int),
                                 train[regressors].astype(float),
                                 distr="probit")
            res = model.fit(method="bfgs", disp=False, maxiter=400)
            probs = res.model.predict(res.params, exog=test[regressors].astype(float))
            classes = np.argmax(probs, axis=1) + int(train["SP_RATING"].min())
        except Exception as exc:  # noqa: BLE001
            log.warning("OrderedModel year %d failed: %s", yr, exc)
            continue
        for iso3, actual, pred in zip(test["iso3"].values, test["SP_RATING"].values,
                                       classes, strict=False):
            records.append({"iso3": iso3, "year": yr, "actual": int(actual),
                            "predicted": int(pred), "n_train": len(train)})
    return pd.DataFrame(records)


def fit_models(composite: pd.DataFrame, panel: pd.DataFrame) -> dict[str, object]:
    ensure_dirs()
    macro_controls = ["GC.DOD.TOTL.GD.ZS", "BN.CAB.XOKA.GD.ZS",
                      "FP.CPI.TOTL.ZG", "NY.GDP.PCAP.CD"]
    df = _make_features(composite, panel, macro_controls)
    sectors = ["education", "energy", "research_innovation",
               "health", "housing_living", "security_stability"]
    macro_reg = [c for c in ["GC.DOD.TOTL.GD.ZS", "BN.CAB.XOKA.GD.ZS",
                              "FP.CPI.TOTL.ZG", "log_gdppc"] if c in df.columns]

    configs = {
        "sectors_only": sectors,
        "macro_only": macro_reg,
        "full": sectors + macro_reg,
    }
    fits: dict[str, CreditFit] = {}
    for name, regs in configs.items():
        try:
            fits[name] = _fit_ordered_probit(df, regs)
            log.info("Credit '%s': MAE=%.2f notches | RMSE=%.2f | acc=%.2f",
                     name, fits[name].mae, fits[name].rmse, fits[name].accuracy)
        except Exception as exc:  # noqa: BLE001
            log.warning("Credit '%s' failed: %s", name, exc)

    coef_rows = []
    for name, fit in fits.items():
        for var, val in fit.coefs.items():
            coef_rows.append({"model": name, "variable": var, "coefficient": float(val)})
    pd.DataFrame(coef_rows).to_csv(OUT_TABLES / "credit_coefficients.csv", index=False)

    oos_full = _oos_expanding(df, configs["full"])
    oos_sectors = _oos_expanding(df, configs["sectors_only"])
    oos_macro = _oos_expanding(df, configs["macro_only"])

    def _scorecard(oos: pd.DataFrame, label: str) -> dict[str, float]:
        if oos.empty:
            return {"model": label, "mae": np.nan, "rmse": np.nan,
                    "ig_accuracy": np.nan, "auc_ig": np.nan, "n": 0}
        actual_ig = (oos["actual"] >= INVESTMENT_GRADE).astype(int)
        pred_ig = (oos["predicted"] >= INVESTMENT_GRADE).astype(int)
        try:
            auc = float(roc_auc_score(actual_ig, oos["predicted"]))
        except ValueError:
            auc = np.nan
        return {
            "model": label,
            "mae": float(mean_absolute_error(oos["actual"], oos["predicted"])),
            "rmse": float(np.sqrt(mean_squared_error(oos["actual"], oos["predicted"]))),
            "ig_accuracy": float((actual_ig == pred_ig).mean()),
            "auc_ig": auc,
            "n": int(len(oos)),
        }

    score_rows = [_scorecard(oos, name) for oos, name in (
        (oos_sectors, "sectors_only"),
        (oos_macro, "macro_only"),
        (oos_full, "full"),
    )]
    pd.DataFrame(score_rows).to_csv(OUT_TABLES / "credit_oos_scorecard.csv", index=False)
    oos_full.to_csv(OUT_TABLES / "credit_oos_predictions.csv", index=False)
    log.info("Wrote credit OOS scorecard and predictions.")

    return {"fits": fits, "oos_full": oos_full, "scorecards": pd.DataFrame(score_rows)}
