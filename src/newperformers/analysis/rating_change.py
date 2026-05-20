"""Predict year-over-year sovereign rating change from KPI scores.

The level model in :mod:`credit` predicts the rating notch. A more
operationally useful quantity is the *direction* of the next move:
upgrade / unchanged / downgrade. We collapse the change to a three-class
ordinal (-1, 0, +1) and fit a pooled ordered logit of this change on the
six sector scores at the previous year. Expanding-window OOS for
T ∈ {2005, ..., 2024}.

Benchmark: random-walk (predict 0 always). Reports accuracy, macro F1,
binary "moves vs stays" AUC.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from ..utils.logging import get_logger
from ..utils.paths import OUT_TABLES, ensure_dirs

log = get_logger(__name__)

SECTORS = ["education", "energy", "research_innovation",
           "health", "housing_living", "security_stability"]


def _build_panel(composite: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    rating = (panel[panel["indicator"] == "SP_RATING"][["iso3", "year", "value"]]
              .rename(columns={"value": "rating"}))
    rating = rating.sort_values(["iso3", "year"])
    rating["rating_prev"] = rating.groupby("iso3")["rating"].shift(1)
    rating["delta"] = rating["rating"] - rating["rating_prev"]
    # Collapse to -1 / 0 / +1
    rating["delta_class"] = np.sign(rating["delta"]).astype("Int64")
    sectors_df = composite[["iso3", "year"] + SECTORS]
    sectors_df["year"] = sectors_df["year"] + 1  # lag KPIs by one year
    return (rating.merge(sectors_df, on=["iso3", "year"], how="inner")
                  .dropna(subset=["delta_class"] + SECTORS))


def fit_oos(composite: pd.DataFrame, panel: pd.DataFrame,
             start_year: int = 2005, end_year: int = 2024) -> dict[str, object]:
    """Expanding-window OOS prediction of rating-change direction."""
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    ensure_dirs()
    df = _build_panel(composite, panel)
    if df.empty:
        return {}

    rows = []
    for yr in range(start_year, end_year + 1):
        train = df[df["year"] < yr]
        test = df[df["year"] == yr]
        if train.empty or test.empty:
            continue
        # Order labels: -1 < 0 < +1 → 0, 1, 2 internally.
        train_y = train["delta_class"].astype(int).to_numpy() + 1
        test_y = test["delta_class"].astype(int).to_numpy() + 1
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = OrderedModel(train_y, train[SECTORS].astype(float),
                                      distr="logit")
                res = model.fit(method="bfgs", disp=False, maxiter=400)
                probs = res.model.predict(res.params,
                                            exog=test[SECTORS].astype(float))
            preds = np.argmax(probs, axis=1)
        except Exception as exc:  # noqa: BLE001
            log.warning("Rating-change OOS %d failed: %s", yr, exc)
            continue
        for iso3, actual, pred in zip(test["iso3"].values, test_y, preds,
                                       strict=False):
            rows.append({"iso3": iso3, "year": yr,
                          "actual_class": int(actual) - 1,
                          "predicted_class": int(pred) - 1})
    if not rows:
        return {}
    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_TABLES / "rating_change_predictions.csv", index=False)

    actual = out_df["actual_class"].to_numpy()
    pred = out_df["predicted_class"].to_numpy()
    cm = confusion_matrix(actual, pred, labels=[-1, 0, 1])
    pd.DataFrame(cm, index=["−1 (down)", "0 (flat)", "+1 (up)"],
                  columns=["pred −1", "pred 0", "pred +1"]).to_csv(
        OUT_TABLES / "rating_change_confusion.csv")

    moves_actual = (actual != 0).astype(int)
    moves_pred_class = pred  # use the predicted-class polarity as a score
    moves_pred_score = (pred != 0).astype(int)
    accuracy = float((actual == pred).mean())
    f1 = float(f1_score(actual, pred, average="macro", labels=[-1, 0, 1]))
    rw_accuracy = float((actual == 0).mean())  # random walk = always 0
    try:
        # AUC of pred-class as continuous score vs the "any move" binary.
        auc = float(roc_auc_score(moves_actual, np.abs(moves_pred_class.astype(float))))
    except ValueError:
        auc = float("nan")

    scorecard = {"n_obs": int(len(actual)),
                  "accuracy": accuracy,
                  "rw_baseline_accuracy": rw_accuracy,
                  "macro_f1": f1,
                  "auc_moves": auc}
    pd.DataFrame([scorecard]).to_csv(
        OUT_TABLES / "rating_change_scorecard.csv", index=False)
    log.info("Rating-change OOS: acc=%.3f (rw=%.3f), F1=%.2f, AUC(moves)=%.2f, n=%d",
             accuracy, rw_accuracy, f1, auc, len(actual))
    return {"predictions": out_df, "scorecard": scorecard,
            "confusion": pd.DataFrame(cm,
                                       index=["−1", "0", "+1"],
                                       columns=["pred −1", "pred 0", "pred +1"])}
