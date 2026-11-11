"""Held-out permutation importance.

Importance is measured on the untouched test split with the fitted
preprocessing + model pipeline. Nothing here is ever hard-coded: without a
fitted pipeline and evaluation data the function refuses to produce values.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.utils.validation import check_is_fitted

from core.modeling import METRIC_LABELS, scoring_name

UNAVAILABLE = "Feature importance unavailable — a valid fitted model and evaluation dataset are required."
EXPLANATION = (
    "Permutation importance is the drop in the model's score on the held-out test set when one feature's "
    "values are randomly shuffled, averaged over several shuffles. It is expressed in units of the scoring "
    "metric — it is not a percentage contribution and the values do not sum to 100%. A value near zero or "
    "negative means shuffling the feature did not hurt (or by chance slightly helped) the score. Strongly "
    "correlated features can share or mask each other's importance. It describes what this model relies on, "
    "not what causes the outcome."
)


class ImportanceUnavailable(ValueError):
    def __init__(self, reason: str = ""):
        super().__init__(UNAVAILABLE + (f" ({reason})" if reason else ""))


def held_out_permutation_importance(
    pipeline: Any, X_test: pd.DataFrame | None, y_test: Any, task: str, metric: str,
    n_classes: int = 2, n_repeats: int = 10, seed: int = 42, model_name: str = "",
) -> dict[str, Any]:
    if pipeline is None or X_test is None or y_test is None:
        raise ImportanceUnavailable()
    if len(X_test) < 2:
        raise ImportanceUnavailable("the evaluation set has fewer than 2 rows")
    try:
        check_is_fitted(pipeline[-1])
    except Exception as exc:
        raise ImportanceUnavailable("the model is not fitted") from exc
    scoring = scoring_name(metric, task, n_classes)
    try:
        res = permutation_importance(pipeline, X_test, y_test, scoring=scoring, n_repeats=int(n_repeats),
                                     random_state=seed, n_jobs=1)
    except Exception as exc:
        raise ImportanceUnavailable(f"{type(exc).__name__}: {exc}") from exc
    if not np.all(np.isfinite(res.importances_mean)):
        raise ImportanceUnavailable(f"{METRIC_LABELS[metric]} could not be scored on the evaluation set")
    order = np.argsort(-res.importances_mean)
    features = [{"feature": str(X_test.columns[i]), "mean": float(res.importances_mean[i]),
                 "std": float(res.importances_std[i])} for i in order]
    notes = []
    if len(X_test) < 50:
        notes.append(f"The test set has only {len(X_test)} rows, so these estimates are unstable.")
    noisy = [f["feature"] for f in features if f["std"] > abs(f["mean"]) and abs(f["mean"]) > 0]
    if noisy:
        notes.append(f"For {len(noisy)} feature(s) the standard deviation exceeds the mean importance, so their "
                     "ranking is not reliable: " + ", ".join(noisy[:6]) + ("…" if len(noisy) > 6 else "") + ".")
    if any(f["mean"] < 0 for f in features):
        notes.append("Negative values are shown as computed: shuffling those features did not reduce the score.")
    return {
        "model": model_name, "method": "Permutation importance (scikit-learn)",
        "evaluated_on": f"held-out test set ({len(X_test):,} rows, never used for fitting)",
        "n_test": int(len(X_test)), "metric": metric, "metric_label": METRIC_LABELS[metric], "scoring": scoring,
        "n_repeats": int(n_repeats), "seed": seed, "features": features, "notes": notes, "explanation": EXPLANATION,
    }
