"""Model diagnostics computed on the held-out test set, plus Plotly figure builders.

Every structure returned here is JSON-serialisable and labelled with the data
it was computed on. Figure builders return ``None`` when the needed numbers do
not exist; nothing is approximated or invented.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve

from core.eda import PALETTE, SAGE, SEQUENTIAL, style

EVALUATED_ON = "held-out test set"
MAX_POINTS = 2000
MIN_SUBGROUP = 10


def _downsample_curve(x: np.ndarray, y: np.ndarray, limit: int = 400) -> tuple[list[float], list[float]]:
    if len(x) > limit:
        idx = np.unique(np.linspace(0, len(x) - 1, limit).astype(int))
        x, y = x[idx], y[idx]
    return [float(v) for v in x], [float(v) for v in y]


def classification_diagnostics(pipe: Any, X_test: pd.DataFrame, y_test: Any, classes: list[str]) -> dict[str, Any]:
    y = np.asarray(y_test)
    pred = np.asarray(pipe.predict(X_test))
    labels = list(range(len(classes)))
    cm = confusion_matrix(y, pred, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)
    out: dict[str, Any] = {"evaluated_on": EVALUATED_ON, "n_test": int(len(y)), "classes": list(classes),
                           "confusion_matrix_normalized": norm.round(6).tolist(),
                           "test_class_counts": {classes[i]: int(row_sums[i, 0]) for i in labels}}
    wrong = pred != y
    pairs = {}
    for a, p in zip(y[wrong], pred[wrong]):
        pairs[(int(a), int(p))] = pairs.get((int(a), int(p)), 0) + 1
    top = sorted(pairs.items(), key=lambda kv: -kv[1])[:5]
    out["errors"] = {"n_misclassified": int(wrong.sum()), "error_rate": float(wrong.mean()) if len(y) else None,
                     "top_confusions": [{"actual": classes[a], "predicted": classes[p], "count": c} for (a, p), c in top]}
    proba = None
    if hasattr(pipe, "predict_proba"):
        try:
            proba = np.asarray(pipe.predict_proba(X_test))
        except Exception:  # noqa: BLE001 - some estimators expose but cannot compute probabilities
            proba = None
    if proba is None or proba.shape[1] != len(classes):
        out["probability_note"] = "Probability-based diagnostics are unavailable for this model."
        return out
    if len(classes) == 2:
        pos = proba[:, 1]
        yb = (y == 1).astype(int)
        if 0 < yb.sum() < len(yb):
            prec, rec, _ = precision_recall_curve(yb, pos)
            r, p = _downsample_curve(rec, prec)
            out["pr_curve"] = {"recall": r, "precision": p, "average_precision": float(average_precision_score(yb, pos)),
                               "positive_class": classes[1], "baseline": float(yb.mean())}
            n_bins = int(max(2, min(10, len(yb) // 20)))
            frac, mean_pred = calibration_curve(yb, pos, n_bins=n_bins, strategy="quantile")
            out["calibration"] = {"mean_predicted": [float(v) for v in mean_pred], "fraction_positive": [float(v) for v in frac],
                                  "n_bins": n_bins, "strategy": "quantile bins of predicted probability",
                                  "brier": float(np.mean((pos - yb) ** 2))}
        else:
            out["pr_note"] = "PR curve and calibration are unavailable: the test set contains a single class."
        edges = np.linspace(0, 1, 21)
        out["probability_hist"] = {"edges": edges.tolist(), "positive_class": classes[1],
                                   "by_actual": {classes[c]: np.histogram(pos[y == c], bins=edges)[0].astype(int).tolist() for c in labels}}
    else:
        per = {}
        for i, cls in enumerate(classes):
            yi = (y == i).astype(int)
            if 0 < yi.sum() < len(yi):
                per[cls] = float(average_precision_score(yi, proba[:, i]))
        out["pr_auc_ovr"] = per
        maxp = proba.max(axis=1)
        edges = np.linspace(0, 1, 21)
        out["confidence_hist"] = {"edges": edges.tolist(),
                                  "correct": np.histogram(maxp[~wrong], bins=edges)[0].astype(int).tolist(),
                                  "incorrect": np.histogram(maxp[wrong], bins=edges)[0].astype(int).tolist()}
    return out


def regression_diagnostics(pipe: Any, X_test: pd.DataFrame, y_test: Any, categorical: list[str] | None = None,
                           seed: int = 0) -> dict[str, Any]:
    y = np.asarray(y_test, dtype=float)
    pred = np.asarray(pipe.predict(X_test), dtype=float)
    resid = pred - y
    abs_err = np.abs(resid)
    idx = np.arange(len(y))
    if len(idx) > MAX_POINTS:
        idx = np.sort(np.random.default_rng(seed).choice(idx, MAX_POINTS, replace=False))
    out: dict[str, Any] = {"evaluated_on": EVALUATED_ON, "n_test": int(len(y)),
                           "residuals": {"predicted": pred[idx].tolist(), "residual": resid[idx].tolist(), "sampled": bool(len(idx) < len(y))},
                           "abs_error_quantiles": {q: float(np.quantile(abs_err, float(q))) for q in ("0.5", "0.75", "0.9", "0.95", "0.99")} if len(y) else {},
                           "mean_residual": float(resid.mean()) if len(y) else None}
    groups = []
    for c in categorical or []:
        if c not in X_test.columns:
            continue
        s = X_test[c].astype(str).where(X_test[c].notna(), "(missing)").to_numpy()
        frame = pd.DataFrame({"g": s, "e": abs_err})
        agg = frame.groupby("g")["e"].agg(["mean", "count"])
        agg = agg[agg["count"] >= MIN_SUBGROUP]
        if len(agg) >= 2:
            groups.append({"column": c, "groups": [{"value": str(k), "mae": float(r["mean"]), "n": int(r["count"])}
                                                  for k, r in agg.sort_values("mean", ascending=False).head(15).iterrows()]})
    out["error_by_subgroup"] = groups
    return out


# ---------------------------------------------------------------- figures
def fig_confusion(cm: list[list[float]] | None, classes: list[str], normalized: bool = False) -> go.Figure | None:
    if not cm:
        return None
    z = np.asarray(cm, dtype=float)
    text = np.vectorize(lambda v: f"{v:.2f}")(z) if normalized else z.astype(int)
    fig = go.Figure(go.Heatmap(z=z, x=classes, y=classes, colorscale=SEQUENTIAL, showscale=False, zmin=0,
                               zmax=1 if normalized else None, text=text,
                               texttemplate="%{text}" if len(classes) <= 15 else None,
                               hovertemplate="actual %{y}, predicted %{x}: %{z}<extra></extra>"))
    fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
    fig.update_xaxes(type="category")
    fig.update_yaxes(type="category", autorange="reversed")
    title = "Confusion matrix — share of each actual class" if normalized else "Confusion matrix — counts"
    return style(fig, f"{title} (test set)", height=max(340, 30 * len(classes) + 160))


def fig_roc(roc: dict | None, auc: float | None, name: str) -> go.Figure | None:
    if not roc:
        return None
    fig = go.Figure([go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name=name, line=dict(color=SAGE, width=2.5)),
                     go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(color="#B9C2B0", dash="dash"))])
    fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate")
    return style(fig, "ROC curve (test set)" + (f" — AUC {auc:.3f}" if auc is not None else ""), height=380)


def fig_pr(pr: dict | None, name: str) -> go.Figure | None:
    if not pr:
        return None
    fig = go.Figure([go.Scatter(x=pr["recall"], y=pr["precision"], mode="lines", name=name, line=dict(color=SAGE, width=2.5)),
                     go.Scatter(x=[0, 1], y=[pr["baseline"]] * 2, mode="lines", name="Positive-class rate",
                                line=dict(color="#B9C2B0", dash="dash"))])
    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", yaxis_range=[0, 1.02])
    return style(fig, f"Precision–recall (test set) — AP {pr['average_precision']:.3f}", height=380)


def fig_calibration(cal: dict | None) -> go.Figure | None:
    if not cal:
        return None
    fig = go.Figure([go.Scatter(x=cal["mean_predicted"], y=cal["fraction_positive"], mode="lines+markers", name="Model",
                                line=dict(color=SAGE, width=2.5)),
                     go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration", line=dict(color="#B9C2B0", dash="dash"))])
    fig.update_layout(xaxis_title="Mean predicted probability", yaxis_title="Observed share positive", xaxis_range=[0, 1], yaxis_range=[0, 1])
    return style(fig, f"Calibration ({cal['n_bins']} quantile bins, test set)", height=380)


def fig_probability(hist: dict | None) -> go.Figure | None:
    if not hist:
        return None
    edges = np.asarray(hist["edges"])
    mids = (edges[:-1] + edges[1:]) / 2
    fig = go.Figure([go.Bar(x=mids, y=counts, name=f"actual {cls}", marker_color=PALETTE[i % len(PALETTE)], opacity=0.75)
                     for i, (cls, counts) in enumerate(hist["by_actual"].items())])
    fig.update_layout(barmode="overlay", bargap=0.02, xaxis_title=f"Predicted probability of '{hist['positive_class']}'", yaxis_title="Test rows")
    return style(fig, "Predicted probabilities by actual class (test set)", height=360)


def fig_confidence(hist: dict | None) -> go.Figure | None:
    if not hist:
        return None
    edges = np.asarray(hist["edges"])
    mids = (edges[:-1] + edges[1:]) / 2
    fig = go.Figure([go.Bar(x=mids, y=hist["correct"], name="Correct", marker_color=SAGE, opacity=0.8),
                     go.Bar(x=mids, y=hist["incorrect"], name="Incorrect", marker_color="#A65D57", opacity=0.8)])
    fig.update_layout(barmode="overlay", xaxis_title="Highest predicted class probability", yaxis_title="Test rows")
    return style(fig, "Prediction confidence (test set)", height=360)


def fig_cv_scores(models: list[dict[str, Any]], metric_label: str) -> go.Figure | None:
    rows = [(m["name"], s) for m in models if m.get("cv") and m["cv"].get("scores") for s in m["cv"]["scores"]]
    if not rows:
        return None
    t = pd.DataFrame(rows, columns=["model", "score"])
    fig = go.Figure()
    for i, (name, g) in enumerate(t.groupby("model", sort=False)):
        fig.add_trace(go.Box(y=g["score"], name=name, boxpoints="all", jitter=0.3, pointpos=0,
                             marker_color=PALETTE[i % len(PALETTE)], line_color=PALETTE[i % len(PALETTE)]))
    fig.update_layout(showlegend=False, yaxis_title=f"{metric_label} per fold")
    return style(fig, f"Cross-validated {metric_label} per fold (training data, not held-out)", height=380)


def fig_actual_vs_predicted(pred: dict | None, target: str) -> go.Figure | None:
    if not pred:
        return None
    actual, p = np.asarray(pred["actual"]), np.asarray(pred["predicted"])
    if not len(actual):
        return None
    lo, hi = float(min(actual.min(), p.min())), float(max(actual.max(), p.max()))
    fig = go.Figure([go.Scatter(x=actual, y=p, mode="markers", marker=dict(color=SAGE, opacity=0.6), name="Test rows"),
                     go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="#B9C2B0", dash="dash"), name="Perfect prediction")])
    fig.update_layout(xaxis_title=f"Actual {target}", yaxis_title="Predicted")
    return style(fig, "Predicted vs actual (test set)", height=380)


def fig_residual_hist(pred: dict | None) -> go.Figure | None:
    if not pred:
        return None
    resid = np.asarray(pred["predicted"]) - np.asarray(pred["actual"])
    fig = go.Figure(go.Histogram(x=resid, nbinsx=40, marker_color=SAGE))
    fig.update_layout(xaxis_title="Residual (predicted − actual)", yaxis_title="Test rows")
    return style(fig, "Residual distribution (test set)", height=360)


def fig_residuals_vs_predicted(diag: dict | None) -> go.Figure | None:
    if not diag or not diag.get("residuals", {}).get("predicted"):
        return None
    r = diag["residuals"]
    fig = go.Figure([go.Scatter(x=r["predicted"], y=r["residual"], mode="markers", marker=dict(color=SAGE, opacity=0.55), name="Test rows")])
    fig.add_hline(y=0, line=dict(color="#B9C2B0", dash="dash"))
    fig.update_layout(xaxis_title="Predicted value", yaxis_title="Residual (predicted − actual)", showlegend=False)
    return style(fig, "Residuals vs predictions (test set)" + (" — seeded sample" if r.get("sampled") else ""), height=380)


def fig_abs_error(diag: dict | None) -> go.Figure | None:
    q = (diag or {}).get("abs_error_quantiles")
    if not q:
        return None
    labels = [f"{int(float(k) * 100)}th pct" for k in q]
    fig = go.Figure(go.Bar(x=labels, y=list(q.values()), marker_color=SAGE, text=[f"{v:.4g}" for v in q.values()], textposition="outside"))
    fig.update_layout(yaxis_title="Absolute error", xaxis_title=None)
    return style(fig, "Absolute error percentiles (test set)", height=340)


def fig_error_by_group(group: dict | None) -> go.Figure | None:
    if not group:
        return None
    t = pd.DataFrame(group["groups"]).iloc[::-1]
    fig = go.Figure(go.Bar(x=t["mae"], y=t["value"], orientation="h", marker_color=SAGE, customdata=t["n"],
                           hovertemplate="%{y}: MAE %{x:.4g} (n = %{customdata})<extra></extra>"))
    fig.update_yaxes(type="category")
    fig.update_layout(xaxis_title="Mean absolute error", yaxis_title=None)
    return style(fig, f"Error by {group['column']} (groups with ≥ {MIN_SUBGROUP} test rows)", height=max(280, 26 * len(t) + 100))
