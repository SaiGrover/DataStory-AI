"""Pre-training suitability assessment.

Produces specific findings (blocker / warning / info) and training
recommendations. There is deliberately no overall percentage score: the status
is "blocked" if any blocker exists, "caution" if any warning exists, else
"ready".
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from core import targets as tg
from core.formatting import count, pct
from core.profiler import infer_kinds

LEAKAGE_CORR = 0.95
LEAKAGE_PURITY = 0.99
MAX_CATEGORIES = 50
SPARSE_PCT = 50.0
IMBALANCE_RATIO = 3.0


def _f(findings, severity, title, detail):
    findings.append({"severity": severity, "title": title, "detail": detail})


def _leakage(df: pd.DataFrame, target: str, features: list[str], kinds: dict[str, str], task: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    y = df[target]
    tname = str(target).lower()
    y_num = None
    if task == "regression":
        y_num = pd.to_numeric(y, errors="coerce")
    else:
        vals = y.dropna().astype(str)
        if vals.nunique() == 2:
            pos = sorted(vals.unique())[1]
            y_num = (y.astype(str) == pos).astype(float).where(y.notna())
    for f in features:
        if f == target:
            continue
        s = df[f]
        if len(tname) >= 3 and tname in str(f).lower():
            out.append((f, f"its name contains the target name '{target}'"))
            continue
        if kinds.get(f) == "numeric" and y_num is not None:
            pair = pd.concat([pd.to_numeric(s, errors="coerce"), y_num], axis=1).dropna()
            if len(pair) >= 10 and pair.iloc[:, 0].nunique() > 1 and pair.iloc[:, 1].nunique() > 1:
                r = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
                if np.isfinite(r) and abs(r) >= LEAKAGE_CORR:
                    out.append((f, f"|r| = {abs(r):.3f} with the target"))
                    continue
        if kinds.get(f) in ("categorical", "numeric") and task == "classification":
            pair = pd.DataFrame({"f": s.astype(str), "y": y.astype(str)})[s.notna() & y.notna()]
            n_levels = pair["f"].nunique()
            if len(pair) >= 20 and 1 < n_levels <= max(MAX_CATEGORIES, 2) and n_levels < len(pair) / 2:
                purity = float(pair.groupby("f")["y"].agg(lambda v: v.value_counts().iloc[0]).sum() / len(pair))
                if purity >= LEAKAGE_PURITY:
                    out.append((f, f"each of its {n_levels} values maps to a single target class ({pct(100 * purity)} purity)"))
    return out


def assess_readiness(df: pd.DataFrame, target: str | None, features: list[str] | None, task: str | None,
                     kinds: dict[str, str] | None = None, test_size: float = 0.2, cv_folds: int = 5) -> dict[str, Any]:
    findings: list[dict[str, str]] = []
    rec: dict[str, Any] = {"task": task, "metric": None, "metric_reason": "", "models": [], "cv": cv_folds, "cv_reason": "",
                           "imbalance": "none", "imbalance_reason": "", "exclude": []}
    kinds = kinds or infer_kinds(df)
    if not target or target not in df.columns:
        _f(findings, "blocker", "No target selected", "Choose the column you want to predict.")
        return {"status": "blocked", "findings": findings, "recommendations": rec}
    v = tg.validate_target(df, target, task, kinds)
    for b in v["blocking"]:
        _f(findings, "blocker", "Target is not a valid prediction target", f"'{target}': {b}.")
    for w in v["warnings"]:
        _f(findings, "warning", "Target warning", f"'{target}': {w}.")
    task = task or (v["candidate"] or {}).get("task")
    rec["task"] = task
    features = [f for f in (features or []) if f in df.columns and f != target]
    if not features:
        _f(findings, "blocker", "No features selected", "Select at least one feature column.")

    y = df[target]
    n_missing_y = int(y.isna().sum())
    n = int(y.notna().sum())
    if n_missing_y:
        _f(findings, "warning" if n_missing_y / max(len(df), 1) > 0.05 else "info", "Missing target values",
           f"{count(n_missing_y)} rows ({pct(100 * n_missing_y / len(df))}) have no target value and will be excluded.")
    if n < 30:
        _f(findings, "blocker" if n < 10 else "warning", "Very small dataset",
           f"Only {count(n)} rows have a target value; metrics on a test split of about {max(1, round(n * test_size))} rows are highly uncertain.")
    elif n < 200:
        _f(findings, "info", "Small dataset", f"{count(n)} labelled rows: expect wide variation between splits; cross-validation is recommended.")

    dups = int(df.duplicated().sum())
    if dups:
        _f(findings, "warning", "Duplicate rows", f"{count(dups)} exact duplicate rows exist. Copies can land in both train and test sets and inflate scores; consider removing them.")

    n_test = int(round(n * test_size))
    if task == "classification" and v["ok"]:
        counts = y.dropna().astype(str).value_counts()
        k = len(counts)
        ratio = float(counts.max() / counts.min()) if counts.min() else float("inf")
        _f(findings, "info", "Class distribution",
           f"{k} classes; largest '{counts.index[0]}' {count(counts.iloc[0])} rows, smallest '{counts.index[-1]}' {count(counts.iloc[-1])} rows (ratio {ratio:.2f}:1).")
        min_test = counts.min() * test_size
        min_train = counts.min() * (1 - test_size)
        if min_test < 1:
            _f(findings, "blocker" if counts.min() < 2 else "warning", "Split feasibility",
               f"The smallest class has {count(counts.min())} row(s); a {pct(100 * test_size, 0)} test split leaves it with fewer than one expected test row.")
        if cv_folds and min_train < cv_folds:
            _f(findings, "warning", "Cross-validation feasibility",
               f"The smallest class would have about {int(min_train)} training rows, fewer than {cv_folds} folds; folds will be reduced automatically.")
        if ratio >= IMBALANCE_RATIO:
            _f(findings, "warning", "Class imbalance", f"Majority/minority ratio is {ratio:.2f}:1. Accuracy alone would be misleading.")
            rec["metric"] = "f1"
            rec["metric_reason"] = "F1 balances precision and recall on the minority class; accuracy is inflated by the majority class when classes are imbalanced."
            rec["imbalance"] = "class_weight"
            rec["imbalance_reason"] = "Class weights re-balance the loss without creating synthetic rows; try SMOTE (training folds only) as a comparison."
        else:
            rec["metric"] = "f1" if k > 2 else "accuracy"
            rec["metric_reason"] = ("Classes are reasonably balanced, so accuracy is interpretable" if k == 2 else
                                    "Weighted F1 summarises performance across several classes") + "; also check per-class recall."
            rec["imbalance_reason"] = "No imbalance handling needed at this class ratio."
        rec["models"] = ["Logistic Regression", "Random Forest", "Gradient Boosting"]
    elif task == "regression" and v["ok"]:
        rec["metric"] = "rmse"
        rec["metric_reason"] = "RMSE is in the target's own units and penalises large errors; read it alongside R² (share of variance explained)."
        rec["models"] = ["Linear Regression", "Ridge Regression", "Random Forest Regressor"]
        yy = pd.to_numeric(y, errors="coerce").dropna()
        if len(yy) > 2 and abs(float(yy.skew())) > 2:
            _f(findings, "info", "Skewed target", f"The target's skewness is {float(yy.skew()):.2f}; errors will be dominated by the long tail. MAE is less sensitive to it than RMSE.")

    if n >= 1000:
        rec["cv"] = 5
        rec["cv_reason"] = "5-fold cross-validation on the training data gives a stable model ranking at this size."
    elif n >= 100:
        rec["cv"] = 5
        rec["cv_reason"] = "Cross-validation is important at this size because a single test split is noisy."
    else:
        rec["cv"] = 3
        rec["cv_reason"] = "3 folds keep enough rows in each fold for a small dataset."

    # features
    for f in features:
        s = df[f]
        ident, why = tg.is_identifier(f, s, len(df))
        if ident:
            rec["exclude"].append({"column": f, "reason": f"identifier-like ({why})"})
            continue
        if kinds.get(f) == "text":
            rec["exclude"].append({"column": f, "reason": "free text — excluded by the modelling pipeline"})
            continue
        if kinds.get(f) == "categorical" and s.nunique() > MAX_CATEGORIES:
            rec["exclude"].append({"column": f, "reason": f"{s.nunique():,} categories (limit {MAX_CATEGORIES}) — excluded by the modelling pipeline"})
            continue
        miss = float(100 * s.isna().mean())
        if miss > SPARSE_PCT:
            rec["exclude"].append({"column": f, "reason": f"{pct(miss)} missing — imputation would dominate this feature"})
        if s.nunique(dropna=True) <= 1:
            rec["exclude"].append({"column": f, "reason": "constant"})
    ids = [e for e in rec["exclude"] if e["reason"].startswith("identifier")]
    if ids:
        _f(findings, "warning", "Identifier-like features",
           "Identifier columns can let a model memorise rows: " + ", ".join(e["column"] for e in ids) + ".")
    other = [e for e in rec["exclude"] if not e["reason"].startswith("identifier")]
    if other:
        _f(findings, "info", "Features that need attention", "; ".join(f"{e['column']} ({e['reason']})" for e in other[:8]) + ".")
    if v["ok"] and features:
        leaks = _leakage(df, target, features, kinds, task)
        for col, why in leaks:
            _f(findings, "warning", "Possible target leakage",
               f"'{col}': {why}. If it is only known after the outcome, exclude it — otherwise test scores will be unrealistically high.")
            rec["exclude"].append({"column": col, "reason": f"possible leakage ({why})"})
    usable = len([f for f in features if f not in {e['column'] for e in rec['exclude']}])
    if features and n and usable / n > 0.5:
        _f(findings, "warning", "Many features relative to rows",
           f"{usable} usable features for {count(n)} labelled rows; models may overfit. Prefer regularised models or fewer features.")
    if any(kinds.get(c) == "datetime" for c in df.columns):
        _f(findings, "info", "Time-ordered data",
           "The dataset has a date column. If you will predict future records, a temporal split (train on earlier, test on later) "
           "is more realistic than the random split used here.")

    status = "blocked" if any(f["severity"] == "blocker" for f in findings) else (
        "caution" if any(f["severity"] == "warning" for f in findings) else "ready")
    return {"status": status, "findings": findings, "recommendations": rec}
