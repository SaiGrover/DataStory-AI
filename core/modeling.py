"""Model training with leakage-safe pipelines.

* The train/test split happens first; every preprocessing step (imputation,
  scaling, encoding, SMOTE) lives inside a Pipeline, so it is fitted on training
  data only — and re-fitted inside each cross-validation fold.
* SMOTE is a step of an imbalanced-learn Pipeline, which applies resampling
  during ``fit`` only. Validation folds and the test set are never resampled.
* Classification and regression have separate metric sets.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from core.profiler import infer_kinds

MAX_CLASSES = 50
MAX_CATEGORIES = 50  # categorical features with more distinct values are excluded
MIN_ROWS = 10

CLASSIFICATION_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
REGRESSION_METRICS = ["mae", "mse", "rmse", "r2"]
METRIC_LABELS = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1-score",
                 "roc_auc": "ROC-AUC", "mae": "MAE", "mse": "MSE", "rmse": "RMSE", "r2": "R²"}
LOWER_IS_BETTER = {"mae", "mse", "rmse"}
SUPPORTS_CLASS_WEIGHT = {"Logistic Regression", "Decision Tree", "Random Forest"}


class ModelingError(ValueError):
    """User-readable reason training could not run."""


def _classifiers(seed: int, class_weight):
    return {
        "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=seed, class_weight=class_weight),
                                {"C": [0.1, 1.0, 10.0]}),
        "Decision Tree": (DecisionTreeClassifier(random_state=seed, class_weight=class_weight),
                          {"max_depth": [3, 5, 10, None], "min_samples_leaf": [1, 5, 20]}),
        "Random Forest": (RandomForestClassifier(n_estimators=150, random_state=seed, class_weight=class_weight, n_jobs=1),
                          {"n_estimators": [100, 200], "max_depth": [None, 8, 16], "min_samples_leaf": [1, 5]}),
        "Gradient Boosting": (GradientBoostingClassifier(random_state=seed),
                              {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [2, 3]}),
        "K-Nearest Neighbors": (KNeighborsClassifier(), {"n_neighbors": [3, 5, 11], "weights": ["uniform", "distance"]}),
    }


def _regressors(seed: int):
    return {
        "Linear Regression": (LinearRegression(), {}),
        "Ridge Regression": (Ridge(random_state=seed), {"alpha": [0.1, 1.0, 10.0, 100.0]}),
        "Lasso Regression": (Lasso(random_state=seed, max_iter=10000), {"alpha": [0.001, 0.01, 0.1, 1.0]}),
        "Decision Tree Regressor": (DecisionTreeRegressor(random_state=seed),
                                    {"max_depth": [3, 5, 10, None], "min_samples_leaf": [1, 5, 20]}),
        "Random Forest Regressor": (RandomForestRegressor(n_estimators=150, random_state=seed, n_jobs=1),
                                    {"n_estimators": [100, 200], "max_depth": [None, 8, 16], "min_samples_leaf": [1, 5]}),
        "Gradient Boosting Regressor": (GradientBoostingRegressor(random_state=seed),
                                        {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [2, 3]}),
    }


def available_models(task: str) -> list[str]:
    return list(_classifiers(0, None) if task == "classification" else _regressors(0))


def detect_task(y: pd.Series) -> tuple[str, str]:
    """Return (task, reason). The UI lets the user override this."""
    non_null = y.dropna()
    n_unique = non_null.nunique()
    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        return "classification", f"the target is non-numeric with {n_unique} distinct values"
    if n_unique <= 10 and (non_null == non_null.round()).all():
        return "classification", f"the target is numeric but takes only {n_unique} distinct whole-number values"
    return "regression", f"the target is numeric with {n_unique} distinct values"


def scoring_name(metric: str, task: str, n_classes: int = 2) -> str:
    if task == "regression":
        return {"r2": "r2", "rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error",
                "mse": "neg_mean_squared_error"}[metric]
    if metric == "accuracy":
        return "accuracy"
    if metric == "roc_auc":
        return "roc_auc" if n_classes == 2 else "roc_auc_ovr_weighted"
    return metric if n_classes == 2 else f"{metric}_weighted"


@dataclass
class Prepared:
    X: pd.DataFrame
    y: pd.Series
    numeric: list[str]
    categorical: list[str]
    excluded: list[dict[str, str]]
    datetime_expanded: dict[str, list[str]]
    dropped_missing_target: int


def prepare_features(df: pd.DataFrame, target: str, features: list[str] | None = None) -> Prepared:
    """Row-wise, stateless preparation (safe to do before splitting)."""
    if target not in df.columns:
        raise ModelingError(f"Target column '{target}' is not in the dataset.")
    features = [c for c in (features or [c for c in df.columns if c != target]) if c != target and c in df.columns]
    if not features:
        raise ModelingError("Select at least one feature column.")
    keep = df[target].notna()
    data = df.loc[keep]
    kinds = infer_kinds(data[features])
    X = pd.DataFrame(index=data.index)
    numeric, categorical, excluded, expanded = [], [], [], {}
    for c in features:
        s, kind = data[c], kinds[c]
        if s.notna().sum() == 0:
            excluded.append({"column": c, "reason": "entirely missing"})
        elif kind == "numeric":
            X[c] = pd.to_numeric(s, errors="coerce").astype(float)
            numeric.append(c)
        elif kind == "datetime":
            dt = pd.to_datetime(s, errors="coerce", format="mixed")
            parts = {f"{c}__year": dt.dt.year, f"{c}__month": dt.dt.month, f"{c}__day": dt.dt.day,
                     f"{c}__dayofweek": dt.dt.dayofweek}
            for name, col in parts.items():
                X[name] = col.astype(float)
                numeric.append(name)
            expanded[c] = list(parts)
        elif kind == "text" or s.nunique() > MAX_CATEGORIES:
            excluded.append({"column": c, "reason": f"free text / {s.nunique()} distinct values (limit {MAX_CATEGORIES})"})
        else:
            X[c] = s.astype(object).where(s.notna(), np.nan).map(lambda v: v if pd.isna(v) else str(v))
            categorical.append(c)
    if X.shape[1] == 0:
        raise ModelingError("None of the selected features can be used for modelling.")
    return Prepared(X, data[target], numeric, categorical, excluded, expanded, int((~keep).sum()))


def build_preprocessor(numeric: list[str], categorical: list[str], scale: bool = True) -> ColumnTransformer:
    num_steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline(num_steps), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    return ColumnTransformer(transformers, remainder="drop")


def build_pipeline(estimator, numeric, categorical, scale=True, smote_k: int | None = None, seed: int = 42):
    pre = build_preprocessor(numeric, categorical, scale)
    if smote_k:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline

        return ImbPipeline([("preprocess", pre), ("smote", SMOTE(random_state=seed, k_neighbors=smote_k)), ("model", estimator)])
    return Pipeline([("preprocess", pre), ("model", estimator)])


@dataclass
class TrainingOutput:
    results: dict[str, Any]
    pipelines: dict[str, Any] = field(default_factory=dict)
    X_train: pd.DataFrame | None = None
    X_test: pd.DataFrame | None = None
    y_train: Any = None
    y_test: Any = None


def _classification_metrics(pipe, X_test, y_test, classes: list[str]) -> dict[str, Any]:
    pred = pipe.predict(X_test)
    binary = len(classes) == 2
    avg = "binary" if binary else "weighted"
    out: dict[str, Any] = {"metrics": {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, average=avg, zero_division=0)),
        "recall": float(recall_score(y_test, pred, average=avg, zero_division=0)),
        "f1": float(f1_score(y_test, pred, average=avg, zero_division=0)),
        "roc_auc": None,
    }, "averaging": f"binary (positive class = '{classes[1]}')" if binary else "weighted across classes"}
    labels = list(range(len(classes)))
    try:
        if hasattr(pipe, "predict_proba") and len(np.unique(y_test)) == len(classes):
            proba = pipe.predict_proba(X_test)
            if binary:
                out["metrics"]["roc_auc"] = float(roc_auc_score(y_test, proba[:, 1]))
                fpr, tpr, _ = roc_curve(y_test, proba[:, 1])
                out["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
            else:
                out["metrics"]["roc_auc"] = float(roc_auc_score(y_test, proba, multi_class="ovr", average="weighted", labels=labels))
        else:
            out["roc_auc_note"] = "ROC-AUC unavailable — the test set does not contain every class."
    except ValueError as exc:
        out["roc_auc_note"] = f"ROC-AUC unavailable — {exc}"
    out["confusion_matrix"] = confusion_matrix(y_test, pred, labels=labels).tolist()
    rep = classification_report(y_test, pred, labels=labels, target_names=classes, output_dict=True, zero_division=0)
    out["class_report"] = {k: {m: float(v) for m, v in rep[k].items()} for k in classes}
    return out


def _regression_metrics(pipe, X_test, y_test) -> dict[str, Any]:
    pred = pipe.predict(X_test)
    mse = float(mean_squared_error(y_test, pred))
    out: dict[str, Any] = {"metrics": {"mae": float(mean_absolute_error(y_test, pred)), "mse": mse,
                                       "rmse": float(np.sqrt(mse)), "r2": float(r2_score(y_test, pred)) if len(y_test) > 1 else None}}
    idx = np.arange(len(pred))
    if len(idx) > 2000:
        idx = np.random.default_rng(0).choice(idx, 2000, replace=False)
    out["predictions"] = {"actual": np.asarray(y_test, dtype=float)[idx].tolist(), "predicted": np.asarray(pred, dtype=float)[idx].tolist()}
    return out


def train_models(
    df: pd.DataFrame, target: str, features: list[str] | None = None, task: str | None = None,
    model_names: list[str] | None = None, test_size: float = 0.2, seed: int = 42, cv_folds: int = 5,
    tuning: str = "none", imbalance: str = "none", scale: bool = True, primary_metric: str | None = None,
    n_iter: int = 8,
) -> TrainingOutput:
    warns: list[str] = []
    prep = prepare_features(df, target, features)
    X, y_raw = prep.X, prep.y
    if prep.dropped_missing_target:
        warns.append(f"{prep.dropped_missing_target:,} rows with a missing target were excluded.")
    if len(X) < MIN_ROWS:
        raise ModelingError(f"Only {len(X)} rows have a target value; at least {MIN_ROWS} are needed.")
    auto_task, _ = detect_task(y_raw)
    task = task or auto_task
    if task not in ("classification", "regression"):
        raise ModelingError(f"Unknown task type '{task}'.")

    classes: list[str] = []
    if task == "classification":
        y_str = y_raw.astype(str)
        counts = y_str.value_counts()
        rare = counts[counts < 2].index.tolist()
        if rare:
            warns.append(f"Classes with a single row cannot be split and were excluded: {', '.join(rare[:8])}{'…' if len(rare) > 8 else ''}.")
            keep = ~y_str.isin(rare)
            X, y_str = X[keep], y_str[keep]
            counts = y_str.value_counts()
        if len(counts) < 2:
            raise ModelingError("The target has a single class, so there is nothing to classify. Choose another target.")
        if len(counts) > MAX_CLASSES:
            raise ModelingError(f"The target has {len(counts)} classes (limit {MAX_CLASSES}). If it is a continuous quantity, switch the task to regression.")
        enc = LabelEncoder().fit(y_str)
        classes = [str(c) for c in enc.classes_]
        y = pd.Series(enc.transform(y_str), index=y_str.index)
        primary_metric = primary_metric or "f1"
        if primary_metric not in CLASSIFICATION_METRICS:
            raise ModelingError(f"'{primary_metric}' is not a classification metric.")
    else:
        y = pd.to_numeric(y_raw, errors="coerce")
        if y.isna().any():
            raise ModelingError("The target contains non-numeric values, so regression is not possible. Switch the task to classification or clean the column.")
        if y.nunique() < 2:
            raise ModelingError("The target is constant, so there is nothing to predict.")
        y = y.astype(float)
        primary_metric = primary_metric or "r2"
        if primary_metric not in REGRESSION_METRICS:
            raise ModelingError(f"'{primary_metric}' is not a regression metric.")
        if imbalance != "none":
            warns.append("Class-imbalance handling applies to classification only and was ignored.")
            imbalance = "none"

    n = len(X)
    n_test = int(round(n * test_size))
    if n_test < 2 or n - n_test < 5:
        raise ModelingError(f"A {test_size:.0%} test split of {n} rows leaves too few rows to train or evaluate. Adjust the split or add data.")
    stratify = None
    if task == "classification":
        if n_test < len(classes) or n - n_test < len(classes):
            raise ModelingError(f"The split leaves fewer rows than classes ({len(classes)}). Use a larger dataset or a different split.")
        stratify = y
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=stratify)
    except ValueError as exc:
        raise ModelingError(f"The train/test split failed: {exc}") from exc
    if n < 100:
        warns.append(f"Small dataset ({n} rows): metrics computed on {len(X_test)} test rows are highly uncertain.")

    # cross-validation folds
    folds = int(cv_folds or 0)
    if tuning != "none" and folds < 2:
        folds = 3
        warns.append("Hyperparameter tuning needs cross-validation; 3 folds were used.")
    min_class_train = int(pd.Series(y_train).value_counts().min()) if task == "classification" else len(y_train)
    if folds >= 2:
        limit = min_class_train if task == "classification" else len(y_train) // 2
        if folds > limit:
            warns.append(f"Cross-validation reduced from {folds} to {max(limit, 0)} folds because the smallest training class has {min_class_train} rows."
                         if task == "classification" else f"Cross-validation reduced to {limit} folds for the available rows.")
            folds = limit
        if folds < 2:
            folds = 0
            if tuning != "none":
                warns.append("Too few rows per class for cross-validation; tuning was skipped.")
                tuning = "none"
    cv = None
    if folds >= 2:
        cv = (StratifiedKFold(folds, shuffle=True, random_state=seed) if task == "classification"
              else KFold(folds, shuffle=True, random_state=seed))

    # imbalance handling
    smote_k = None
    if imbalance == "smote":
        smallest_fit = min_class_train if not folds else int(np.floor(min_class_train * (folds - 1) / folds))
        if smallest_fit < 2:
            warns.append("SMOTE needs at least 2 training rows in the smallest class; it was not applied.")
            imbalance = "none"
        else:
            smote_k = min(5, smallest_fit - 1)
    class_weight = "balanced" if imbalance == "class_weight" else None

    registry = _classifiers(seed, class_weight) if task == "classification" else _regressors(seed)
    model_names = model_names or list(registry)
    unknown = [m for m in model_names if m not in registry]
    if unknown:
        raise ModelingError(f"Not available for {task}: {', '.join(unknown)}")
    scoring = scoring_name(primary_metric, task, len(classes) or 2)
    sign = -1.0 if primary_metric in LOWER_IS_BETTER else 1.0

    out = TrainingOutput(results={}, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    models: list[dict[str, Any]] = []
    for name in model_names:
        est, grid = registry[name]
        entry: dict[str, Any] = {"name": name, "notes": [], "cv": None, "best_params": None}
        if imbalance == "class_weight" and name not in SUPPORTS_CLASS_WEIGHT:
            entry["notes"].append("Class weighting is not supported by this model; trained without it.")
        pipe = build_pipeline(est, prep.numeric, prep.categorical, scale, smote_k, seed)
        t0 = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", UserWarning)
                if tuning != "none" and grid and cv is not None:
                    params = {f"model__{k}": v for k, v in grid.items()}
                    if tuning == "grid":
                        search = GridSearchCV(pipe, params, scoring=scoring, cv=cv, n_jobs=1, refit=True)
                    else:
                        total = int(np.prod([len(v) for v in grid.values()]))
                        search = RandomizedSearchCV(pipe, params, n_iter=min(n_iter, total), scoring=scoring, cv=cv,
                                                    n_jobs=1, refit=True, random_state=seed)
                    search.fit(X_train, y_train)
                    pipe = search.best_estimator_
                    i = search.best_index_
                    entry["best_params"] = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
                    entry["cv"] = {"metric": primary_metric, "mean": float(sign * search.cv_results_["mean_test_score"][i]),
                                   "std": float(search.cv_results_["std_test_score"][i]), "folds": folds,
                                   "candidates": int(len(search.cv_results_["params"]))}
                else:
                    if tuning != "none" and not grid:
                        entry["notes"].append("No hyperparameters to tune for this model.")
                    if cv is not None:
                        scores = cross_validate(pipe, X_train, y_train, scoring=scoring, cv=cv, n_jobs=1)["test_score"]
                        entry["cv"] = {"metric": primary_metric, "mean": float(sign * scores.mean()),
                                       "std": float(scores.std()), "folds": folds}
                    pipe.fit(X_train, y_train)
            entry.update(_classification_metrics(pipe, X_test, y_test, classes) if task == "classification"
                         else _regression_metrics(pipe, X_test, y_test))
            out.pipelines[name] = pipe
        except Exception as exc:  # one failing model must not sink the leaderboard
            entry["error"] = f"{type(exc).__name__}: {exc}"
        entry["fit_seconds"] = round(time.perf_counter() - t0, 2)
        if entry["cv"] and not np.isfinite(entry["cv"]["mean"]):
            entry["cv"] = None
            entry["notes"].append("Cross-validation score unavailable — at least one fold could not be scored.")
        models.append(entry)

    ok = [m for m in models if "error" not in m]
    if not ok:
        raise ModelingError("Every selected model failed to train: " + "; ".join(f"{m['name']} — {m['error']}" for m in models))
    use_cv = all(m["cv"] for m in ok)

    def rank_value(m):
        v = m["cv"]["mean"] if use_cv else m["metrics"].get(primary_metric)
        if v is None:
            return float("-inf")
        return -v if primary_metric in LOWER_IS_BETTER else v

    if not use_cv and any(m["metrics"].get(primary_metric) is None for m in ok):
        warns.append(f"{METRIC_LABELS[primary_metric]} is unavailable for some models; they rank last.")
    ok.sort(key=rank_value, reverse=True)
    label = METRIC_LABELS[primary_metric]
    basis = (f"mean {folds}-fold cross-validated {label} on the training data (test set not used for selection)" if use_cv
             else f"{label} on the held-out test set. Because the same test set also selects the winner, its score is "
                  "slightly optimistic; enable cross-validation for an unbiased selection")
    train_counts = {classes[int(k)]: int(v) for k, v in pd.Series(y_train).value_counts().sort_index().items()} if classes else None
    out.results = {
        "task": task, "task_auto": auto_task, "target": target,
        "features": {"numeric": prep.numeric, "categorical": prep.categorical, "excluded": prep.excluded,
                     "datetime_expanded": prep.datetime_expanded, "input_columns": list(X.columns)},
        "n_rows": n, "n_train": len(X_train), "n_test": len(X_test), "test_size": test_size, "seed": seed,
        "stratified": task == "classification", "cv_folds": folds, "tuning": tuning,
        "imbalance": imbalance, "smote_k": smote_k, "scale": scale,
        "preprocessing": ("Median imputation" + (" + standard scaling" if scale else "") + " for numeric features; "
                          "most-frequent imputation + one-hot encoding for categorical features. All steps are fitted "
                          "on training data only, inside the pipeline" + (" and inside each CV fold." if folds else ".")),
        "primary_metric": primary_metric, "ranking_basis": basis, "ranked_by_cv": use_cv,
        "classes": classes, "class_counts_train": train_counts,
        "models": ok + [m for m in models if "error" in m],
        "best_model": ok[0]["name"], "warnings": warns,
        "metric_names": CLASSIFICATION_METRICS if task == "classification" else REGRESSION_METRICS,
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
    return out


def leaderboard(results: dict[str, Any]) -> pd.DataFrame:
    """Tabular view shared by the dashboard and the reports."""
    rows = []
    for m in results["models"]:
        if "error" in m:
            continue
        row = {"Model": m["name"]}
        if m["cv"]:
            row[f"CV {METRIC_LABELS[m['cv']['metric']]} (mean ± sd)"] = f"{m['cv']['mean']:.4f} ± {m['cv']['std']:.4f}"
        for k in results["metric_names"]:
            row[f"Test {METRIC_LABELS[k]}"] = m["metrics"].get(k)
        row["Fit time (s)"] = m["fit_seconds"]
        rows.append(row)
    return pd.DataFrame(rows)
