"""Dataset profiling. Every number returned here is computed from the frame passed in."""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

NEAR_CONSTANT_THRESHOLD = 0.95  # share of non-missing rows taken by the most frequent value
IQR_MULTIPLIER = 1.5  # Tukey fences
OUTLIER_METHOD = (
    f"Tukey's rule: a value is flagged when it lies below Q1 − {IQR_MULTIPLIER}×IQR or above "
    f"Q3 + {IQR_MULTIPLIER}×IQR. Computed per numeric column with more than two distinct values."
)
_ID_NAME = re.compile(r"(^|[_\s])(id|uuid|guid|key|index)($|[_\s])|id$", re.IGNORECASE)
_DATE_HINT = re.compile(r"\d[-/:.]\d")
_MONTH_HINT = re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b|"
                         r"\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?,?\s+\d{4}\b", re.IGNORECASE)
_ID_VALUE = re.compile(r"^[A-Za-z]{0,4}[-_]?\d+$")


def _looks_like_datetime(s: pd.Series) -> bool:
    sample = s.dropna().astype(str).str.strip().head(200)
    if sample.empty:
        return False
    if not (sample.str.contains(_DATE_HINT).mean() > 0.9 or sample.str.contains(_MONTH_HINT).mean() > 0.9):
        return False
    try:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        return False
    return parsed.notna().mean() >= 0.9


def column_kind(s: pd.Series) -> str:
    """Classify a column as numeric / categorical / datetime / text."""
    if ptypes.is_bool_dtype(s):
        return "categorical"
    if ptypes.is_datetime64_any_dtype(s):
        return "datetime"
    if ptypes.is_numeric_dtype(s):
        return "numeric"
    if isinstance(s.dtype, pd.CategoricalDtype):
        return "categorical"
    non_null = s.dropna()
    if non_null.empty:
        return "categorical"
    if _looks_like_datetime(s):
        return "datetime"
    as_str = non_null.astype(str)
    n_unique = as_str.nunique()
    if as_str.str.len().mean() > 40 or (n_unique > 50 and n_unique / len(as_str) > 0.5):
        return "text"
    return "categorical"


def multi_value_separator(s: pd.Series) -> str | None:
    """Return ", " when a text column holds comma-separated lists (e.g. 'United States, India')."""
    x = s.dropna().astype(str).head(2000)
    if len(x) < 20:
        return None
    has = x.str.contains(", ", regex=False)
    if has.mean() < 0.1:
        return None
    tokens = x.str.split(", ").explode().str.strip()
    if tokens.nunique() >= 0.5 * len(tokens) or tokens.str.len().mean() > 40:
        return None
    return ", "


def infer_kinds(df: pd.DataFrame) -> dict[str, str]:
    return {c: column_kind(df[c]) for c in df.columns}


def iqr_bounds(s: pd.Series) -> tuple[float, float] | None:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.nunique() <= 2:
        return None
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    return float(q1 - IQR_MULTIPLIER * iqr), float(q3 + IQR_MULTIPLIER * iqr)


def count_outliers(s: pd.Series) -> int | None:
    b = iqr_bounds(s)
    if b is None:
        return None
    x = pd.to_numeric(s, errors="coerce")
    return int(((x < b[0]) | (x > b[1])).sum())


def _is_id_candidate(name: str, s: pd.Series, kind: str, n_rows: int) -> bool:
    if n_rows < 20:
        return False
    non_null = s.dropna()
    if kind in ("categorical", "text") and len(non_null) and non_null.nunique() >= 0.98 * len(non_null)             and non_null.astype(str).str.match(_ID_VALUE).mean() > 0.95 and len(non_null) >= 0.95 * n_rows:
        return True  # codes such as s1, s2, … or INV-0001
    if s.isna().any() or s.nunique() != n_rows:
        return False
    if kind == "numeric":
        if not ptypes.is_integer_dtype(s):
            return False
        return bool(_ID_NAME.search(name)) or bool((s.sort_values().diff().dropna() == 1).all())
    if kind in ("categorical", "text"):
        return s.astype(str).str.len().mean() <= 40
    return False


def _f(x: Any) -> float | None:
    """JSON-safe float (None for NaN / inf)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def profile_dataset(df: pd.DataFrame, target: str | None = None) -> dict[str, Any]:
    n_rows, n_cols = df.shape
    kinds = infer_kinds(df)
    cells = n_rows * n_cols
    missing_by_col = df.isna().sum()
    missing_cells = int(missing_by_col.sum())
    dup_rows = int(df.duplicated().sum()) if n_rows else 0

    columns: list[dict[str, Any]] = []
    constant, near_constant, id_candidates = [], [], []
    for c in df.columns:
        s = df[c]
        kind = kinds[c]
        non_null = s.dropna()
        n_unique = int(non_null.nunique())
        info: dict[str, Any] = {
            "name": c, "dtype": str(s.dtype), "kind": kind,
            "missing": int(missing_by_col[c]),
            "missing_pct": _f(100 * missing_by_col[c] / n_rows) if n_rows else None,
            "unique": n_unique,
            "unique_pct": _f(100 * n_unique / len(non_null)) if len(non_null) else None,
        }
        if len(non_null):
            vc = non_null.value_counts()
            top_share = vc.iloc[0] / len(non_null)
            info["top_value"] = str(vc.index[0])
            info["top_count"] = int(vc.iloc[0])
            info["top_share_pct"] = _f(100 * top_share)
            if n_unique == 1:
                constant.append(c)
            elif top_share >= NEAR_CONSTANT_THRESHOLD:
                near_constant.append(c)
        elif n_rows:
            constant.append(c)  # entirely missing
        if kind == "numeric" and len(non_null):
            x = non_null.astype(float)
            info.update(
                mean=_f(x.mean()), std=_f(x.std()), min=_f(x.min()), q1=_f(x.quantile(0.25)),
                median=_f(x.median()), q3=_f(x.quantile(0.75)), max=_f(x.max()),
                skew=_f(x.skew()) if len(x) > 2 else None,
            )
            out = count_outliers(s)
            info["outliers"] = out
            info["outlier_pct"] = _f(100 * out / len(non_null)) if out is not None else None
        if kind == "datetime" and len(non_null):
            dt = pd.to_datetime(non_null, errors="coerce", format="mixed").dropna()
            if len(dt):
                info["min"], info["max"] = str(dt.min()), str(dt.max())
        if _is_id_candidate(c, s, kind, n_rows):
            id_candidates.append(c)
        columns.append(info)

    kind_counts = {k: sum(1 for v in kinds.values() if v == k) for k in ("numeric", "categorical", "datetime", "text")}
    dtype_counts = df.dtypes.astype(str).value_counts().to_dict()

    profile: dict[str, Any] = {
        "rows": n_rows, "columns": n_cols, "cells": cells,
        "memory_mb": _f(df.memory_usage(deep=True).sum() / 1024 / 1024),
        "missing_cells": missing_cells,
        "missing_pct": _f(100 * missing_cells / cells) if cells else None,
        "columns_with_missing": int((missing_by_col > 0).sum()),
        "duplicate_rows": dup_rows,
        "duplicate_pct": _f(100 * dup_rows / n_rows) if n_rows else None,
        "kinds": kinds, "kind_counts": kind_counts, "dtype_counts": {str(k): int(v) for k, v in dtype_counts.items()},
        "column_details": columns,
        "constant_columns": constant, "near_constant_columns": near_constant,
        "id_candidates": id_candidates,
        "outlier_method": OUTLIER_METHOD,
        "outlier_columns": sorted(
            [{"column": c["name"], "outliers": c["outliers"], "outlier_pct": c["outlier_pct"]}
             for c in columns if c.get("outliers")],
            key=lambda r: -r["outliers"],
        ),
        "target": None,
    }
    if target and target in df.columns:
        profile["target"] = target_summary(df[target], kinds[target])
        profile["target"]["name"] = target
    return profile


def target_summary(y: pd.Series, kind: str) -> dict[str, Any]:
    non_null = y.dropna()
    out: dict[str, Any] = {"missing": int(y.isna().sum()), "unique": int(non_null.nunique()), "kind": kind}
    discrete = kind != "numeric" or non_null.nunique() <= 20
    out["looks_discrete"] = bool(discrete)
    if discrete and len(non_null):
        vc = non_null.astype(str).value_counts()
        out["class_counts"] = {str(k): int(v) for k, v in vc.items()}
        out["minority_share_pct"] = _f(100 * vc.min() / vc.sum())
        out["majority_share_pct"] = _f(100 * vc.max() / vc.sum())
        out["imbalance_ratio"] = _f(vc.max() / vc.min())
    return out


def correlation_pairs(df: pd.DataFrame, kinds: dict[str, str], min_abs: float = 0.0, exclude: list[str] | None = None) -> list[dict[str, Any]]:
    """Pearson correlations between numeric columns, strongest first."""
    cols = [c for c, k in kinds.items() if k == "numeric" and c not in (exclude or []) and df[c].nunique() > 1]
    if len(cols) < 2:
        return []
    corr = df[cols].corr(numeric_only=True)
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) >= min_abs:
                n = int(df[[a, b]].dropna().shape[0])
                pairs.append({"a": a, "b": b, "r": float(r), "n": n})
    return sorted(pairs, key=lambda p: -abs(p["r"]))
