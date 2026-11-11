"""Cleaning operations.

Each operation is a pure function ``(df, **params) -> (new_df, detail)``. Inputs
are never mutated. ``apply_operation`` wraps a call and produces a history
record with before/after counts so the UI can show exactly what changed.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
import pandas as pd
from pandas.api import types as ptypes


class CleaningError(ValueError):
    """User-readable reason an operation could not run."""


def _need(df: pd.DataFrame, columns: list[str]) -> list[str]:
    if not columns:
        raise CleaningError("Select at least one column.")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise CleaningError(f"Column(s) not found: {', '.join(missing)}")
    return list(columns)


def _numeric_only(df: pd.DataFrame, columns: list[str], what: str) -> None:
    bad = [c for c in columns if not ptypes.is_numeric_dtype(df[c]) or ptypes.is_bool_dtype(df[c])]
    if bad:
        raise CleaningError(f"{what} needs numeric columns; not numeric: {', '.join(bad)}")


# ---------------------------------------------------------------- operations
def impute(df, columns, strategy="median", value=None, n_neighbors=5):
    columns = _need(df, columns)
    out = df.copy()
    filled = 0
    if strategy == "knn":
        _numeric_only(df, columns, "KNN imputation")
        from sklearn.impute import KNNImputer

        num_cols = [c for c in df.columns if ptypes.is_numeric_dtype(df[c]) and not ptypes.is_bool_dtype(df[c])
                    and df[c].notna().any()]
        columns = [c for c in columns if c in num_cols]
        if not columns:
            raise CleaningError("The selected columns contain no values to learn from.")
        k = max(1, min(int(n_neighbors), len(df) - 1))
        imputed = KNNImputer(n_neighbors=k).fit_transform(df[num_cols].astype(float))
        imputed = pd.DataFrame(imputed, columns=num_cols, index=df.index)
        for c in columns:
            filled += int(df[c].isna().sum())
            out[c] = imputed[c]
        return out, f"KNN imputation (k={k}, neighbours measured on {len(num_cols)} numeric columns) filled {filled:,} values."
    used = {}
    for c in columns:
        s = df[c]
        n = int(s.isna().sum())
        if n == 0:
            continue
        if strategy in ("mean", "median"):
            _numeric_only(df, [c], f"{strategy.title()} imputation")
            if s.notna().sum() == 0:
                raise CleaningError(f"'{c}' is entirely missing; there is no {strategy} to use.")
            fill = float(s.mean() if strategy == "mean" else s.median())
            out[c] = s.astype(float).fillna(fill)
        elif strategy == "mode":
            mode = s.mode(dropna=True)
            if mode.empty:
                raise CleaningError(f"'{c}' is entirely missing; there is no mode to use.")
            fill = mode.iloc[0]
            out[c] = s.fillna(fill)
        elif strategy == "constant":
            if value is None or value == "":
                raise CleaningError("Provide the constant value to fill with.")
            fill = value
            if ptypes.is_numeric_dtype(s):
                try:
                    fill = float(value)
                except (TypeError, ValueError) as exc:
                    raise CleaningError(f"'{c}' is numeric; '{value}' is not a number.") from exc
                out[c] = s.astype(float).fillna(fill)
            else:
                out[c] = s.astype(object).fillna(str(value))
        else:
            raise CleaningError(f"Unknown imputation strategy '{strategy}'.")
        used[c] = fill
        filled += n
    if not filled:
        return out, "No missing values in the selected columns; nothing changed."
    shown = ", ".join(f"{c} ← {v:.4g}" if isinstance(v, float) else f"{c} ← {v}" for c, v in list(used.items())[:6])
    return out, f"{strategy.title()} imputation filled {filled:,} values ({shown}{'…' if len(used) > 6 else ''})."


def drop_missing_rows(df, columns=None, how="any"):
    columns = _need(df, columns) if columns else list(df.columns)
    out = df.dropna(subset=columns, how=how).reset_index(drop=True)
    return out, f"Dropped {len(df) - len(out):,} rows with {how} missing value in {len(columns)} column(s)."


def drop_columns(df, columns):
    columns = _need(df, columns)
    if len(columns) >= df.shape[1]:
        raise CleaningError("Cannot drop every column.")
    return df.drop(columns=columns), f"Removed columns: {', '.join(columns)}."


def drop_sparse_columns(df, threshold_pct=50.0):
    share = df.isna().mean() * 100
    cols = [c for c in df.columns if share[c] > threshold_pct]
    if len(cols) >= df.shape[1]:
        raise CleaningError("That threshold would remove every column.")
    if not cols:
        return df.copy(), f"No column has more than {threshold_pct:g}% missing values; nothing changed."
    return df.drop(columns=cols), f"Removed {len(cols)} column(s) with > {threshold_pct:g}% missing: {', '.join(cols)}."


def drop_duplicates(df, columns=None, keep="first"):
    subset = _need(df, columns) if columns else None
    out = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
    scope = f"on {', '.join(subset)}" if subset else "across all columns"
    return out, f"Removed {len(df) - len(out):,} duplicate rows ({scope}, keeping the {keep} occurrence)."


def convert_type(df, columns, to="numeric"):
    columns = _need(df, columns)
    out = df.copy()
    notes = []
    for c in columns:
        s = df[c]
        before_na = int(s.isna().sum())
        if to == "numeric":
            cleaned = s.astype(str).str.replace(r"[,\s$€£%]", "", regex=True) if not ptypes.is_numeric_dtype(s) else s
            new = pd.to_numeric(cleaned, errors="coerce")
            new[s.isna()] = np.nan
        elif to == "integer":
            new = pd.to_numeric(s, errors="coerce").round().astype("Int64")
        elif to == "string":
            new = s.astype("string")
        elif to == "category":
            new = s.astype("category")
        elif to == "datetime":
            new = pd.to_datetime(s, errors="coerce", format="mixed")
        elif to == "boolean":
            m = {"true": True, "yes": True, "y": True, "1": True, "t": True,
                 "false": False, "no": False, "n": False, "0": False, "f": False}
            new = s.astype(str).str.strip().str.lower().map(m).astype("boolean")
            new[s.isna()] = pd.NA
        else:
            raise CleaningError(f"Unknown target type '{to}'.")
        lost = int(new.isna().sum()) - before_na
        out[c] = new
        notes.append(f"{c}: {s.dtype} → {new.dtype}" + (f" ({lost:,} unparseable values became missing)" if lost > 0 else ""))
    return out, "; ".join(notes) + "."


def normalize_categories(df, columns, case="lower", strip=True, mapping=None):
    columns = _need(df, columns)
    out = df.copy()
    notes = []
    for c in columns:
        s = df[c]
        mask = s.notna()
        new = s.astype(object).copy()
        vals = s[mask].astype(str)
        if strip:
            vals = vals.str.strip().str.replace(r"\s+", " ", regex=True)
        if case in ("lower", "upper", "title"):
            vals = getattr(vals.str, case)()
        if mapping:
            vals = vals.replace({str(k): str(v) for k, v in mapping.items()})
        new[mask] = vals
        out[c] = new if s.dtype == object else new.astype("string")
        notes.append(f"{c}: {s[mask].nunique()} → {vals.nunique()} distinct values")
    return out, "Normalised categories — " + "; ".join(notes) + "."


def clean_text(df, columns, strip=True, collapse_spaces=True, lower=False, remove_punctuation=False):
    columns = _need(df, columns)
    out = df.copy()
    changed = 0
    for c in columns:
        s = df[c]
        mask = s.notna()
        vals = s[mask].astype(str)
        orig = vals.copy()
        if strip:
            vals = vals.str.strip()
        if collapse_spaces:
            vals = vals.str.replace(r"\s+", " ", regex=True)
        if lower:
            vals = vals.str.lower()
        if remove_punctuation:
            vals = vals.str.replace(r"[^\w\s]", "", regex=True)
        changed += int((vals != orig).sum())
        new = s.astype(object).copy()
        new[mask] = vals
        out[c] = new if s.dtype == object else new.astype("string")
    return out, f"Text cleanup modified {changed:,} values in {len(columns)} column(s)."


def parse_dates(df, columns, dayfirst=False):
    columns = _need(df, columns)
    out = df.copy()
    notes = []
    for c in columns:
        before = int(df[c].isna().sum())
        new = pd.to_datetime(df[c], errors="coerce", dayfirst=dayfirst, format="mixed")
        failed = int(new.isna().sum()) - before
        if new.notna().sum() == 0:
            raise CleaningError(f"No value in '{c}' could be parsed as a date.")
        out[c] = new
        notes.append(f"{c}: parsed {int(new.notna().sum()):,} dates" + (f", {failed:,} unparseable → missing" if failed else ""))
    return out, "; ".join(notes) + "."


def outlier_mask(s: pd.Series, method: str = "iqr", threshold: float | None = None) -> tuple[pd.Series, float, float]:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    if method == "iqr":
        k = 1.5 if threshold is None else float(threshold)
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
    elif method == "zscore":
        k = 3.0 if threshold is None else float(threshold)
        mu, sd = x.mean(), x.std()
        if not sd or np.isnan(sd):
            return pd.Series(False, index=s.index), float("nan"), float("nan")
        lo, hi = mu - k * sd, mu + k * sd
    else:
        raise CleaningError(f"Unknown outlier method '{method}'.")
    return (x < lo) | (x > hi), float(lo), float(hi)


def handle_outliers(df, columns, method="iqr", threshold=None, action="cap"):
    columns = _need(df, columns)
    _numeric_only(df, columns, "Outlier handling")
    out = df.copy()
    drop = pd.Series(False, index=df.index)
    notes, total = [], 0
    for c in columns:
        mask, lo, hi = outlier_mask(df[c], method, threshold)
        n = int(mask.sum())
        total += n
        if n == 0:
            continue
        if action == "cap":
            out[c] = df[c].astype(float).clip(lo, hi)
        elif action == "null":
            out[c] = df[c].astype(float).mask(mask)
        elif action == "remove":
            drop |= mask
        else:
            raise CleaningError(f"Unknown outlier action '{action}'.")
        notes.append(f"{c}: {n:,} outside [{lo:.4g}, {hi:.4g}]")
    if action == "remove":
        out = out[~drop].reset_index(drop=True)
    if total == 0:
        return out, f"No outliers detected with the {method} method; nothing changed."
    verb = {"cap": "Capped", "null": "Set to missing", "remove": f"Removed {int(drop.sum()):,} rows containing"}[action]
    return out, f"{verb} outliers ({method}) — " + "; ".join(notes) + "."


def scale(df, columns, method="standard"):
    columns = _need(df, columns)
    _numeric_only(df, columns, "Scaling")
    out = df.copy()
    for c in columns:
        x = df[c].astype(float)
        if method == "standard":
            centre, spread = x.mean(), x.std(ddof=0)
        elif method == "minmax":
            centre, spread = x.min(), x.max() - x.min()
        elif method == "robust":
            centre, spread = x.median(), x.quantile(0.75) - x.quantile(0.25)
        else:
            raise CleaningError(f"Unknown scaling method '{method}'.")
        if not spread or np.isnan(spread):
            raise CleaningError(f"'{c}' has no spread, so it cannot be scaled with the {method} method.")
        out[c] = (x - centre) / spread
    return out, (f"{method.title()} scaling applied to {', '.join(columns)}. Statistics were computed on the full "
                 "dataset, so use the modelling page's built-in scaling (fitted on training data only) for ML.")


def encode(df, columns, method="onehot", drop_first=False, max_categories=30):
    columns = _need(df, columns)
    out = df.copy()
    notes = []
    for c in columns:
        n = df[c].nunique(dropna=True)
        if method == "onehot":
            if n > max_categories:
                raise CleaningError(f"'{c}' has {n} categories; one-hot encoding is limited to {max_categories}.")
            dummies = pd.get_dummies(df[c], prefix=c, drop_first=drop_first, dtype=int)
            clash = [d for d in dummies.columns if d in out.columns]
            if clash:
                raise CleaningError(f"Encoding '{c}' would create existing column(s): {', '.join(clash)}")
            pos = out.columns.get_loc(c)
            out = pd.concat([out.iloc[:, :pos], dummies, out.iloc[:, pos + 1:]], axis=1)
            notes.append(f"{c} → {dummies.shape[1]} indicator columns")
        elif method == "ordinal":
            cats = sorted(df[c].dropna().astype(str).unique())
            codes = df[c].astype(str).map({v: i for i, v in enumerate(cats)})
            out[c] = codes.where(df[c].notna()).astype("Int64")
            notes.append(f"{c} → integer codes 0–{len(cats) - 1} (alphabetical order)")
        else:
            raise CleaningError(f"Unknown encoding method '{method}'.")
    return out, "Encoded " + "; ".join(notes) + "."


def rename_columns(df, mapping):
    mapping = {k: str(v).strip() for k, v in (mapping or {}).items() if str(v).strip() and str(v).strip() != k}
    if not mapping:
        raise CleaningError("Enter at least one new column name.")
    _need(df, list(mapping))
    final = [mapping.get(c, c) for c in df.columns]
    if len(set(final)) != len(final):
        raise CleaningError("Renaming would create duplicate column names.")
    return df.rename(columns=mapping), "Renamed " + ", ".join(f"{k} → {v}" for k, v in mapping.items()) + "."


OPERATIONS: dict[str, tuple[str, Callable[..., tuple[pd.DataFrame, str]]]] = {
    "impute": ("Fill missing values", impute),
    "drop_missing_rows": ("Drop rows with missing values", drop_missing_rows),
    "drop_sparse_columns": ("Drop mostly-empty columns", drop_sparse_columns),
    "drop_duplicates": ("Remove duplicate rows", drop_duplicates),
    "convert_type": ("Convert data type", convert_type),
    "parse_dates": ("Parse dates", parse_dates),
    "normalize_categories": ("Normalise categories", normalize_categories),
    "clean_text": ("Clean text", clean_text),
    "handle_outliers": ("Handle outliers", handle_outliers),
    "scale": ("Scale numeric columns", scale),
    "encode": ("Encode categorical columns", encode),
    "rename_columns": ("Rename columns", rename_columns),
    "drop_columns": ("Remove columns", drop_columns),
}


# ------------------------------------------------------------------ history
def snapshot(df: pd.DataFrame) -> dict[str, int]:
    return {"rows": len(df), "columns": df.shape[1], "missing_cells": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()) if len(df) else 0}


def cells_changed(before: pd.DataFrame, after: pd.DataFrame) -> int | None:
    """Count of differing cells when shape and labels are unchanged; otherwise None."""
    if before.shape != after.shape or list(before.columns) != list(after.columns):
        return None
    na_a, na_b = before.isna().to_numpy(), after.isna().to_numpy()
    # compare via string form, with missing values blanked out so NA comparisons never raise
    a = before.astype(object).where(~na_a, "").astype(str).to_numpy()
    b = after.astype(object).where(~na_b, "").astype(str).to_numpy()
    diff = ((a != b) & ~na_a & ~na_b) | (na_a != na_b)
    return int(diff.sum())


def preview_operation(df: pd.DataFrame, op: str, params: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run an operation on a copy and return (result, record). Never mutates ``df``."""
    if op not in OPERATIONS:
        raise CleaningError(f"Unknown operation '{op}'.")
    if df.empty:
        raise CleaningError("The dataset has no rows to clean.")
    label, fn = OPERATIONS[op]
    try:
        new_df, detail = fn(df, **params)
    except CleaningError:
        raise
    except Exception as exc:
        raise CleaningError(f"{label} failed: {exc}") from exc
    if new_df.empty or new_df.shape[1] == 0:
        raise CleaningError(f"{label} would leave the dataset empty, so it was not applied.")
    record = {
        "operation": op, "label": label, "params": _jsonable(params), "detail": detail,
        "before": snapshot(df), "after": snapshot(new_df),
        "cells_changed": cells_changed(df, new_df),
        "columns_added": [c for c in new_df.columns if c not in df.columns],
        "columns_removed": [c for c in df.columns if c not in new_df.columns],
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
    record["changed"] = bool(
        record["before"] != record["after"] or record["cells_changed"] or record["columns_added"]
        or record["columns_removed"] or not df.dtypes.astype(str).equals(new_df.dtypes.astype(str))
    )
    return new_df, record


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x


def replay(original: pd.DataFrame, history: list[dict[str, Any]]) -> pd.DataFrame:
    """Rebuild the working frame by re-running recorded operations on the original."""
    df = original
    for rec in history:
        df, _ = preview_operation(df, rec["operation"], rec["params"])
    return df


def suggest_steps(profile: dict[str, Any]) -> list[str]:
    """Evidence-based suggestions. These are prompts for the analyst, not claims of benefit."""
    tips = []
    if profile["duplicate_rows"]:
        tips.append(f"{profile['duplicate_rows']:,} exact duplicate rows exist — review and consider removing them.")
    for c in profile["column_details"]:
        pct = c["missing_pct"] or 0
        if pct > 50:
            tips.append(f"'{c['name']}' is {pct:.1f}% missing — consider dropping it or treating missingness as information.")
        elif pct > 0:
            how = "median" if c["kind"] == "numeric" else "mode"
            tips.append(f"'{c['name']}' is {pct:.1f}% missing — {how} imputation is a common starting point.")
    for c in profile["constant_columns"]:
        tips.append(f"'{c}' holds a single value — it carries no information for analysis.")
    for c in profile["id_candidates"]:
        tips.append(f"'{c}' looks like an identifier — usually excluded from modelling.")
    for r in profile["outlier_columns"][:5]:
        tips.append(f"'{r['column']}' has {r['outliers']:,} values outside the Tukey fences ({r['outlier_pct']:.1f}%) — inspect before deciding to cap or keep.")
    return tips
