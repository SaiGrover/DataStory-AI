"""Target-suitability assessment for supervised learning.

Every column is assessed as a possible prediction target. Identifiers, free
text, constants and unusable high-cardinality columns are blocked; the rest are
ranked by a documented score. Nothing here assumes a dataset *must* have a
target: when no column is a convincing outcome, the recommendation is
descriptive analysis and clustering.
"""
from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from core.formatting import pct
from core.profiler import infer_kinds

MAX_CLASSES = 50          # modelling refuses more classes than this
HIGH_CARDINALITY = 20     # above this a classification target is discouraged
MIN_MINORITY = 5          # smallest class below this is a warning
MAX_MISSING_PCT = 30.0    # more missing target values than this is a warning
SCORE_WEIGHTS = {"outcome_name": 0.35, "structure": 0.25, "completeness": 0.20, "balance": 0.20}
SCORE_FORMULA = (
    "score = 0.35 × outcome-name hint (1 if the column name matches a common outcome word, else 0) "
    "+ 0.25 × structure (classification: 1.0 for 2–10 classes, 0.6 for 11–20, 0.2 for 21–50; regression: 0.8 for a "
    "continuous numeric column) + 0.20 × completeness (share of non-missing values) "
    "+ 0.20 × balance (classification: normalised class entropy; regression: 1.0 with ≥ 10 distinct values, else 0.5). "
    "Blocked columns get no score. 'Recommended' needs an outcome-name hint and score ≥ 0.6; without a hint a "
    "column is at most 'possible'."
)

OUTCOME_HINT = re.compile(
    r"(^|[_\s\-.])(churn(ed)?|attrition|target|label|class|outcome|survived|survival|price|sales|revenue|default(ed)?|"
    r"fraud(ulent)?|is_fraud|y|grade|g3|cnt|count|species|diagnosis|result|response|converted|conversion|"
    r"exited|left|status|score|salary|income|charges|quality|rating_target)($|[_\s\-.])",
    re.IGNORECASE,
)
_ID_NAME = re.compile(r"(^|[_\s\-.])(id|uuid|guid|key|idx|index|record_?no|row_?num(ber)?)($|[_\s\-.])|_id$|[a-z]Id$|[a-z]ID$|^ID$",
                      re.IGNORECASE)
_ID_NAME_CS = re.compile(r"[a-z](Id|ID)$")
_UUID = re.compile(r"^[0-9a-fA-F]{8}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{12}$")


def _name_is_id(name: str) -> bool:
    n = str(name)
    return bool(re.search(r"(^|[_\s\-.])(id|uuid|guid|key|idx|index)($|[_\s\-.])", n, re.I)
                or re.search(r"_id$", n, re.I) or _ID_NAME_CS.search(n) or n.lower() in ("id", "uuid", "guid"))


def is_identifier(name: str, s: pd.Series, n_rows: int | None = None) -> tuple[bool, str]:
    """Return (is_identifier, reason)."""
    n_rows = len(s) if n_rows is None else n_rows
    non_null = s.dropna()
    if non_null.empty:
        return False, ""
    n_unique = int(non_null.nunique())
    ratio = n_unique / len(non_null)
    as_str = non_null.astype(str)
    if as_str.head(500).str.match(_UUID).mean() >= 0.9:
        return True, "values are UUID-shaped"
    if _name_is_id(name) and (ratio >= 0.5 or n_unique > 50):
        return True, f"name '{name}' follows an identifier pattern and {pct(100 * ratio)} of values are distinct"
    if ptypes.is_integer_dtype(non_null) and not ptypes.is_bool_dtype(non_null) and len(non_null) >= 20 and n_unique == len(non_null):
        d = np.diff(np.sort(non_null.to_numpy()))
        if len(d) and (d == 1).all():
            return True, "every value is unique and the values form a consecutive integer sequence"
    if (not ptypes.is_numeric_dtype(non_null) and len(non_null) >= 20 and ratio >= 0.95 and as_str.str.len().mean() <= 24
            and as_str.str.match(r"^[A-Za-z]{0,6}[-_#]?\d+[A-Za-z0-9\-_]*$").mean() >= 0.9):
        return True, f"{pct(100 * ratio)} of values are distinct short codes (e.g. '{as_str.iloc[0]}')"
    return False, ""


def _entropy(counts: pd.Series) -> float | None:
    if len(counts) < 2:
        return None
    p = counts.to_numpy(dtype=float) / counts.sum()
    return float(-(p * np.log(p)).sum() / math.log(len(p)))


def _task_for(s: pd.Series, kind: str) -> str | None:
    non_null = s.dropna()
    if kind == "numeric" and not ptypes.is_bool_dtype(s):
        n_unique = non_null.nunique()
        if n_unique <= 10 and (non_null.astype(float) == non_null.astype(float).round()).all():
            return "classification"
        return "regression"
    if kind == "categorical" or ptypes.is_bool_dtype(s):
        return "classification"
    return None


def evaluate_column(df: pd.DataFrame, column: str, kind: str, task: str | None = None) -> dict[str, Any]:
    """Assess one column as a target. ``task`` forces the task type (e.g. a user override)."""
    s = df[column]
    n_rows = len(df)
    non_null = s.dropna()
    n_unique = int(non_null.nunique())
    missing_pct = float(100 * s.isna().mean()) if n_rows else 100.0
    c: dict[str, Any] = {"column": column, "task": None, "verdict": "blocked", "score": None, "reasons": [], "warnings": [],
                         "blocking": [], "n_unique": n_unique, "missing_pct": missing_pct, "minority_class_size": None,
                         "entropy": None, "kind": kind, "outcome_hint": bool(OUTCOME_HINT.search(str(column)))}
    if non_null.empty:
        c["blocking"].append("every value is missing")
    elif n_unique < 2:
        c["blocking"].append("the column holds a single value, so there is nothing to predict")
    ident, why = is_identifier(column, s, n_rows)
    if ident:
        c["blocking"].append(f"looks like a record identifier ({why}); identifiers carry no generalisable outcome")
    if kind == "text":
        c["blocking"].append("free text (long or mostly unique strings) is not a supervised-learning target")
    if kind == "datetime":
        c["blocking"].append("dates are not supported as a target; derive a numeric or categorical outcome first")
    inferred = _task_for(s, kind) if not non_null.empty else None
    task = task or inferred
    c["task"] = task
    if task == "regression" and not (ptypes.is_numeric_dtype(s) and not ptypes.is_bool_dtype(s)):
        c["blocking"].append("regression needs a numeric target")
    if task == "classification" and not non_null.empty:
        counts = non_null.astype(str).value_counts()
        c["minority_class_size"] = int(counts.min())
        c["entropy"] = _entropy(counts)
        if len(counts) > MAX_CLASSES:
            c["blocking"].append(f"{len(counts):,} classes exceeds the limit of {MAX_CLASSES}; this is not a meaningful classification target")
        elif float(counts.median()) < 2:
            c["blocking"].append("most classes have a single row, so the model could not learn or be evaluated")
        elif len(counts) > HIGH_CARDINALITY:
            c["warnings"].append(f"high-cardinality target: {len(counts)} classes; many classes will have few examples")
        if not c["blocking"] and counts.min() < MIN_MINORITY:
            c["warnings"].append(f"the smallest class ('{counts.index[-1]}') has only {int(counts.min())} row(s)")
    if task is None and not c["blocking"]:
        c["blocking"].append("the column type does not support classification or regression")
    if missing_pct > MAX_MISSING_PCT:
        c["warnings"].append(f"{pct(missing_pct)} of target values are missing; those rows cannot be used")
    if n_rows < 50:
        c["warnings"].append(f"only {n_rows} rows; any model will have very uncertain performance")
    if c["blocking"]:
        return c

    # score
    hint = 1.0 if c["outcome_hint"] else 0.0
    if task == "classification":
        k = n_unique
        structure = 1.0 if k <= 10 else 0.6 if k <= HIGH_CARDINALITY else 0.2
        balance = c["entropy"] if c["entropy"] is not None else 0.0
        c["reasons"].append(f"{k} classes; smallest class has {c['minority_class_size']:,} rows")
    else:
        structure = 0.8
        balance = 1.0 if n_unique >= 10 else 0.5
        c["reasons"].append(f"numeric with {n_unique:,} distinct values")
    completeness = 1 - missing_pct / 100
    w = SCORE_WEIGHTS
    score = w["outcome_name"] * hint + w["structure"] * structure + w["completeness"] * completeness + w["balance"] * balance
    c["score"] = round(float(score), 3)
    if hint:
        c["reasons"].insert(0, f"the name '{column}' matches a common outcome variable")
    if hint and score >= 0.6 and not any("high-cardinality" in x for x in c["warnings"]):
        c["verdict"] = "recommended"
    elif c["warnings"] and any(("high-cardinality" in x) or ("missing" in x) for x in c["warnings"]):
        c["verdict"] = "discouraged"
    elif score >= 0.35:
        c["verdict"] = "possible"
    else:
        c["verdict"] = "discouraged"
    return c


_ORDER = {"recommended": 0, "possible": 1, "discouraged": 2, "blocked": 3}


def assess_targets(df: pd.DataFrame, kinds: dict[str, str] | None = None) -> dict[str, Any]:
    kinds = kinds or infer_kinds(df)
    cands = [evaluate_column(df, c, kinds.get(c, "categorical")) for c in df.columns]
    cands.sort(key=lambda c: (_ORDER[c["verdict"]], -(c["score"] or 0), str(c["column"])))
    rec = [c for c in cands if c["verdict"] == "recommended"]
    possible = [c for c in cands if c["verdict"] == "possible"]
    best = rec[0]["column"] if rec else None
    viable = bool(rec or possible)
    if rec:
        top = rec[0]
        msg = (f"'{top['column']}' is the most plausible {top['task']} target: " + "; ".join(top["reasons"]) + ".")
    elif possible:
        names = ", ".join(f"'{c['column']}'" for c in possible[:4])
        msg = ("No column has a clear outcome name. Columns such as " + names + " could be modelled if one of them is "
               "genuinely the outcome you care about; otherwise descriptive analysis and clustering are the better fit "
               "for this dataset.")
    else:
        msg = ("No column is a suitable supervised-learning target (identifiers, free text, constants or very "
               "high-cardinality columns only). Use descriptive analysis and clustering instead.")
    return {"candidates": cands, "supervised_viable": viable, "recommendation": msg, "best": best,
            "score_formula": SCORE_FORMULA}


def validate_target(df: pd.DataFrame, target: str | None, task: str | None = None,
                    kinds: dict[str, str] | None = None) -> dict[str, Any]:
    """Check a chosen target. Blocking problems cannot be overridden; warnings can, explicitly."""
    if not target or target not in df.columns:
        return {"ok": False, "blocking": ["no target column selected"], "warnings": [], "overridable": False,
                "discouraged": False, "candidate": None}
    kinds = kinds or infer_kinds(df[[target]])
    c = evaluate_column(df, target, kinds.get(target, "categorical"), task)
    discouraged = c["verdict"] == "discouraged"
    return {"ok": not c["blocking"], "blocking": c["blocking"], "warnings": c["warnings"],
            "overridable": not c["blocking"] and discouraged, "discouraged": discouraged, "candidate": c}
