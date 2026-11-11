"""Data-quality dimensions with explicit definitions.

A dimension is reported only when it can actually be computed. Validity and
consistency require user-configured rules; without rules they are
"Not evaluated", never estimated.
"""
from __future__ import annotations

import operator
import re
from typing import Any

import pandas as pd
from core.formatting import pct as _pct

NOT_EVALUATED = "Not evaluated — validation rules not configured."

DEFINITIONS = {
    "completeness": (
        "Non-missing cells ÷ all cells, across the included columns. Columns you exclude "
        "(for example free-text notes that are legitimately optional) are left out of both "
        "numerator and denominator and are listed next to the score."
    ),
    "uniqueness": (
        "Rows that are not an exact repeat of an earlier row ÷ all rows. The first occurrence "
        "of a repeated row counts as unique; every later copy counts as a duplicate. All columns "
        "are compared."
    ),
    "validity": (
        "Values that satisfy the configured single-column rules (numeric range, allowed values, "
        "parseable type) ÷ non-missing values those rules apply to. Missing values are counted "
        "under completeness, not here."
    ),
    "consistency": (
        "Records that satisfy the configured cross-field or format rules ÷ records where every "
        "column involved in the rule is present."
    ),
}

VALIDITY_TYPES = {
    "range": "Numeric range",
    "allowed": "Allowed values",
    "numeric": "Parses as a number",
    "datetime": "Parses as a date",
}
CONSISTENCY_TYPES = {
    "compare": "Cross-field comparison",
    "format": "Format pattern (regex)",
    "case": "Consistent letter case",
}
_OPS = {"<=": operator.le, "<": operator.lt, ">=": operator.ge, ">": operator.gt, "==": operator.eq, "!=": operator.ne}

HEALTH_WEIGHTS = {"completeness": 0.4, "uniqueness": 0.2, "validity": 0.2, "consistency": 0.2}
HEALTH_FORMULA = (
    "Weighted average of the evaluated dimensions (completeness 40%, uniqueness 20%, validity 20%, "
    "consistency 20%). Weights of dimensions that were not evaluated are redistributed "
    "proportionally over the rest. This is a DataStory AI convenience summary with arbitrary "
    "weights, not a standardised or validated statistical measure, and it says nothing about "
    "whether the data is fit for a particular purpose."
)


def completeness(df: pd.DataFrame, exclude: list[str] | None = None) -> dict[str, Any]:
    exclude = [c for c in (exclude or []) if c in df.columns]
    sub = df.drop(columns=exclude)
    cells = sub.shape[0] * sub.shape[1]
    if cells == 0:
        return {"value": None, "status": "unavailable", "message": "Unavailable — no cells to evaluate.",
                "excluded_columns": exclude}
    missing = int(sub.isna().sum().sum())
    return {
        "value": 100 * (cells - missing) / cells, "status": "evaluated",
        "passed": cells - missing, "applicable": cells, "excluded_columns": exclude,
        "definition": DEFINITIONS["completeness"],
    }


def uniqueness(df: pd.DataFrame) -> dict[str, Any]:
    n = len(df)
    if n == 0:
        return {"value": None, "status": "unavailable", "message": "Unavailable — the dataset has no rows."}
    dups = int(df.duplicated().sum())
    return {"value": 100 * (n - dups) / n, "status": "evaluated", "passed": n - dups, "applicable": n,
            "definition": DEFINITIONS["uniqueness"]}


def describe_rule(rule: dict[str, Any]) -> str:
    t, c = rule["type"], rule.get("column")
    if t == "range":
        lo, hi = rule.get("min"), rule.get("max")
        if lo is not None and hi is not None:
            return f"{c} between {lo:g} and {hi:g}"
        return f"{c} ≥ {lo:g}" if lo is not None else f"{c} ≤ {hi:g}"
    if t == "allowed":
        return f"{c} in {{{', '.join(map(str, rule['values']))}}}"
    if t == "numeric":
        return f"{c} parses as a number"
    if t == "datetime":
        return f"{c} parses as a date"
    if t == "compare":
        return f"{c} {rule['op']} {rule['other']}"
    if t == "format":
        return f"{c} matches /{rule['pattern']}/"
    if t == "case":
        return f"{c} uses one spelling per value regardless of letter case"
    return str(rule)


def evaluate_rule(df: pd.DataFrame, rule: dict[str, Any]) -> dict[str, Any]:
    """Return passed / applicable counts for one rule, or an error entry."""
    res = {"rule": describe_rule(rule), "type": rule["type"], "dimension": rule["dimension"]}
    cols = [rule.get("column")] + ([rule["other"]] if rule["type"] == "compare" else [])
    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        return {**res, "error": f"Column(s) no longer in the dataset: {', '.join(missing_cols)}"}
    s = df[rule["column"]]
    present = s.notna()
    try:
        t = rule["type"]
        if t == "range":
            x = pd.to_numeric(s, errors="coerce")
            ok = x.notna()
            if rule.get("min") is not None:
                ok &= x >= rule["min"]
            if rule.get("max") is not None:
                ok &= x <= rule["max"]
        elif t == "allowed":
            ok = s.astype(str).isin([str(v) for v in rule["values"]])
        elif t == "numeric":
            ok = pd.to_numeric(s, errors="coerce").notna()
        elif t == "datetime":
            ok = pd.to_datetime(s, errors="coerce", format="mixed").notna()
        elif t == "compare":
            o = df[rule["other"]]
            present &= o.notna()
            a, b = pd.to_numeric(s, errors="coerce"), pd.to_numeric(o, errors="coerce")
            if a[present].isna().any() or b[present].isna().any():  # not numeric: try dates
                a, b = pd.to_datetime(s, errors="coerce", format="mixed"), pd.to_datetime(o, errors="coerce", format="mixed")
            ok = _OPS[rule["op"]](a, b) & a.notna() & b.notna()
        elif t == "format":
            pat = re.compile(rule["pattern"])
            ok = s.astype(str).map(lambda v: bool(pat.fullmatch(v)))
        elif t == "case":
            as_str = s.astype(str).str.strip()
            spellings = as_str[present].groupby(as_str[present].str.lower()).nunique()
            ambiguous = set(spellings[spellings > 1].index)
            ok = ~as_str.str.lower().isin(ambiguous)
        else:
            return {**res, "error": f"Unknown rule type '{t}'."}
    except re.error as exc:
        return {**res, "error": f"Invalid regular expression: {exc}"}
    except Exception as exc:  # defensive: a bad rule must never crash the page
        return {**res, "error": f"Rule could not be evaluated: {exc}"}
    applicable = int(present.sum())
    passed = int((ok & present).sum())
    return {**res, "passed": passed, "applicable": applicable,
            "value": 100 * passed / applicable if applicable else None}


def _rule_dimension(df: pd.DataFrame, rules: list[dict[str, Any]], dimension: str) -> dict[str, Any]:
    mine = [r for r in rules if r["dimension"] == dimension]
    if not mine:
        return {"value": None, "status": "not_evaluated", "message": NOT_EVALUATED, "rules": [],
                "definition": DEFINITIONS[dimension]}
    results = [evaluate_rule(df, r) for r in mine]
    good = [r for r in results if "error" not in r and r["applicable"] > 0]
    if not good:
        return {"value": None, "status": "not_evaluated", "rules": results, "definition": DEFINITIONS[dimension],
                "message": "Not evaluated — the configured rules had no applicable values."}
    passed, applicable = sum(r["passed"] for r in good), sum(r["applicable"] for r in good)
    return {"value": 100 * passed / applicable, "status": "evaluated", "passed": passed, "applicable": applicable,
            "rules": results, "definition": DEFINITIONS[dimension]}


def health_score(dimensions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    included = {k: d["value"] for k, d in dimensions.items() if d.get("value") is not None}
    if not included:
        return {"value": None, "status": "unavailable", "message": "Unavailable — no quality dimension could be evaluated."}
    total_w = sum(HEALTH_WEIGHTS[k] for k in included)
    weights = {k: HEALTH_WEIGHTS[k] / total_w for k in included}
    return {
        "value": sum(included[k] * weights[k] for k in included), "status": "evaluated",
        "included": list(included), "excluded": [k for k in dimensions if k not in included],
        "weights_used": weights, "formula": HEALTH_FORMULA,
    }


def assess_quality(df: pd.DataFrame, rules: list[dict[str, Any]] | None = None,
                   exclude_from_completeness: list[str] | None = None) -> dict[str, Any]:
    rules = rules or []
    dims = {
        "completeness": completeness(df, exclude_from_completeness),
        "uniqueness": uniqueness(df),
        "validity": _rule_dimension(df, rules, "validity"),
        "consistency": _rule_dimension(df, rules, "consistency"),
    }
    return {"dimensions": dims, "health": health_score(dims), "definitions": DEFINITIONS}


def format_dimension(d: dict[str, Any]) -> str:
    """Single display string used by the dashboard, the assistant and reports alike."""
    return f"{_pct(d['value'], 1)}" if d.get("value") is not None else d.get("message", "Unavailable")
