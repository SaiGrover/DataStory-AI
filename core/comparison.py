"""Original vs cleaned comparison.

Differences are computed per column on the original (v0) and the active version.
Cell-level changes come from core.impact, which aligns rows by their stable
index, so rows removed by cleaning are reported as removals — never as changed
values on the rows that would otherwise shift up.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core import eda, quality_metrics
from core.impact import compute_impact
from core.profiler import count_outliers, infer_kinds


def column_summary(df: pd.DataFrame, kinds: dict[str, str]) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        s = df[c]
        row = {"column": c, "dtype": str(s.dtype), "kind": kinds.get(c), "missing": int(s.isna().sum()),
               "distinct": int(s.nunique(dropna=True))}
        if kinds.get(c) == "numeric":
            x = pd.to_numeric(s, errors="coerce")
            row.update(mean=x.mean(), std=x.std(), min=x.min(), median=x.median(), max=x.max(), outliers=count_outliers(s))
        rows.append(row)
    return pd.DataFrame(rows).set_index("column")


def compare(original: pd.DataFrame, current: pd.DataFrame) -> dict[str, Any]:
    ko, kc = infer_kinds(original), infer_kinds(current)
    so, sc = column_summary(original, ko), column_summary(current, kc)
    common = [c for c in original.columns if c in current.columns]
    stats = ["missing", "distinct", "mean", "std", "min", "median", "max", "outliers"]
    diff_rows = []
    for c in common:
        r: dict[str, Any] = {"column": c, "dtype before": so.loc[c, "dtype"], "dtype after": sc.loc[c, "dtype"]}
        for k in stats:
            a = so.loc[c, k] if k in so.columns else np.nan
            b = sc.loc[c, k] if k in sc.columns else np.nan
            r[f"{k} before"], r[f"{k} after"] = a, b
        diff_rows.append(r)
    impact = compute_impact(original, current)
    q_o = quality_metrics.assess_quality(original)
    q_c = quality_metrics.assess_quality(current)
    overview = pd.DataFrame([
        {"measure": "Rows", "original": len(original), "current": len(current)},
        {"measure": "Columns", "original": original.shape[1], "current": current.shape[1]},
        {"measure": "Missing cells", "original": int(original.isna().sum().sum()), "current": int(current.isna().sum().sum())},
        {"measure": "Duplicate rows", "original": int(original.duplicated().sum()), "current": int(current.duplicated().sum())},
        {"measure": "Completeness %", "original": q_o["dimensions"]["completeness"]["value"], "current": q_c["dimensions"]["completeness"]["value"]},
        {"measure": "Uniqueness %", "original": q_o["dimensions"]["uniqueness"]["value"], "current": q_c["dimensions"]["uniqueness"]["value"]},
    ])
    overview["change"] = overview["current"] - overview["original"]
    return {"overview": overview, "columns": pd.DataFrame(diff_rows), "impact": impact,
            "removed_columns": [c for c in original.columns if c not in current.columns],
            "added_columns": [c for c in current.columns if c not in original.columns],
            "kinds_before": ko, "kinds_after": kc}


def distribution_overlay(original: pd.DataFrame, current: pd.DataFrame, col: str, bins: int = 40) -> go.Figure | None:
    a = pd.to_numeric(original[col], errors="coerce").dropna() if col in original else pd.Series(dtype=float)
    b = pd.to_numeric(current[col], errors="coerce").dropna() if col in current else pd.Series(dtype=float)
    if a.empty and b.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=a, nbinsx=bins, name=f"Original (n={len(a):,})", opacity=0.55, marker_color="#B9C2B0"))
    fig.add_trace(go.Histogram(x=b, nbinsx=bins, name=f"Current (n={len(b):,})", opacity=0.65, marker_color=eda.SAGE))
    fig.update_layout(barmode="overlay", xaxis_title=col, yaxis_title="Rows")
    return eda.style(fig, f"Distribution of {col}: original vs current", height=380)


def category_shift(original: pd.DataFrame, current: pd.DataFrame, col: str, top_n: int = 15) -> tuple[go.Figure, pd.DataFrame]:
    a = original[col].astype("object").where(original[col].notna(), "(missing)").astype(str).value_counts()
    b = current[col].astype("object").where(current[col].notna(), "(missing)").astype(str).value_counts()
    t = pd.DataFrame({"original": a, "current": b}).fillna(0).astype(int)
    t = t.loc[t.max(axis=1).sort_values(ascending=False).index[:top_n]]
    fig = go.Figure([go.Bar(y=t.index, x=t["original"], name="Original", orientation="h", marker_color="#B9C2B0"),
                     go.Bar(y=t.index, x=t["current"], name="Current", orientation="h", marker_color=eda.SAGE)])
    fig.update_layout(barmode="group", xaxis_title="Rows")
    fig.update_yaxes(autorange="reversed", type="category", automargin=True)
    return eda.style(fig, f"Top values of {col}: original vs current", height=max(320, 30 * len(t) + 110)), t.reset_index(names=col)


def missing_comparison(original: pd.DataFrame, current: pd.DataFrame) -> go.Figure | None:
    cols = [c for c in original.columns if original[c].isna().any() or (c in current and current[c].isna().any())]
    if not cols:
        return None
    a = [int(original[c].isna().sum()) for c in cols]
    b = [int(current[c].isna().sum()) if c in current else 0 for c in cols]
    fig = go.Figure([go.Bar(x=cols, y=a, name="Original", marker_color="#C98B3A"),
                     go.Bar(x=cols, y=b, name="Current", marker_color=eda.SAGE)])
    fig.update_layout(barmode="group", yaxis_title="Missing cells")
    fig.update_xaxes(type="category", tickangle=-30 if len(cols) > 6 else 0, automargin=True)
    return eda.style(fig, "Missing values per column: original vs current", height=380)
