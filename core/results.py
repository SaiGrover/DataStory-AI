"""The shared analytical-results structure.

The dashboard, the insight engine, the dataset assistant and every report
format read from the dictionary built here. A number that is not in this
structure is not shown anywhere.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from core import cleaning, insights, profiler, quality_metrics
from core.data_loader import DatasetMeta

UNAVAILABLE = "Unavailable"


def build_results(
    df: pd.DataFrame, meta: DatasetMeta, *, original: pd.DataFrame | None = None,
    history: list[dict[str, Any]] | None = None, rules: list[dict[str, Any]] | None = None,
    exclude_from_completeness: list[str] | None = None, target: str | None = None,
    modeling: dict[str, Any] | None = None, importance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target = target if target in df.columns else None
    profile = profiler.profile_dataset(df, target)
    quality = quality_metrics.assess_quality(df, rules, exclude_from_completeness)
    exclude = list(profile["id_candidates"])
    pairs = profiler.correlation_pairs(df, profile["kinds"], exclude=exclude)
    results: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "dataset": meta.to_dict(),
        "profile": profile,
        "quality": quality,
        "rules": list(rules or []),
        "correlations": {"method": "Pearson, pairwise complete observations, identifier-like columns excluded",
                         "pairs": pairs[:25], "n_pairs": len(pairs)},
        "cleaning": {
            "history": list(history or []),
            "original": cleaning.snapshot(original if original is not None else df),
            "current": cleaning.snapshot(df),
            "suggestions": cleaning.suggest_steps(profile),
        },
        "target": target,
        "modeling": modeling,
        "importance": importance,
    }
    results["insights"] = insights.generate(results)
    return results


def fmt(value: Any, digits: int = 4, pct: bool = False) -> str:
    """Uniform formatting; missing values are always shown as unavailable, never as a number."""
    if value is None or (isinstance(value, float) and value != value):
        return UNAVAILABLE
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.{digits}f}%" if pct else f"{value:,.{digits}f}"
    return str(value)
