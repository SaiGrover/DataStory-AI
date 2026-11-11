"""The shared analytical-results structure.

The dashboard, the insight engine, the dataset assistant and every report
format read from the dictionary built here. A number that is not in this
structure is not shown anywhere.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

import re

from core import cleaning, insights, profiler, quality_metrics
from core.data_loader import DatasetMeta, lineage

UNAVAILABLE = "Unavailable"


def build_results(
    df: pd.DataFrame, meta: DatasetMeta, *, original: pd.DataFrame | None = None,
    history: list[dict[str, Any]] | None = None, rules: list[dict[str, Any]] | None = None,
    exclude_from_completeness: list[str] | None = None, target: str | None = None,
    modeling: dict[str, Any] | None = None, importance: dict[str, Any] | None = None,
    clustering: dict[str, Any] | None = None, charts: list[dict[str, Any]] | None = None,
    overview: dict[str, Any] | None = None,
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
        "clustering": clustering,
        "charts": list(charts or []),
        "overview": overview,
        "lineage": lineage(meta, history),
        "data_version": f"v{len(history or [])}",
    }
    results["insights"] = insights.generate(results)
    results["evidence"] = evidence_register(results, list(df.columns))
    return results


def evidence_register(results: dict[str, Any], columns: list[str]) -> list[dict[str, Any]]:
    """One EvidenceRecord per finding: dataset version, computation, columns and limitations.

    Findings get stable identifiers (F1, F2, …) that reports and the assistant cite.
    """
    version = results["data_version"]
    ds_id = results["dataset"].get("dataset_id")
    out = []
    by_len = sorted(columns, key=len, reverse=True)
    for i, ins in enumerate(results["insights"], 1):
        ins["id"] = f"F{i}"
        ins["dataset_version"] = version
        text = " ".join([ins["title"], ins["observation"], *map(str, ins["evidence"])])
        cols = [c for c in by_len if len(c) > 1 and re.search(r"(?<![\w])" + re.escape(c) + r"(?![\w])", text)]
        out.append({
            "id": ins["id"], "finding": ins["title"], "category": ins["category"], "dataset_id": ds_id,
            "dataset_version": version, "computation": ins["source"] or "Computed from the active dataset",
            "columns": cols[:8], "evidence": ins["evidence"], "limitations": ins["limitation"],
            "generated_at": results["generated_at"],
        })
    return out


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
