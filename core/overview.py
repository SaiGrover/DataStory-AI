"""Dataset-aware overview: what the data appears to be, what it supports, what to watch out for.

Domain inference is keyword evidence over column names (and a few telling
values). It reports the matched evidence and falls back to "General / unknown"
when the evidence is weak, rather than inventing a domain. The user can
override it.
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

from core import charts, targets
from core.formatting import pct

DOMAINS: dict[str, dict[str, Any]] = {
    "Retail & sales": {"words": ["sales", "revenue", "order", "product", "quantity", "price", "discount", "store", "sku", "invoice",
                                 "customer", "profit", "category", "region", "shipping", "unit"]},
    "Customer analytics": {"words": ["churn", "tenure", "customer", "contract", "subscription", "monthlycharges", "totalcharges",
                                     "plan", "usage", "segment", "loyalty", "retention"]},
    "Human resources": {"words": ["employee", "attrition", "salary", "department", "jobrole", "job", "hire", "yearsatcompany",
                                  "performance", "overtime", "manager", "monthlyincome", "education"]},
    "Finance": {"words": ["loan", "credit", "balance", "transaction", "account", "default", "interest", "income", "debt", "fraud",
                          "amount", "payment", "bank", "stock", "close", "open", "volume"]},
    "Marketing": {"words": ["campaign", "click", "impression", "conversion", "ctr", "channel", "spend", "lead", "email", "ad",
                            "utm", "engagement", "budget"]},
    "Healthcare": {"words": ["patient", "diagnosis", "disease", "bmi", "blood", "glucose", "insulin", "cholesterol", "age",
                             "symptom", "hospital", "treatment", "mean_radius", "malignant", "progression"]},
    "Education": {"words": ["student", "grade", "school", "exam", "score", "g1", "g2", "g3", "absences", "studytime", "course",
                            "teacher", "failures"]},
    "Entertainment & media": {"words": ["title", "director", "cast", "genre", "listed_in", "rating", "release_year", "duration",
                                        "show", "movie", "episode", "season", "views", "artist", "album"]},
    "Real estate": {"words": ["bedroom", "bedrooms", "bathroom", "bathrooms", "sqft", "house", "property", "lot", "zipcode",
                              "rooms", "yearbuilt", "medv", "housing"]},
    "Transportation & mobility": {"words": ["trip", "ride", "station", "distance", "passenger", "fare", "route", "vehicle",
                                            "bike", "rental", "cnt", "weathersit", "pickup", "dropoff", "flight", "delay"]},
}
GENERAL = "General / unknown"
GENERIC = {"price", "region", "amount", "customer", "age", "category", "rating", "income", "score", "type", "title", "unit",
           "volume", "open", "close", "segment", "plan", "usage", "job", "education", "course", "views"}  # weak evidence
MIN_EVIDENCE = 2  # distinct keyword matches needed before a domain is suggested


def _tokens(name: str) -> set[str]:
    raw = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    parts = set(re.split(r"[^a-z0-9]+", raw.lower())) - {""}
    parts.add(re.sub(r"[^a-z0-9]", "", name.lower()))
    return parts


def infer_domain(df: pd.DataFrame) -> dict[str, Any]:
    """{"domain", "confidence" (low/medium/high), "evidence": [...], "scores": {...}} — uncertainty-aware."""
    toks: dict[str, set[str]] = {c: _tokens(c) for c in df.columns}
    scores: dict[str, list[str]] = {}
    for dom, spec in DOMAINS.items():
        hits = []
        for c, ts in toks.items():
            for w in spec["words"]:
                if w in ts:
                    hits.append(f"column '{c}' matches '{w}'")
                    break
        scores[dom] = hits
    # a few value-level hints
    for c in df.columns:
        if df[c].dtype == object:
            vals = set(df[c].dropna().astype(str).head(500).str.lower())
            if {"movie", "tv show"} <= vals:
                scores["Entertainment & media"].append(f"'{c}' contains 'Movie' and 'TV Show'")
    def weight(hits: list[str]) -> float:
        return sum(0.5 if any(f"matches '{g}'" in h for g in GENERIC) else 1.0 for h in hits)

    ranked = sorted(scores.items(), key=lambda kv: -weight(kv[1]))
    best, ev = ranked[0]
    runner = weight(ranked[1][1]) if len(ranked) > 1 else 0
    if len(ev) < MIN_EVIDENCE:
        return {"domain": GENERAL, "confidence": "low", "evidence": ev,
                "scores": {k: len(v) for k, v in scores.items()},
                "note": "Too few domain-specific column names to suggest a domain."}
    margin = weight(ev) - runner
    conf = "high" if len(ev) >= 4 and margin >= 2 else ("medium" if margin >= 1 else "low")
    note = "" if conf != "low" else f"Close to '{ranked[1][0]}'; treat the suggested domain as a guess."
    return {"domain": best, "confidence": conf, "evidence": ev[:6], "scores": {k: len(v) for k, v in scores.items()}, "note": note}


def build_overview(df: pd.DataFrame, profile: dict[str, Any], quality: dict[str, Any], target: str | None = None,
                   domain_override: str | None = None) -> dict[str, Any]:
    kinds = profile["kinds"]
    details = {c["name"]: c for c in profile["column_details"]}
    dom = infer_domain(df)
    domain = domain_override or dom["domain"]
    tg = targets.assess_targets(df, kinds)
    ids = profile["id_candidates"]
    dates = [c for c, k in kinds.items() if k == "datetime"]
    text = [c for c, k in kinds.items() if k == "text" and c not in ids]
    high_card = [c for c in usable_high_cardinality(profile)]
    numeric = [c for c, k in kinds.items() if k == "numeric" and c not in ids]
    cats = charts.usable_categoricals(df, profile)

    understands = [
        f"{profile['rows']:,} rows × {profile['columns']:,} columns: {profile['kind_counts']['numeric']} numeric, "
        f"{profile['kind_counts']['categorical']} categorical, {profile['kind_counts']['datetime']} date and {profile['kind_counts']['text']} free-text column(s).",
        (f"Likely domain: {domain} ({'set by you' if domain_override else dom['confidence'] + ' confidence'})." if domain != GENERAL
         else "The domain is unclear from the column names, so DataStory treats this as a general tabular dataset."),
    ]
    if ids:
        understands.append(f"Identifier-like column(s): {', '.join(ids)} — one value per record; excluded from charts, targets and models by default.")
    if dates:
        understands.append(f"Date column(s): {', '.join(dates)} — enables trends over time.")
    if text:
        understands.append(f"Free-text column(s): {', '.join(text[:5])} — summarised, not modelled.")
    best = next((c for c in tg["candidates"] if c["verdict"] == "recommended"), None)
    if best:
        understands.append(f"Plausible supervised target: {best['column']} ({best['task']}).")
    elif tg["supervised_viable"]:
        poss = [c["column"] for c in tg["candidates"] if c["verdict"] == "possible"][:3]
        understands.append(f"No column is an obvious outcome to predict; possible targets: {', '.join(poss)}.")
    else:
        understands.append("No suitable supervised-learning target was found; descriptive analysis and clustering fit better.")

    # risks
    risks: list[dict[str, str]] = []
    if profile["missing_cells"]:
        worst = sorted([c for c in profile["column_details"] if c["missing"]], key=lambda c: -c["missing"])[:3]
        risks.append({"level": "warning" if (profile["missing_pct"] or 0) > 5 else "info", "title": "Missing values",
                      "detail": f"{profile['missing_cells']:,} missing cells ({pct(profile['missing_pct'], 2)} of all cells); most in "
                                + ", ".join(f"{c['name']} ({pct(c['missing_pct'])})" for c in worst) + "."})
    if profile["duplicate_rows"]:
        risks.append({"level": "warning", "title": "Duplicate rows",
                      "detail": f"{profile['duplicate_rows']:,} exact duplicate rows ({pct(profile['duplicate_pct'])})."})
    for c in high_card[:3]:
        risks.append({"level": "info", "title": f"High cardinality: {c}",
                      "detail": f"{details[c]['unique']:,} distinct values — charts show the top values and models exclude it."})
    if profile["constant_columns"]:
        risks.append({"level": "info", "title": "Constant columns", "detail": ", ".join(profile["constant_columns"]) + " hold a single value."})
    if target and profile.get("target") and profile["target"].get("imbalance_ratio") and profile["target"]["imbalance_ratio"] >= 3:
        t = profile["target"]
        risks.append({"level": "warning", "title": "Class imbalance",
                      "detail": f"{target}: the largest class is {t['imbalance_ratio']:.1f}× the smallest; prefer F1 / PR-AUC over accuracy."})
    for c in profile["near_constant_columns"][:3]:
        risks.append({"level": "info", "title": f"Near-constant: {c}", "detail": f"One value covers {pct(details[c]['top_share_pct'])} of rows."})
    if profile["outlier_columns"]:
        o = profile["outlier_columns"][0]
        risks.append({"level": "info", "title": "Outliers", "detail": f"{o['column']} has {o['outliers']:,} values outside the Tukey fences ({pct(o['outlier_pct'])})."})
    for c in tg["candidates"]:
        if c["verdict"] == "blocked" and c["column"] in ids:
            why = (c.get("blocking") or []) + (c.get("reasons") or []) + (c.get("warnings") or [])
            risks.append({"level": "info", "title": f"Not a valid target: {c['column']}",
                          "detail": why[0] if why else "Identifier-like column (one value per record)."})
            break

    # recommended analyses → each opens a page (with preset where feasible)
    recs: list[dict[str, Any]] = []
    rec_charts = charts.recommend(df, profile, target=target, domain=domain, limit=6)
    recs.append({"title": "Explore recommended charts", "detail": f"{len(rec_charts)} charts chosen for this dataset's columns.",
                 "page": "views/eda.py", "preset": {"eda_tab": "Recommended"}, "icon": "insights"})
    if profile["missing_cells"] or profile["duplicate_rows"]:
        recs.append({"title": "Review data quality", "detail": "Completeness, uniqueness and optional validation rules.",
                     "page": "views/quality.py", "preset": {}, "icon": "health_and_safety"})
    if best:
        recs.append({"title": f"Predict {best['column']}", "detail": f"Train {best['task']} baselines with leakage-safe pipelines.",
                     "page": "views/ml.py", "preset": {"target": best["column"]}, "icon": "model_training"})
    if len(numeric) + len(cats) >= 2 and profile["rows"] >= 30:
        recs.append({"title": "Find segments", "detail": "Cluster rows on numeric and low-cardinality features — no target needed.",
                     "page": "views/clustering.py", "preset": {}, "icon": "bubble_chart"})
    if dates:
        recs.append({"title": f"Trend over {dates[0]}", "detail": "Records per month, with optional measures over time.",
                     "page": "views/eda.py", "preset": {"eda_tab": "Recommended"}, "icon": "timeline"})
    recs.append({"title": "Ask the copilot", "detail": "Ask questions or request charts in plain language.",
                 "page": "views/assistant.py", "preset": {}, "icon": "forum"})

    steps = []
    if profile["missing_cells"] or profile["duplicate_rows"] or profile["outlier_columns"]:
        steps.append("Clean: handle duplicates and missing values (preview shows the exact impact before you apply it).")
    steps.append("Explore: open the recommended charts and note anything surprising.")
    steps.append(f"Model: train baselines for {best['column']}." if best else
                 "Segment: try clustering, since no clear prediction target exists.")
    steps.append("Report: assemble findings with their evidence and export a report or notebook.")

    return {"domain": dom, "domain_used": domain, "targets": tg, "understands": understands, "risks": risks,
            "recommendations": recs, "chart_recommendations": rec_charts, "next_steps": steps,
            "clustering_suitable": len(numeric) + len(cats) >= 2 and profile["rows"] >= 30,
            "health": quality.get("health", {})}


def usable_high_cardinality(profile: dict[str, Any]) -> list[str]:
    return [c["name"] for c in profile["column_details"]
            if c["kind"] == "categorical" and c["unique"] > 50 and c["name"] not in profile["id_candidates"]]
