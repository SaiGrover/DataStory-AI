"""Tool-based analytical copilot.

The copilot never executes generated code. A question is mapped to one of a
fixed set of analytical tools, each with a JSON schema; arguments are validated
(types, allowed values, existing columns) and the tool runs deterministic
pandas code on the active session's data. The answer text, tables and charts
come from the tool's output.

Routing: when a Groq key is configured the LLM may *choose* a tool and its
arguments (function calling); otherwise — or when the LLM fails or proposes an
invalid call — a deterministic keyword router picks the tool. LLM-written prose
is checked against the tool's verified output by the same number guard used
elsewhere; unverifiable drafts are discarded.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ai import assistant, llm
from core import charts as ch
from core import eda, nl_chart
from core.formatting import metric, num, pct

UNSUPPORTED = ("I can only run DataStory's built-in analyses (summaries, statistics, grouping, filtering, correlations, charts, "
               "model and quality results). I don't execute code, run SQL or shell commands, or access files or the network.")
_CODE = re.compile(r"(\b(exec|eval|subprocess|shell|bash|powershell|curl|wget)\b|\bimport\s+\w+|\bos\.\w+|__\w+__|\bopen\(|"
                   r"\brequests\.|python code|run (this|the following)( \w+)? code|\bselect\s+.+\s+from\s+\w+|\bdrop\s+table\b|"
                   r"\bdelete\s+from\b|\brm\s+-\w+|\bsql\b)", re.I)


@dataclass
class ToolResult:
    text: str
    tool: str
    args: dict[str, Any]
    table: pd.DataFrame | None = None
    spec: dict[str, Any] | None = None
    figure: Any = None
    facts: str = ""
    sources: list[str] = field(default_factory=list)
    navigate: tuple[str, str] | None = None  # (page path, label)
    followups: list[str] = field(default_factory=list)
    ok: bool = True


@dataclass
class Ctx:
    df: pd.DataFrame
    results: dict[str, Any]
    kinds: dict[str, str]
    previous_spec: dict[str, Any] | None = None


# ----------------------------------------------------------------- tools
def _col(ctx: Ctx, name: Any, kinds: tuple[str, ...] | None = None) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError("A column name is required.")
    if name not in ctx.df.columns:
        hit = nl_chart.resolve_columns(name, list(ctx.df.columns))
        if not hit:
            raise ValueError(f"There is no column called '{name}'. Columns include: {', '.join(list(ctx.df.columns)[:10])}.")
        name = hit[0]
    if kinds and ctx.kinds[name] not in kinds:
        raise ValueError(f"'{name}' is {ctx.kinds[name]}; this needs a {' or '.join(kinds)} column.")
    return name


def t_dataset_summary(ctx: Ctx) -> ToolResult:
    p, ds = ctx.results["profile"], ctx.results["dataset"]
    kc = p["kind_counts"]
    text = (f"**{ds['name']}** has {p['rows']:,} rows and {p['columns']:,} columns ({kc['numeric']} numeric, {kc['categorical']} categorical, "
            f"{kc['datetime']} date, {kc['text']} text). Provenance: {ds.get('provenance_label', ds.get('provenance'))}. "
            f"Missing cells: {p['missing_cells']:,} ({pct(p['missing_pct'], 2)}); duplicate rows: {p['duplicate_rows']:,}. "
            f"Active version: {ctx.results['data_version']}.")
    table = pd.DataFrame([{"column": c["name"], "type": c["kind"], "missing": c["missing"], "distinct": c["unique"]} for c in p["column_details"]])
    return ToolResult(text, "dataset_summary", {}, table=table, facts=text + " " + table.to_string(), sources=["Profile of the active dataset"],
                      followups=["What are the biggest issues in this dataset?", "Show missing values by column"])


def t_quality_issues(ctx: Ctx) -> ToolResult:
    ov_risks = []
    p, q = ctx.results["profile"], ctx.results["quality"]
    if p["missing_cells"]:
        worst = sorted([c for c in p["column_details"] if c["missing"]], key=lambda c: -c["missing"])[:5]
        ov_risks.append(("Missing values", f"{p['missing_cells']:,} cells ({pct(p['missing_pct'], 2)}), mostly in "
                         + ", ".join(f"{c['name']} ({pct(c['missing_pct'])})" for c in worst)))
    if p["duplicate_rows"]:
        ov_risks.append(("Duplicates", f"{p['duplicate_rows']:,} exact duplicate rows ({pct(p['duplicate_pct'])})"))
    if p["id_candidates"]:
        ov_risks.append(("Identifier-like columns", ", ".join(p["id_candidates"]) + " — exclude from charts and models"))
    if p["constant_columns"]:
        ov_risks.append(("Constant columns", ", ".join(p["constant_columns"])))
    if p["near_constant_columns"]:
        ov_risks.append(("Near-constant columns", ", ".join(p["near_constant_columns"])))
    for o in p["outlier_columns"][:3]:
        ov_risks.append((f"Outliers in {o['column']}", f"{o['outliers']:,} values outside the Tukey fences ({pct(o['outlier_pct'])})"))
    t = p.get("target")
    if t and t.get("imbalance_ratio") and t["imbalance_ratio"] >= 3:
        ov_risks.append(("Class imbalance", f"{t['name']}: largest class {t['imbalance_ratio']:.1f}× the smallest"))
    h = q["health"]
    head = f"Health score: {h['value']:.0f}/100. " if h.get("value") is not None else ""
    if not ov_risks:
        text = head + "No missing values, duplicates, identifier issues or outlier concentrations were found."
    else:
        text = head + "The main issues, from the profile of the active dataset:\n" + "\n".join(f"- **{a}** — {b}" for a, b in ov_risks)
    return ToolResult(text, "quality_issues", {}, table=pd.DataFrame(ov_risks, columns=["issue", "evidence"]) if ov_risks else None,
                      facts=text, sources=["Profiler", "Quality assessment"], navigate=("views/cleaning.py", "Open Clean"),
                      followups=["What cleaning steps should I perform?", "Show missing values by column"])


def t_column_stats(ctx: Ctx, column: str) -> ToolResult:
    c = _col(ctx, column)
    d = next(x for x in ctx.results["profile"]["column_details"] if x["name"] == c)
    rows = {k: d.get(k) for k in ("kind", "dtype", "missing", "missing_pct", "unique", "mean", "std", "min", "q1", "median", "q3", "max",
                                  "skew", "outliers", "top_value", "top_count", "top_share_pct") if d.get(k) is not None}
    if d["kind"] == "numeric" and d.get("mean") is not None:
        text = (f"**{c}**: {d['unique']:,} distinct values, {d['missing']:,} missing. Mean {num(d['mean'])}, median {num(d['median'])}, "
                f"standard deviation {num(d['std'])}, range {num(d['min'])} to {num(d['max'])}.")
        spec = ch.ChartSpec("histogram", x=c).to_dict()
    else:
        text = (f"**{c}** ({d['kind']}): {d['unique']:,} distinct values, {d['missing']:,} missing. "
                + (f"Most frequent: '{d['top_value']}' ({d['top_count']:,} rows, {pct(d['top_share_pct'])} of non-missing)." if d.get("top_value") else ""))
        spec = None if ch.chart_suitability(ctx.df, c, ctx.results["profile"]) else ch.ChartSpec("bar", x=c, top_n=15).to_dict()
    table = pd.DataFrame({"statistic": list(rows), "value": [str(v) for v in rows.values()]})
    return ToolResult(text, "column_stats", {"column": c}, table=table, spec=spec, facts=text + " " + table.to_string(),
                      sources=[f"Profile of column {c}"])


def t_missing_by_column(ctx: Ctx) -> ToolResult:
    p = ctx.results["profile"]
    if not p["missing_cells"]:
        return ToolResult("No column has missing values.", "missing_by_column", {}, facts="no missing values", sources=["Profile"])
    out = ch.render(ctx.df, ch.ChartSpec("missing"), ctx.kinds)
    return ToolResult(out.interpretation, "missing_by_column", {}, table=out.table, spec=ch.ChartSpec("missing").to_dict(),
                      facts=out.interpretation + " " + out.table.to_string(), sources=["Missing-value counts on the active dataset"])


def t_group_aggregate(ctx: Ctx, group: str, value: str, agg: str = "mean", top_n: int = 15) -> ToolResult:
    g = _col(ctx, group)
    v = _col(ctx, value)
    if agg not in ("mean", "median", "sum", "min", "max", "count", "rate"):
        raise ValueError(f"Unsupported aggregation '{agg}'.")
    spec = ch.ChartSpec("bar", x=g, y=None if agg == "count" else v, agg=agg, top_n=max(1, min(int(top_n), 50)))
    out = ch.render(ctx.df, spec, ctx.kinds)
    return ToolResult(out.interpretation, "group_aggregate", {"group": g, "value": v, "agg": agg}, table=out.table, spec=spec.to_dict(),
                      facts=out.interpretation + " " + out.table.to_string(), sources=[f"{agg} of {v} grouped by {g}"])


def t_filter_count(ctx: Ctx, filters: list[dict[str, Any]]) -> ToolResult:
    clean = []
    for f in filters or []:
        if not isinstance(f, dict) or f.get("op") not in ch.FILTER_OPS:
            raise ValueError("Each filter needs a column, an operator (==, !=, >, >=, <, <=, in, not_in, contains) and a value.")
        clean.append({"column": _col(ctx, f.get("column")), "op": f["op"], "value": f.get("value")})
    sub = ch.apply_filters(ctx.df, clean)
    cond = " and ".join(f"{f['column']} {f['op']} {f['value']}" for f in clean) or "no filter"
    text = f"{len(sub):,} of {len(ctx.df):,} rows ({pct(100 * len(sub) / max(len(ctx.df), 1))}) match {cond}."
    return ToolResult(text, "filter_count", {"filters": clean}, table=sub.head(0).T.reset_index().rename(columns={"index": "column"}).head(0),
                      facts=text, sources=["Row filter on the active dataset"])


def t_correlations(ctx: Ctx, column: str | None = None, top_n: int = 10) -> ToolResult:
    pairs = ctx.results["correlations"]["pairs"]
    if column:
        c = _col(ctx, column, ("numeric",))
        pairs = [p for p in pairs if c in (p["a"], p["b"])]
    if not pairs:
        return ToolResult("No numeric correlations are available for this request.", "correlations", {"column": column}, facts="none",
                          sources=["Pearson correlations"])
    t = pd.DataFrame(pairs[:top_n])
    text = "Strongest Pearson correlations: " + "; ".join(f"{p['a']} ↔ {p['b']} r = {p['r']:.2f} (n = {p['n']:,})" for p in pairs[:5]) + ". " + eda.CAUSAL_NOTE
    cols = sorted({x for p in pairs[:8] for x in (p["a"], p["b"])})
    return ToolResult(text, "correlations", {"column": column}, table=t, spec=ch.ChartSpec("heatmap", columns=cols).to_dict() if len(cols) >= 2 else None,
                      facts=text + " " + t.to_string(), sources=[ctx.results["correlations"]["method"]])


def _assoc(df: pd.DataFrame, f: str, t: str, kinds: dict[str, str]) -> tuple[float | None, str]:
    d = df[[f, t]].dropna()
    if len(d) < 10 or d[f].nunique() < 2 or d[t].nunique() < 2:
        return None, "insufficient data"
    t_num = kinds[t] == "numeric" and d[t].nunique() > 12
    f_num = kinds[f] == "numeric" and d[f].nunique() > 12
    if f_num and t_num:
        return float(abs(d[f].astype(float).corr(d[t].astype(float)))), "|Pearson r|"
    if f_num or t_num:  # correlation ratio eta: numeric explained by categories
        y, g = (d[f].astype(float), d[t].astype(str)) if f_num else (d[t].astype(float), d[f].astype(str))
        means = y.groupby(g).transform("mean")
        ss_b, ss_t = ((means - y.mean()) ** 2).sum(), ((y - y.mean()) ** 2).sum()
        return (float(np.sqrt(ss_b / ss_t)) if ss_t else None), "correlation ratio η"
    tab = pd.crosstab(d[f].astype(str), d[t].astype(str))
    if tab.shape[0] > 60:
        return None, "too many categories"
    from scipy.stats import chi2_contingency

    chi2 = chi2_contingency(tab, correction=False)[0]
    k = min(tab.shape) - 1
    return (float(np.sqrt(chi2 / (tab.values.sum() * k))) if k else None), "Cramér's V"


def t_target_associations(ctx: Ctx, top_n: int = 10) -> ToolResult:
    target = ctx.results.get("target")
    if not target:
        return ToolResult("No target column is selected. Choose one on the Model step (or ask about correlations instead).",
                          "target_associations", {}, ok=False, navigate=("views/ml.py", "Choose a target"))
    ids = set(ctx.results["profile"]["id_candidates"])
    rows = []
    for f in ctx.df.columns:
        if f == target or f in ids or ctx.kinds[f] not in ("numeric", "categorical"):
            continue
        v, m = _assoc(ctx.df, f, target, ctx.kinds)
        if v is not None and np.isfinite(v):
            rows.append({"feature": f, "strength": round(v, 4), "measure": m})
    if not rows:
        return ToolResult("No feature could be compared with the target.", "target_associations", {}, ok=False)
    t = pd.DataFrame(rows).sort_values("strength", ascending=False).head(int(top_n))
    fig = go.Figure(go.Bar(x=t["strength"][::-1], y=t["feature"][::-1], orientation="h", marker_color=eda.SAGE,
                           customdata=t["measure"][::-1], hovertemplate="%{y}: %{x:.3f} (%{customdata})<extra></extra>"))
    fig.update_layout(xaxis_title="Association strength (0–1)")
    eda.style(fig, f"Association with {target}", height=max(280, 28 * len(t) + 100))
    top = t.iloc[0]
    text = (f"Features most strongly associated with **{target}** (unadjusted, computed on the active data): "
            + "; ".join(f"{r.feature} {r.strength:.2f} ({r.measure})" for r in t.head(5).itertuples())
            + f". The strongest is {top['feature']}. Association is not causation, and strong associations can signal leakage — check how "
              "each feature was recorded. For model-based importance, use held-out permutation importance on the Results step.")
    return ToolResult(text, "target_associations", {"top_n": top_n}, table=t, figure=fig, facts=text + " " + t.to_string(),
                      sources=["Pairwise association: |Pearson r|, correlation ratio η, or Cramér's V by column type"],
                      followups=["Which features influenced model predictions?", "Which model performed best?"])


def t_make_chart(ctx: Ctx, request: str) -> ToolResult:
    prev = ch.ChartSpec.from_dict(ctx.previous_spec) if ctx.previous_spec else None
    res = nl_chart.parse(request, ctx.df, ctx.kinds, ctx.results.get("target"), prev)
    if res.spec is None:
        msg = " ".join(res.errors) + (f" {res.clarification}" if res.clarification else "")
        return ToolResult(msg, "make_chart", {"request": request}, ok=False, facts=msg)
    out = ch.render(ctx.df, res.spec, ctx.kinds)
    text = out.interpretation + ("\n\n" + " ".join(f"_{a}_" for a in res.assumptions) if res.assumptions else "")
    return ToolResult(text, "make_chart", {"request": request}, table=out.table, spec=res.spec.to_dict(),
                      facts=out.interpretation + " " + out.table.head(30).to_string(), sources=[f"Chart: {res.spec.describe()}"] + out.notes,
                      followups=["Now split it by another column", "Add this chart to the report"])


def t_model_summary(ctx: Ctx) -> ToolResult:
    m = ctx.results.get("modeling")
    if not m:
        return ToolResult("No model has been trained yet, so there is nothing to compare.", "model_summary", {}, ok=False,
                          navigate=("views/ml.py", "Open Model"))
    ok = [x for x in m["models"] if "error" not in x]
    rows = []
    for x in ok:
        r = {"model": x["name"]}
        if x.get("cv"):
            r[f"CV {x['cv']['metric']} (training folds)"] = round(x["cv"]["mean"], 4)
        for k in m["metric_names"]:
            r[f"test {k}"] = x["metrics"].get(k)
        rows.append(r)
    t = pd.DataFrame(rows)
    best = ok[0]
    pm = m["primary_metric"]
    text = (f"Best model: **{m['best_model']}**, ranked by {m['ranking_basis']}. Its held-out test {pm} is "
            f"{metric(best['metrics'].get(pm))} on {m['n_test']:,} test rows (trained on {m['n_train']:,}).")
    if len(ok) > 1:
        text += f" Runner-up: {ok[1]['name']} (test {pm} {metric(ok[1]['metrics'].get(pm))})."
    text += " Differences between close scores may not be meaningful on a single test split."
    return ToolResult(text, "model_summary", {}, table=t, facts=text + " " + t.to_string(), sources=["Model leaderboard (held-out test set)"],
                      navigate=("views/model_results.py", "Open Results"),
                      followups=["Explain the confusion matrix", "Which features influenced model predictions?"])


def t_confusion_matrix(ctx: Ctx) -> ToolResult:
    m = ctx.results.get("modeling")
    if not m or m["task"] != "classification":
        return ToolResult("A confusion matrix needs a trained classification model.", "confusion_matrix", {}, ok=False,
                          navigate=("views/ml.py", "Open Model"))
    best = next(x for x in m["models"] if "error" not in x)
    cm = np.array(best["confusion_matrix"])
    classes = m["classes"]
    lines = []
    for i, c in enumerate(classes):
        tot = int(cm[i].sum())
        lines.append(f"actual '{c}': {int(cm[i, i]):,} of {tot:,} predicted correctly ({pct(100 * cm[i, i] / tot if tot else None)})")
    text = (f"For **{best['name']}** on {m['n_test']:,} held-out test rows, rows are actual classes and columns are predictions; the diagonal "
            f"counts correct predictions ({int(np.trace(cm)):,} in total). " + "; ".join(lines) + ".")
    if len(classes) == 2:
        fp, fn = int(cm[0, 1]), int(cm[1, 0])
        text += f" There are {fp:,} false positives (predicted '{classes[1]}' but actually '{classes[0]}') and {fn:,} false negatives."
    t = pd.DataFrame(cm, index=[f"actual {c}" for c in classes], columns=[f"pred {c}" for c in classes]).reset_index(names="")
    fig = go.Figure(go.Heatmap(z=cm, x=classes, y=classes, colorscale=eda.SEQUENTIAL, showscale=False, text=cm, texttemplate="%{text}"))
    fig.update_xaxes(type="category", title="Predicted")
    fig.update_yaxes(type="category", autorange="reversed", title="Actual")
    eda.style(fig, f"Confusion matrix — {best['name']}", height=360)
    return ToolResult(text, "confusion_matrix", {}, table=t, figure=fig, facts=text + " " + t.to_string(), sources=["Held-out confusion matrix"])


def t_cleaning_suggestions(ctx: Ctx) -> ToolResult:
    tips = ctx.results["cleaning"]["suggestions"]
    if not tips:
        text = "The profile found nothing that clearly needs cleaning."
    else:
        text = "Steps worth considering (each is previewed with its exact impact before you apply it):\n" + "\n".join(f"- {t}" for t in tips[:10])
    return ToolResult(text, "cleaning_suggestions", {}, facts=text, sources=["Profiler-based suggestions"], navigate=("views/cleaning.py", "Open Clean"))


def t_segments(ctx: Ctx) -> ToolResult:
    cl = ctx.results.get("clustering")
    if not cl:
        return ToolResult("No clustering has been run on the current data. Run it on the Segments page (no target needed); "
                          "I can then describe the segments from their computed profiles.", "segments", {}, ok=False,
                          navigate=("views/clustering.py", "Open Segments"))
    names = cl.get("names") or {}
    sizes = "; ".join(f"{names.get(str(k), 'Cluster ' + str(k))}: {v:,} rows" for k, v in cl["sizes"].items())
    met = cl["metrics"]
    text = (f"{cl['algorithm']} found {cl['n_clusters']} cluster(s) on {', '.join(cl['features'] or [])}: {sizes}. "
            + (f"Silhouette {met['silhouette']:.3f}. " if met.get("silhouette") is not None else "Silhouette not defined for this result. ")
            + "\n" + "\n".join(f"- {d}" for d in cl.get("descriptions", [])[:6]))
    return ToolResult(text, "segments", {}, facts=text, sources=["Clustering result for the current data version"],
                      navigate=("views/clustering.py", "Open Segments"))


def t_insights(ctx: Ctx, category: str | None = None) -> ToolResult:
    ins = ctx.results["insights"]
    if category:
        ins = [i for i in ins if i["category"].lower() == category.lower()] or ins
    text = "Main findings from the computed analysis:\n" + "\n".join(f"- **[{i['id']}] {i['title']}** — {i['observation']}" for i in ins[:8])
    return ToolResult(text, "insights", {"category": category}, facts=text, sources=[f"Evidence register {i['id']}" for i in ins[:8]],
                      navigate=("views/insights.py", "Open Insights"))


def t_feature_importance(ctx: Ctx) -> ToolResult:
    imp = ctx.results.get("importance")
    if not imp:
        return ToolResult("Feature importance has not been computed. On the Results step, choose “Compute importance” — it is measured "
                          "by shuffling each feature on the held-out test set.", "feature_importance", {}, ok=False,
                          navigate=("views/model_results.py", "Open Results"))
    t = pd.DataFrame(imp["features"]).head(10)
    text = (f"Held-out permutation importance for {imp['model']} ({imp['metric_label']}, {imp['n_repeats']} repeats): "
            + "; ".join(f"{r.feature} {r.mean:.4f} ± {r.std:.4f}" for r in t.head(5).itertuples())
            + ". Importance reflects this model's reliance on a feature, not a causal effect.")
    return ToolResult(text, "feature_importance", {}, table=t, facts=text + " " + t.to_string(), sources=["Permutation importance on the test set"])


TOOLS: dict[str, dict[str, Any]] = {
    "dataset_summary": {"fn": t_dataset_summary, "description": "Overview of the active dataset: size, column types, provenance, missing values.",
                        "parameters": {"type": "object", "properties": {}}},
    "quality_issues": {"fn": t_quality_issues, "description": "The biggest data-quality issues (missing values, duplicates, identifiers, outliers, imbalance).",
                       "parameters": {"type": "object", "properties": {}}},
    "column_stats": {"fn": t_column_stats, "description": "Summary statistics for one column.",
                     "parameters": {"type": "object", "properties": {"column": {"type": "string"}}, "required": ["column"]}},
    "missing_by_column": {"fn": t_missing_by_column, "description": "Missing values per column, with a chart.",
                          "parameters": {"type": "object", "properties": {}}},
    "group_aggregate": {"fn": t_group_aggregate, "description": "Aggregate a value column by a grouping column (mean, median, sum, min, max, count, rate).",
                        "parameters": {"type": "object", "properties": {"group": {"type": "string"}, "value": {"type": "string"},
                                                                         "agg": {"type": "string", "enum": ["mean", "median", "sum", "min", "max", "count", "rate"]},
                                                                         "top_n": {"type": "integer", "minimum": 1, "maximum": 50}},
                                       "required": ["group", "value"]}},
    "filter_count": {"fn": t_filter_count, "description": "Count rows matching filters.",
                     "parameters": {"type": "object", "properties": {"filters": {"type": "array", "items": {"type": "object", "properties": {
                         "column": {"type": "string"}, "op": {"type": "string", "enum": ch.FILTER_OPS}, "value": {}}}}}, "required": ["filters"]}},
    "correlations": {"fn": t_correlations, "description": "Strongest numeric correlations, optionally involving one column.",
                     "parameters": {"type": "object", "properties": {"column": {"type": "string"}, "top_n": {"type": "integer"}}}},
    "target_associations": {"fn": t_target_associations, "description": "Features most strongly associated with the selected target.",
                            "parameters": {"type": "object", "properties": {"top_n": {"type": "integer"}}}},
    "make_chart": {"fn": t_make_chart, "description": "Create a chart from a plain-language request (distribution, comparison, trend, relationship, heatmap).",
                   "parameters": {"type": "object", "properties": {"request": {"type": "string"}}, "required": ["request"]}},
    "model_summary": {"fn": t_model_summary, "description": "Which model performed best and the evaluation metrics.",
                      "parameters": {"type": "object", "properties": {}}},
    "confusion_matrix": {"fn": t_confusion_matrix, "description": "Explain the confusion matrix of the best classification model.",
                         "parameters": {"type": "object", "properties": {}}},
    "feature_importance": {"fn": t_feature_importance, "description": "Held-out permutation feature importance of the trained model.",
                           "parameters": {"type": "object", "properties": {}}},
    "cleaning_suggestions": {"fn": t_cleaning_suggestions, "description": "Cleaning steps worth considering.",
                             "parameters": {"type": "object", "properties": {}}},
    "segments": {"fn": t_segments, "description": "Describe the customer/record segments found by clustering.",
                 "parameters": {"type": "object", "properties": {}}},
    "insights": {"fn": t_insights, "description": "Summarise the main findings.",
                 "parameters": {"type": "object", "properties": {"category": {"type": "string"}}}},
}


def validate_args(name: str, args: Any) -> dict[str, Any]:
    """Strict argument validation against the tool schema (types, enums, required, no extras)."""
    if name not in TOOLS:
        raise ValueError(f"Unknown tool '{name}'.")
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ValueError("Tool arguments must be an object.")
    schema = TOOLS[name]["parameters"]
    props = schema.get("properties", {})
    extra = set(args) - set(props)
    if extra:
        raise ValueError(f"Unexpected argument(s) for {name}: {', '.join(sorted(extra))}.")
    for req in schema.get("required", []):
        if req not in args or args[req] in (None, ""):
            raise ValueError(f"{name} needs '{req}'.")
    py = {"string": str, "integer": int, "array": list, "object": dict}
    for k, v in args.items():
        spec = props[k]
        t = spec.get("type")
        if t and v is not None and not isinstance(v, py.get(t, object)):
            if t == "integer" and isinstance(v, str) and v.isdigit():
                args[k] = int(v)
                continue
            raise ValueError(f"Argument '{k}' must be {t}.")
        if "enum" in spec and v not in spec["enum"]:
            raise ValueError(f"Argument '{k}' must be one of {', '.join(map(str, spec['enum']))}.")
        if isinstance(v, str) and len(v) > 500:
            raise ValueError(f"Argument '{k}' is too long.")
    return args


def run_tool(name: str, args: dict[str, Any], ctx: Ctx) -> ToolResult:
    args = validate_args(name, dict(args or {}))
    fn: Callable[..., ToolResult] = TOOLS[name]["fn"]
    try:
        return fn(ctx, **args)
    except (ValueError, ch.ChartSpecError) as exc:
        return ToolResult(str(exc), name, args, ok=False, facts=str(exc))


# ---------------------------------------------------------------- routing
_CHART = re.compile(r"\b(plot|chart|graph|visuali[sz]e|histogram|scatter|heat ?map|bar chart|line chart|box ?plot|violin|"
                    r"distribution of|draw|show (me )?(the )?(distribution|trend|relationship|counts?|top))\b", re.I)


def route(question: str, ctx: Ctx) -> tuple[str, dict[str, Any]] | None:
    q = question.strip()
    low = q.lower()
    cols = nl_chart.resolve_columns(q, list(ctx.df.columns))
    if re.search(r"missing values? (by|per) column|show missing|missingness", low):
        return "missing_by_column", {}
    if re.search(r"confusion matrix", low):
        return "confusion_matrix", {}
    if re.search(r"(best|top|winning) model|model perform|which model|leaderboard|compare (the )?models", low):
        return "model_summary", {}
    if re.search(r"feature importance|influenc\w* (the )?(model|prediction)|important features|drives? the model", low):
        return "feature_importance", {}
    if re.search(r"associated with the target|associat\w+ with \w+|predictive of|related to the target|strongest (association|relationship)s?", low) \
            and not _CHART.search(low):
        return "target_associations", {}
    if re.search(r"cleaning steps?|should i clean|what (should|to) clean|how (should|do) i clean", low):
        return "cleaning_suggestions", {}
    if re.search(r"segments?|clusters?|groups of (customers|users|records)", low) and not _CHART.search(low):
        return "segments", {}
    if re.search(r"biggest issues|quality issues|problems with (the|this) data|data quality|issues in (the|this) dataset", low):
        return "quality_issues", {}
    if re.search(r"summari[sz]e (the )?(main )?findings|key findings|main findings|what did you find|insights", low):
        return "insights", {}
    if re.search(r"correlat", low) and not re.search(r"heat ?map|matrix|plot|chart", low):
        return "correlations", {"column": cols[0]} if cols and ctx.kinds.get(cols[0]) == "numeric" else {}
    if _CHART.search(low) or (ctx.previous_spec and re.match(r"^\s*(now|same|and|also|instead|what about)\b", low)):
        return "make_chart", {"request": q}
    if re.search(r"how many rows|how many (records|entries)", low) and re.search(r"\bwhere\b|\bwith\b|\bfor\b", low):
        filters, _ = nl_chart._value_filters(q, ctx.df, ctx.kinds, cols)
        if filters:
            return "filter_count", {"filters": filters}
        return "_clarify", {"text": "I couldn't match that condition to a column and value in this dataset. Try “how many rows where "
                                    "<column> is <value>”, e.g. with one of: " + ", ".join(list(ctx.df.columns)[:8]) + "."}
    m = re.search(r"\b(average|mean|median|total|sum|max(?:imum)?|min(?:imum)?)\s+(?:of\s+)?(.+?)\s+(?:by|per|for each|across)\s+(.+)", low)
    if m:
        agg = {"average": "mean", "total": "sum", "maximum": "max", "minimum": "min"}.get(m.group(1), m.group(1))
        v = nl_chart.resolve_columns(m.group(2), list(ctx.df.columns))
        g = nl_chart.resolve_columns(m.group(3), list(ctx.df.columns))
        if v and g:
            return "group_aggregate", {"group": g[0], "value": v[0], "agg": agg}
        return "make_chart", {"request": q}  # lets the chart parser explain which column it could not find
    if re.search(r"^\s*compare\b|\b(rate|share)s?\b.*\b(by|across|per)\b", low):
        return "make_chart", {"request": q}
    if re.search(r"how many rows|how many columns|overview|describe (the|this) dataset|dataset summary|what is this dataset", low):
        return "dataset_summary", {}
    if len(cols) == 1 and re.search(r"\b(stats|statistics|summary|describe|tell me about|mean|average|median|range|distinct|unique)\b", low):
        return "column_stats", {"column": cols[0]}
    return None


def _llm_route(question: str, ctx: Ctx, history: list[dict[str, Any]]) -> tuple[str, dict[str, Any]] | None:
    cfg = llm.get_config()
    if cfg is None:
        return None
    tools = [{"type": "function", "function": {"name": n, "description": t["description"], "parameters": t["parameters"]}}
             for n, t in TOOLS.items()]
    cols = ", ".join(f"{c} ({ctx.kinds[c]})" for c in list(ctx.df.columns)[:60])
    sys = ("You route questions about a tabular dataset to exactly one analysis tool. Use only the listed tools and only column names "
           f"from this list: {cols}. If no tool fits, reply with the single word NONE.")
    msgs = [{"role": "system", "content": sys}]
    for h in history[-4:]:
        if h.get("role") == "user":
            msgs.append({"role": "user", "content": str(h.get("content", ""))[:400]})
    msgs.append({"role": "user", "content": question[:1000]})
    try:
        msg = llm.chat_tools(msgs, tools, config=cfg)
    except llm.LLMError:
        return None
    calls = msg.get("tool_calls") or []
    if not calls:
        return None
    fn = calls[0].get("function", {})
    try:
        args = json.loads(fn.get("arguments") or "{}")
        name = fn.get("name")
        return name, validate_args(name, args)
    except (ValueError, json.JSONDecodeError):
        return None


def ask(question: str, ctx: Ctx, history: list[dict[str, Any]] | None = None, use_llm: bool = True) -> dict[str, Any]:
    """Answer a question. Returns a JSON-friendly message dict (plus optional table/figure objects)."""
    q = (question or "").strip()
    base = {"role": "assistant", "question": q, "tool": None, "args": {}, "spec": None, "table": None, "figure": None,
            "sources": [], "navigate": None, "followups": [], "note": None, "mode": "deterministic"}
    if not q:
        return {**base, "answer": "Ask a question about the active dataset or its analysis."}
    if _CODE.search(q):
        return {**base, "answer": UNSUPPORTED, "tool": "refused"}
    routed, routed_by = None, "rules"
    if use_llm and llm.get_config() is not None:
        routed = _llm_route(q, ctx, history or [])
        routed_by = "llm" if routed else "rules"
    routed = routed or route(q, ctx)
    if routed is None:
        legacy = assistant.answer(q, ctx.results, use_llm=use_llm)
        return {**base, "answer": legacy["answer"], "sources": legacy["sources"], "mode": legacy["mode"], "note": legacy.get("note"),
                "tool": "retrieval"}
    name, args = routed
    if name == "_clarify":
        return {**base, "answer": args["text"], "tool": "clarification"}
    res = run_tool(name, args, ctx)
    out = {**base, "answer": res.text, "tool": name, "args": res.args, "spec": res.spec, "table": res.table, "figure": res.figure,
           "sources": res.sources, "navigate": res.navigate, "followups": res.followups, "routed_by": routed_by, "ok": res.ok}
    if res.ok and use_llm and llm.get_config() is not None and res.facts:
        try:
            text = llm.chat([
                {"role": "system", "content": "Rephrase the verified analysis result for the user in 2-5 sentences. Use only numbers present in "
                                              "the RESULT. Keep caveats about causation, sampling and evaluation method. Do not add new claims."},
                {"role": "user", "content": f"QUESTION: {q}\n\nRESULT:\n{res.facts[:6000]}"}], max_tokens=350)
            bad = llm.unverified_numbers(text, res.facts + " " + q)
            if bad:
                out["note"] = f"The language model's wording contained figures not in the computed result ({', '.join(bad[:4])}); showing the verified result."
            else:
                out["answer"], out["mode"] = text, "llm"
        except llm.LLMError as exc:
            out["note"] = f"{exc} Showing the verified result."
    return out
