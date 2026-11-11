"""Deterministic insight engine.

Each insight follows Observation → Evidence → Interpretation → Limitation →
Suggested next step, and is assembled only from values already present in the
shared results structure. No language model is involved here.
"""
from __future__ import annotations

from typing import Any

from core.data_loader import SYNTHETIC_LABEL
from core.eda import CAUSAL_NOTE
from core.modeling import LOWER_IS_BETTER, METRIC_LABELS


def _insight(category, title, observation, evidence, interpretation, limitation, next_step, severity="info", source=""):
    return {"category": category, "title": title, "observation": observation, "evidence": evidence,
            "interpretation": interpretation, "limitation": limitation, "next_step": next_step,
            "severity": severity, "source": source}


def generate(results: dict[str, Any]) -> list[dict[str, Any]]:
    p, q, ds = results["profile"], results["quality"], results["dataset"]
    out: list[dict[str, Any]] = []
    kc = p["kind_counts"]

    if ds.get("synthetic"):
        out.append(_insight(
            "Provenance", SYNTHETIC_LABEL,
            f"'{ds['name']}' is a synthetic demonstration dataset.",
            [ds["provenance"]],
            "Every pattern reported below is a property of generated values. It illustrates how DataStory AI works, not how the real world behaves.",
            "No conclusion from this dataset should be transferred to real passengers, customers, riders or students.",
            "Upload your own data to obtain findings that mean something.", "attention", "Sample dataset registry"))

    out.append(_insight(
        "Overview", "Dataset shape and composition",
        f"The dataset has {p['rows']:,} rows and {p['columns']:,} columns.",
        [f"Numeric columns: {kc['numeric']}", f"Categorical columns: {kc['categorical']}",
         f"Datetime columns: {kc['datetime']}", f"Text columns: {kc['text']}",
         f"Memory footprint: {p['memory_mb']:.2f} MB"],
        "Column types were inferred automatically from the values and decide which charts and model preprocessing apply.",
        "Type inference is heuristic: numeric codes (for example 1/2/3 for a category) are reported as numeric.",
        "Check the column table on the Dataset page and convert any mis-typed column on the Data Cleaning page.",
        source="Profiler · column type inference"))

    # missing data
    miss_cols = sorted([c for c in p["column_details"] if c["missing"]], key=lambda c: -c["missing"])
    if miss_cols:
        worst = miss_cols[0]
        sev = "warning" if worst["missing_pct"] > 30 else "attention"
        out.append(_insight(
            "Data quality", "Missing values",
            f"{p['missing_cells']:,} cells are missing ({p['missing_pct']:.2f}% of all cells), across {len(miss_cols)} column(s).",
            [f"{c['name']}: {c['missing']:,} missing ({c['missing_pct']:.1f}%)" for c in miss_cols[:6]],
            f"Missingness is concentrated in '{worst['name']}'. Rows with gaps are dropped or imputed by most analyses, which changes what the remaining data represents.",
            "The data alone cannot tell whether values are missing at random or for a systematic reason.",
            "Decide per column: impute (median/mode/KNN), drop the column if it is mostly empty, or keep missingness as a signal.",
            sev, "Profiler · missing-value counts"))
    else:
        out.append(_insight(
            "Data quality", "No missing values", "Every cell in the dataset is populated.",
            [f"Completeness: {q['dimensions']['completeness']['value']:.1f}%"],
            "No imputation is needed before analysis.",
            "Placeholder codes such as 0, -1 or 'unknown' are not detected as missing.",
            "Scan categorical value lists for placeholder codes.", source="Quality · completeness"))

    if p["duplicate_rows"]:
        out.append(_insight(
            "Data quality", "Exact duplicate rows",
            f"{p['duplicate_rows']:,} rows ({p['duplicate_pct']:.2f}%) are exact copies of an earlier row.",
            [f"Uniqueness: {q['dimensions']['uniqueness']['value']:.2f}%"],
            "Duplicates inflate counts and can leak identical records into both the training and the test split.",
            "Identical rows can be legitimate when few columns are recorded; only all-column matches are counted.",
            "Review the duplicates and remove them on the Data Cleaning page if they are not genuine repeats.",
            "attention", "Quality · uniqueness"))

    structural = []
    if p["constant_columns"]:
        structural.append(f"Constant: {', '.join(p['constant_columns'])}")
    if p["near_constant_columns"]:
        structural.append(f"Near-constant (one value ≥ 95% of rows): {', '.join(p['near_constant_columns'])}")
    if p["id_candidates"]:
        structural.append(f"Identifier-like (unique per row): {', '.join(p['id_candidates'])}")
    if structural:
        out.append(_insight(
            "Structure", "Columns with little analytical value",
            "Some columns are constant, nearly constant, or look like row identifiers.", structural,
            "Such columns add no signal to a model; identifiers can even let a model memorise rows.",
            "Identifier detection is a heuristic based on uniqueness and naming.",
            "Exclude these columns from modelling unless you know they carry meaning.", source="Profiler · cardinality"))

    if p["outlier_columns"]:
        top = p["outlier_columns"][:5]
        out.append(_insight(
            "Distribution", "Potential outliers",
            f"{len(p['outlier_columns'])} numeric column(s) contain values outside the Tukey fences.",
            [f"{r['column']}: {r['outliers']:,} values ({r['outlier_pct']:.1f}%)" for r in top],
            "Extreme values can dominate means, correlations and linear models.",
            "The 1.5×IQR rule flags many legitimate values in skewed distributions; a flagged value is not necessarily an error.",
            "Inspect the box plots, then keep, cap or remove outliers deliberately.",
            "attention" if top[0]["outlier_pct"] > 5 else "info", "Profiler · Tukey 1.5×IQR rule"))

    skewed = sorted([c for c in p["column_details"] if c.get("skew") is not None and abs(c["skew"]) > 1
                     and c["name"] not in p["id_candidates"]], key=lambda c: -abs(c["skew"]))
    if skewed:
        out.append(_insight(
            "Distribution", "Skewed numeric distributions",
            f"{len(skewed)} numeric column(s) have absolute skewness above 1.",
            [f"{c['name']}: skew {c['skew']:.2f}, mean {c['mean']:.4g}, median {c['median']:.4g}" for c in skewed[:5]],
            "For these columns the mean is pulled toward the long tail, so the median is a better summary of a typical value.",
            "Skewness is sensitive to a handful of extreme values.",
            "Consider a log or rank transform before linear modelling.", source="Profiler · descriptive statistics"))

    pairs = results["correlations"]["pairs"]
    if pairs:
        strong = [x for x in pairs if abs(x["r"]) >= 0.7]
        shown = strong[:5] or pairs[:3]
        out.append(_insight(
            "Relationships", "Correlated numeric columns",
            (f"{len(strong)} pair(s) of numeric columns have |r| ≥ 0.7." if strong else
             f"No pair of numeric columns reaches |r| ≥ 0.7; the strongest is r = {pairs[0]['r']:.2f}."),
            [f"{x['a']} ↔ {x['b']}: r = {x['r']:.3f} (n = {x['n']:,})" for x in shown],
            ("Strongly correlated features carry overlapping information, which makes linear-model coefficients and "
             "permutation importances harder to attribute to either one." if strong else
             "Numeric features are not strongly linearly redundant."),
            CAUSAL_NOTE + " Pearson r only captures linear association.",
            "Plot the listed pairs on the Exploratory Analysis page before drawing conclusions.",
            source="Correlation matrix (Pearson)"))

    t = p.get("target")
    if t and t.get("class_counts"):
        counts = t["class_counts"]
        ratio = t["imbalance_ratio"]
        level = "balanced" if ratio < 1.5 else ("moderately imbalanced" if ratio < 3 else "strongly imbalanced")
        out.append(_insight(
            "Target", f"Class balance of '{t['name']}'",
            f"The target has {len(counts)} classes and is {level} (largest : smallest = {ratio:.2f} : 1).",
            [f"{k}: {v:,} rows" for k, v in list(counts.items())[:8]],
            ("Accuracy alone can look good by favouring the majority class; F1, recall and ROC-AUC are more informative here."
             if ratio >= 1.5 else "Accuracy is a reasonable headline metric because classes are of similar size."),
            "Balance in this sample may differ from the population the model would be used on.",
            ("Try class weighting or SMOTE on the Machine Learning page and compare recall for the minority class."
             if ratio >= 1.5 else "No rebalancing is needed."),
            "attention" if ratio >= 3 else "info", "Profiler · target class counts"))
    elif t and t.get("missing"):
        out.append(_insight(
            "Target", f"Missing target values in '{t['name']}'", f"{t['missing']:,} rows have no target value.",
            [f"Missing target rows: {t['missing']:,}"], "Rows without a target cannot be used for supervised training.",
            "Why the target is missing is unknown.", "These rows are excluded automatically during training.",
            "attention", "Profiler · target summary"))

    m = results.get("modeling")
    if m:
        ok = [x for x in m["models"] if "error" not in x]
        best = ok[0]
        metric = m["primary_metric"]
        label = METRIC_LABELS[metric]
        ev = []
        for x in ok[:6]:
            line = f"{x['name']}: test {label} = {_v(x['metrics'].get(metric))}"
            if x["cv"]:
                line += f", CV {label} = {x['cv']['mean']:.4f} ± {x['cv']['std']:.4f}"
            ev.append(line)
        interp = f"'{best['name']}' ranks first by {m['ranking_basis']}."
        if len(ok) > 1 and best["metrics"].get(metric) is not None and ok[1]["metrics"].get(metric) is not None:
            gap = abs(best["metrics"][metric] - ok[1]["metrics"][metric])
            interp += f" The gap to the runner-up ('{ok[1]['name']}') on the test set is {gap:.4f} {label}" + (
                ", which is small relative to fold-to-fold variation." if best["cv"] and gap < best["cv"]["std"] else ".")
        if m["task"] == "classification" and best["metrics"].get("accuracy") is not None and m.get("class_counts_train"):
            tc = m["class_counts_train"]
            base = max(tc.values()) / sum(tc.values())
            interp += (f" Always predicting the majority training class would score about {base:.4f} accuracy; "
                       f"the best model scores {best['metrics']['accuracy']:.4f}.")
        if m["task"] == "regression" and best["metrics"].get("r2") is not None and best["metrics"]["r2"] <= 0:
            interp += " An R² at or below zero means the model predicts no better than the mean of the target."
        out.append(_insight(
            "Modelling", f"Model comparison ({m['task']})",
            f"{len(ok)} model(s) were trained to predict '{m['target']}' on {m['n_train']:,} rows and evaluated on {m['n_test']:,} held-out rows.",
            ev, interp,
            f"Scores come from a single {1 - m['test_size']:.0%}/{m['test_size']:.0%} split with seed {m['seed']}; a different split would give somewhat different numbers"
            + ("." if m["n_test"] >= 100 else f", and {m['n_test']} test rows is a small evaluation set."),
            "Validate on fresh data before relying on the model, and review errors by class or by residual plot.",
            source="Model training · held-out test set" + (" + cross-validation" if m["cv_folds"] else "")))

    imp = results.get("importance")
    if imp:
        feats = imp["features"]
        top = feats[:5]
        direction = "increase" if imp["metric"] in LOWER_IS_BETTER else "drop"
        out.append(_insight(
            "Explainability", f"What '{imp['model']}' relies on",
            f"Shuffling '{top[0]['feature']}' changes the test {imp['metric_label']} the most (mean {direction} {abs(top[0]['mean']):.4f}).",
            [f"{f['feature']}: {f['mean']:+.4f} ± {f['std']:.4f}" for f in top],
            f"Values are the mean change in {imp['metric_label']} over {imp['n_repeats']} shuffles on the {imp['evaluated_on']}. "
            "Features near zero contribute little to this particular model.",
            "Importance is not a percentage contribution, does not sum to 100%, and is not causal. " + " ".join(imp["notes"]),
            "Check whether the top features are available at prediction time and make domain sense.",
            source="Permutation importance · held-out test set"))
    return out


def _v(x) -> str:
    return "Unavailable" if x is None else f"{x:.4f}"


def facts_text(insights: list[dict[str, Any]]) -> str:
    """Flatten insights into the verified fact sheet handed to an optional LLM."""
    lines = []
    for i, ins in enumerate(insights, 1):
        lines.append(f"[{i}] {ins['title']} ({ins['category']})")
        lines.append(f"  Observation: {ins['observation']}")
        lines += [f"  Evidence: {e}" for e in ins["evidence"]]
        lines.append(f"  Interpretation: {ins['interpretation']}")
        lines.append(f"  Limitation: {ins['limitation']}")
        lines.append(f"  Next step: {ins['next_step']}")
    return "\n".join(lines)


def narrative(insights: list[dict[str, Any]]) -> str:
    """Deterministic plain-language story used when no LLM is configured (or its output fails validation)."""
    parts = []
    for ins in insights:
        parts.append(f"**{ins['title']}.** {ins['observation']} {ins['interpretation']} *Caveat:* {ins['limitation']}")
    return "\n\n".join(parts)
