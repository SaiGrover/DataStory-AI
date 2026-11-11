"""Retrieval over verified analysis facts and a small concept glossary.

Documents of kind "analysis" are generated from the shared results structure,
so the assistant can only quote values the application actually computed.
Documents of kind "concept" explain methods and contain no dataset numbers.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core.data_loader import SYNTHETIC_LABEL
from core.modeling import METRIC_LABELS
from core.quality_metrics import HEALTH_FORMULA, format_dimension

CONCEPTS = [
    ("concept_accuracy", "What accuracy means", "Accuracy is the share of test rows the classifier labelled correctly. It can be misleading when classes are imbalanced, because always predicting the majority class already scores high."),
    ("concept_precision_recall", "What precision, recall and F1 mean", "Precision is the share of predicted positives that are truly positive. Recall is the share of true positives the model found. F1-score is the harmonic mean of precision and recall, useful when classes are imbalanced. For multi-class targets DataStory AI reports class-weighted averages."),
    ("concept_roc_auc", "What ROC-AUC means", "ROC-AUC measures how well the model ranks positive cases above negative ones across all thresholds. 0.5 is no better than chance; 1.0 is perfect ranking. It is unavailable when the model cannot output probabilities or the test set lacks a class."),
    ("concept_confusion", "How to read a confusion matrix", "A confusion matrix counts test rows by actual class (rows) and predicted class (columns). The diagonal holds correct predictions; off-diagonal cells show which classes are confused with each other."),
    ("concept_regression", "What MAE, MSE, RMSE and R² mean", "MAE is the average absolute prediction error in the target's units. MSE is the average squared error, and RMSE is its square root, which penalises large errors more than MAE. R² is the share of target variance explained on the test set; zero or negative means no better than predicting the mean."),
    ("concept_cv", "What cross-validation is", "Cross-validation splits the training data into k folds, trains on k−1 and scores on the remaining fold, rotating through all folds. The mean and standard deviation show how stable a model's score is. DataStory AI refits preprocessing inside each fold to avoid leakage."),
    ("concept_leakage", "What data leakage is", "Leakage happens when information from the test data influences training, for example scaling or imputing with statistics from the whole dataset. It inflates scores. DataStory AI splits first and fits all preprocessing on the training portion only."),
    ("concept_smote", "What SMOTE and class weighting do", "SMOTE creates artificial minority-class rows by interpolating between neighbours; class weighting makes errors on rare classes cost more. Both aim to improve minority recall. DataStory AI applies SMOTE only to training folds; validation and test data are never resampled."),
    ("concept_permutation", "What permutation importance is", "Permutation importance shuffles one feature in the held-out test set and measures how much the model's score changes. It is in score units, not percentages, can be negative, is affected by correlated features, and does not show causation."),
    ("concept_imputation", "Ways to handle missing values", "Median imputation is robust for skewed numeric columns, mean suits symmetric ones, mode suits categories, and KNN imputation fills a value from similar rows. Dropping rows is simplest but discards data. No method is guaranteed to improve a model; measure the effect."),
    ("concept_outliers", "How outliers are detected", "DataStory AI flags values below Q1 − 1.5×IQR or above Q3 + 1.5×IQR (Tukey's rule). A flagged value is unusual, not necessarily wrong. Options are to keep, cap at the fences, or remove."),
    ("concept_correlation", "What correlation shows", "Pearson correlation r ranges from −1 to 1 and measures linear association between two numeric columns. It does not capture non-linear relationships and never proves that one variable causes another."),
    ("concept_quality", "How data-quality scores are defined", "Completeness is non-missing cells over all cells. Uniqueness is non-duplicate rows over all rows. Validity and consistency are only computed from rules you configure; otherwise they are reported as not evaluated. " + HEALTH_FORMULA),
    ("concept_tuning", "What hyperparameter tuning does", "Grid search tries every combination in a small parameter grid; randomized search samples a fixed number of combinations. Each candidate is scored with cross-validation on the training data, and the best one is refitted."),
]


def _doc(i, title, text, source, kind="analysis"):
    return {"id": i, "title": title, "text": text, "source": source, "kind": kind}


def build_documents(results: dict[str, Any]) -> list[dict[str, Any]]:
    p, q, ds = results["profile"], results["quality"], results["dataset"]
    dims = q["dimensions"]
    docs = []
    label = f" {SYNTHETIC_LABEL}. {ds['provenance']}" if ds.get("synthetic") else f" Provenance: {ds['provenance']}"
    kc = p["kind_counts"]
    docs.append(_doc("overview", "Dataset overview",
                     f"Dataset '{ds['name']}' has {p['rows']:,} rows and {p['columns']:,} columns: {kc['numeric']} numeric, "
                     f"{kc['categorical']} categorical, {kc['datetime']} datetime and {kc['text']} text columns."
                     f"{label}", "Profiler"))
    health = q["health"]
    found = []
    if p["missing_cells"]:
        found.append(f"{p['missing_cells']:,} missing cells in {p['columns_with_missing']} column(s)")
    if p["duplicate_rows"]:
        found.append(f"{p['duplicate_rows']:,} duplicate rows")
    if p["constant_columns"]:
        found.append(f"{len(p['constant_columns'])} constant column(s)")
    if p["outlier_columns"]:
        found.append(f"{len(p['outlier_columns'])} column(s) with outliers by Tukey's rule")
    issues = "; ".join(found) or "none detected by the profiler"
    docs.append(_doc("quality", "Data quality assessment",
                     f"Completeness is {format_dimension(dims['completeness'])}; uniqueness is {format_dimension(dims['uniqueness'])}. "
                     f"Validity: {format_dimension(dims['validity']).rstrip('.')}. Consistency: {format_dimension(dims['consistency']).rstrip('.')}. "
                     f"Main issues found: {issues}. "
                     + (f"Data health score: {health['value']:.1f} out of 100, computed from {', '.join(health['included'])}."
                        if health.get("value") is not None else "Data health score: unavailable."),
                     "Quality metrics"))
    miss = sorted([c for c in p["column_details"] if c["missing"]], key=lambda c: -c["missing"])
    docs.append(_doc("missing", "Missing values",
                     (f"{p['missing_cells']:,} cells are missing ({p['missing_pct']:.2f}% of all cells). Columns with the most missing values: "
                      + "; ".join(f"{c['name']} {c['missing']:,} ({c['missing_pct']:.1f}%)" for c in miss[:10]) + ".")
                     if miss else "There are no missing values in any column.", "Profiler · missing-value counts"))
    docs.append(_doc("duplicates", "Duplicate rows",
                     f"{p['duplicate_rows']:,} exact duplicate rows were found ({(p['duplicate_pct'] or 0):.2f}% of rows).",
                     "Profiler · duplicate detection"))
    structure = []
    if p["constant_columns"]:
        structure.append("Constant columns: " + ", ".join(p["constant_columns"]) + ".")
    if p["near_constant_columns"]:
        structure.append("Near-constant columns: " + ", ".join(p["near_constant_columns"]) + ".")
    if p["id_candidates"]:
        structure.append("Identifier-like columns: " + ", ".join(p["id_candidates"]) + ".")
    docs.append(_doc("structure", "Constant and identifier columns",
                     " ".join(structure) or "No constant, near-constant or identifier-like columns were detected.", "Profiler · cardinality"))
    oc = p["outlier_columns"]
    docs.append(_doc("outliers", "Outliers",
                     ("Outliers by Tukey's 1.5×IQR rule: " + "; ".join(f"{r['column']} {r['outliers']:,} ({r['outlier_pct']:.1f}%)" for r in oc[:10]) + ".")
                     if oc else "No numeric column has values outside the Tukey 1.5×IQR fences.", "Profiler · Tukey rule"))
    pairs = results["correlations"]["pairs"]
    docs.append(_doc("correlations", "Correlated features",
                     ("Strongest Pearson correlations between numeric columns (association, not causation): "
                      + "; ".join(f"{x['a']} and {x['b']} r = {x['r']:.3f}" for x in pairs[:8]) + ".")
                     if pairs else "Correlations were not computed because fewer than two numeric columns with variation exist.",
                     "Correlation matrix"))
    t = p.get("target")
    if t and t.get("class_counts"):
        docs.append(_doc("balance", "Target class balance and imbalance",
                         f"Target '{t['name']}' class counts: " + "; ".join(f"{k} {v:,}" for k, v in list(t["class_counts"].items())[:12])
                         + f". Largest to smallest class ratio is {t['imbalance_ratio']:.2f} to 1; the smallest class is {t['minority_share_pct']:.1f}% of rows. "
                         + ("The dataset is imbalanced." if t["imbalance_ratio"] >= 1.5 else "The dataset is not notably imbalanced."),
                         "Profiler · target class counts"))
    elif t:
        docs.append(_doc("balance", "Target summary", f"Target '{t['name']}' is continuous with {t['unique']:,} distinct values, so class imbalance does not apply.", "Profiler"))
    else:
        docs.append(_doc("balance", "Target class balance and imbalance", "No target column has been selected yet, so whether the dataset is imbalanced cannot be determined. Select a target on the Machine Learning page.", "Application state"))

    cl = results["cleaning"]
    if cl["history"]:
        docs.append(_doc("cleaning_history", "Cleaning steps applied",
                         f"{len(cl['history'])} cleaning step(s) were applied: " + " ".join(f"({i}) {h['label']}: {h['detail']}" for i, h in enumerate(cl["history"], 1))
                         + f" Rows went from {cl['original']['rows']:,} to {cl['current']['rows']:,}.", "Transformation history"))
    docs.append(_doc("cleaning_suggestions", "Cleaning steps to consider",
                     ("Suggested cleaning steps to consider based on the profile: " + " ".join(cl["suggestions"][:10]))
                     if cl["suggestions"] else "The profile shows no missing values, duplicates, constant columns or flagged outliers, so no cleaning step is specifically suggested.",
                     "Profiler-based suggestions"))

    for c in p["column_details"][:80]:
        txt = f"Column '{c['name']}' is {c['kind']} (dtype {c['dtype']}) with {c['unique']:,} distinct values and {c['missing']:,} missing ({(c['missing_pct'] or 0):.1f}%)."
        if c.get("mean") is not None:
            txt += f" Mean {c['mean']:.4g}, median {c['median']:.4g}, standard deviation {(c['std'] or 0):.4g}, minimum {c['min']:.4g}, maximum {c['max']:.4g}."
            if c.get("outliers") is not None:
                txt += f" {c['outliers']:,} outliers by Tukey's rule."
        elif c.get("top_value") is not None:
            txt += f" Most frequent value '{c['top_value']}' appears {c['top_count']:,} times."
        docs.append(_doc(f"col::{c['name']}", f"Column {c['name']}", txt, "Profiler · column statistics"))

    m = results.get("modeling")
    if m:
        ok = [x for x in m["models"] if "error" not in x]
        best = ok[0]
        docs.append(_doc("model_setup", "Modelling methodology",
                         f"Task: {m['task']} predicting '{m['target']}'. {m['n_train']:,} training rows and {m['n_test']:,} held-out test rows "
                         f"(test size {m['test_size']:.0%}, seed {m['seed']}). {m['preprocessing']} Cross-validation folds: {m['cv_folds'] or 'none'}. "
                         f"Tuning: {m['tuning']}. Imbalance handling: {m['imbalance']}.", "Model training configuration"))
        lines = []
        for x in ok:
            ms = ", ".join(f"{METRIC_LABELS[k]} {x['metrics'][k]:.4f}" for k in m["metric_names"] if x["metrics"].get(k) is not None)
            lines.append(f"{x['name']}: {ms}" + (f", cross-validated {METRIC_LABELS[x['cv']['metric']]} {x['cv']['mean']:.4f} ± {x['cv']['std']:.4f}" if x["cv"] else ""))
        docs.append(_doc("leaderboard", "Model performance and best model",
                         f"Which model performed best: '{best['name']}', ranked by {m['ranking_basis']}. Test-set results — " + "; ".join(lines) + ".",
                         "Model leaderboard · held-out test set"))
    else:
        docs.append(_doc("leaderboard", "Model performance and best model",
                         "No model has been trained yet, so which model performed best and its evaluation metrics cannot be determined. Train models on the Machine Learning page.",
                         "Application state"))
    imp = results.get("importance")
    if imp:
        docs.append(_doc("importance", "Feature importance — which features influenced predictions",
                         f"Permutation importance for '{imp['model']}' on the {imp['evaluated_on']}, scored by {imp['metric_label']} over {imp['n_repeats']} repeats. "
                         "Mean change in score when shuffled (± standard deviation): "
                         + "; ".join(f"{f['feature']} {f['mean']:+.4f} ± {f['std']:.4f}" for f in imp["features"][:10]) + ". "
                         "These are score changes, not percentage contributions, and do not show causation.",
                         "Permutation importance · held-out test set"))
    else:
        docs.append(_doc("importance", "Feature importance — which features influenced predictions",
                         "Feature importance unavailable — a valid fitted model and evaluation dataset are required. Train a model, then compute permutation importance on the Model Results page.",
                         "Application state"))
    docs += [_doc(i, t, x, "Method glossary", "concept") for i, t, x in CONCEPTS]
    return docs


class Retriever:
    def __init__(self, docs: list[dict[str, Any]]):
        self.docs = docs
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vec.fit_transform([f"{d['title']}. {d['title']}. {d['text']}" for d in docs])

    def search(self, query: str, k: int = 4, min_score: float = 0.05) -> list[tuple[dict[str, Any], float]]:
        if not query.strip():
            return []
        sims = cosine_similarity(self.vec.transform([query]), self.matrix)[0]
        order = np.argsort(-sims)[:k]
        return [(self.docs[i], float(sims[i])) for i in order if sims[i] >= min_score]
