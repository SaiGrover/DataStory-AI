"""Held-out diagnostics stored by train_models, and the figure builders."""
import json

import numpy as np

from core import data_loader as dl
from core import diagnostics as dg
from core import modeling


def _train(sample, target, models, cv=3, **kw):
    df, _ = dl.load_sample(sample)
    return modeling.train_models(df, target, model_names=models, cv_folds=cv, **kw)


def test_binary_classification_diagnostics():
    out = _train("titanic", "Survived", ["Logistic Regression", "Decision Tree"])
    res = out.results
    json.dumps(res)  # everything is serialisable
    for m in res["models"]:
        d = m["diagnostics"]
        assert d["evaluated_on"] == "held-out test set" and d["n_test"] == res["n_test"]
        assert {"confusion_matrix_normalized", "pr_curve", "calibration", "probability_hist", "errors"} <= set(d)
        cm = np.array(d["confusion_matrix_normalized"])
        assert np.allclose(cm.sum(axis=1), 1)
        assert m["cv"]["scores"] and len(m["cv"]["scores"]) == m["cv"]["folds"]
        assert abs(np.mean(m["cv"]["scores"]) - m["cv"]["mean"]) < 1e-9
        assert "not the held-out" in m["cv"]["evaluated_on"] and m["metrics_evaluated_on"].startswith("held-out")
        assert 0 <= d["pr_curve"]["average_precision"] <= 1
        assert d["errors"]["n_misclassified"] == res["n_test"] - int(np.trace(np.array(m["confusion_matrix"])))
    m = res["models"][0]
    d = m["diagnostics"]
    for fig in (dg.fig_confusion(m["confusion_matrix"], res["classes"]), dg.fig_confusion(d["confusion_matrix_normalized"], res["classes"], True),
                dg.fig_roc(m["roc_curve"], m["metrics"]["roc_auc"], m["name"]), dg.fig_pr(d["pr_curve"], m["name"]),
                dg.fig_calibration(d["calibration"]), dg.fig_probability(d["probability_hist"]), dg.fig_cv_scores(res["models"], "F1")):
        assert fig is not None


def test_multiclass_has_no_binary_only_diagnostics():
    out = _train("iris", "species", ["Logistic Regression"])
    d = out.results["models"][0]["diagnostics"]
    assert "pr_curve" not in d and "calibration" not in d
    assert set(d["pr_auc_ovr"]) == set(out.results["classes"])
    assert dg.fig_confidence(d["confidence_hist"]) is not None


def test_regression_diagnostics_and_tuned_cv_scores():
    out = _train("student", "G3", ["Ridge Regression"], tuning="random")
    m = out.results["models"][0]
    d = m["diagnostics"]
    assert len(d["residuals"]["predicted"]) == out.results["n_test"]
    assert d["abs_error_quantiles"]["0.5"] <= d["abs_error_quantiles"]["0.9"]
    assert d["error_by_subgroup"] and all(g["n"] >= dg.MIN_SUBGROUP for x in d["error_by_subgroup"] for g in x["groups"])
    assert len(m["cv"]["scores"]) == m["cv"]["folds"]
    for fig in (dg.fig_actual_vs_predicted(m["predictions"], "G3"), dg.fig_residual_hist(m["predictions"]),
                dg.fig_residuals_vs_predicted(d), dg.fig_abs_error(d), dg.fig_error_by_group(d["error_by_subgroup"][0])):
        assert fig is not None


def test_no_cv_means_no_scores_and_figure_is_none():
    out = _train("iris", "species", ["Decision Tree"], cv=0)
    assert out.results["models"][0]["cv"] is None
    assert dg.fig_cv_scores(out.results["models"], "F1") is None
    assert dg.fig_pr(None, "x") is None and dg.fig_calibration(None) is None


def test_smote_pipeline_still_resamples_training_only():
    out = _train("churn", "Churn", ["Logistic Regression"], imbalance="smote")
    pipe = out.pipelines["Logistic Regression"]
    assert "smote" in pipe.named_steps
    # the stored test set is untouched by resampling
    assert len(out.y_test) == out.results["n_test"]
    assert out.results["models"][0]["diagnostics"]["n_test"] == out.results["n_test"]
