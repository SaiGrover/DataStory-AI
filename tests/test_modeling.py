import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline as ImbPipeline

from core import explainability as ex
from core import modeling


def test_task_detection():
    assert modeling.detect_task(pd.Series(["a", "b"]))[0] == "classification"
    assert modeling.detect_task(pd.Series([0, 1, 1, 0]))[0] == "classification"
    assert modeling.detect_task(pd.Series(np.linspace(0, 1, 50)))[0] == "regression"


def test_train_test_separation(clf_df):
    out = modeling.train_models(clf_df, "label", cv_folds=0, model_names=["Logistic Regression"])
    assert set(out.X_train.index).isdisjoint(out.X_test.index)
    assert len(out.X_train) + len(out.X_test) == len(clf_df)
    assert out.results["n_test"] == len(out.X_test) == 80


def test_preprocessing_fitted_on_training_rows_only(reg_df):
    out = modeling.train_models(reg_df, "y", cv_folds=0, model_names=["Linear Regression"])
    scaler = out.pipelines["Linear Regression"].named_steps["preprocess"].named_transformers_["num"].named_steps["scale"]
    cols = out.results["features"]["numeric"]
    np.testing.assert_allclose(scaler.mean_, out.X_train[cols].mean().to_numpy())
    assert not np.allclose(scaler.mean_, reg_df[cols].mean().to_numpy())


def test_classification_metrics_only_for_classification(clf_df):
    r = modeling.train_models(clf_df, "label", cv_folds=3).results
    assert r["task"] == "classification" and r["metric_names"] == modeling.CLASSIFICATION_METRICS
    for m in r["models"]:
        assert set(m["metrics"]) == set(modeling.CLASSIFICATION_METRICS)
        assert np.array(m["confusion_matrix"]).sum() == r["n_test"]
    assert r["models"][0]["metrics"]["roc_auc"] > 0.8


def test_regression_metrics_only_for_regression(reg_df):
    r = modeling.train_models(reg_df, "y", cv_folds=3).results
    assert r["task"] == "regression"
    for m in r["models"]:
        assert set(m["metrics"]) == set(modeling.REGRESSION_METRICS)
        assert m["metrics"]["rmse"] == pytest.approx(np.sqrt(m["metrics"]["mse"]))
    assert r["models"][0]["metrics"]["r2"] > 0.9


def test_wrong_metric_for_task_is_rejected(reg_df):
    with pytest.raises(modeling.ModelingError):
        modeling.train_models(reg_df, "y", primary_metric="f1")


def test_smote_lives_inside_the_pipeline_and_test_set_is_untouched(clf_df):
    out = modeling.train_models(clf_df, "label", cv_folds=3, imbalance="smote", model_names=["Logistic Regression"])
    pipe = out.pipelines["Logistic Regression"]
    assert isinstance(pipe, ImbPipeline) and list(pipe.named_steps) == ["preprocess", "smote", "model"]
    assert list(pipe.named_steps).index("smote") > list(pipe.named_steps).index("preprocess")
    # test split still has its natural class mix (stratified, not resampled)
    counts = pd.Series(out.y_test).value_counts()
    assert counts.min() != counts.max()
    assert len(out.X_test) == out.results["n_test"]
    assert out.results["smote_k"] >= 1


def test_tuning_and_ranking_use_cross_validation(clf_df):
    r = modeling.train_models(clf_df, "label", cv_folds=3, tuning="random", model_names=["Decision Tree", "Logistic Regression"]).results
    assert r["ranked_by_cv"] and all(m["cv"] for m in r["models"])
    assert r["models"][0]["cv"]["mean"] >= r["models"][1]["cv"]["mean"]
    assert r["models"][0]["best_params"]


@pytest.mark.parametrize("df,target,match", [
    (pd.DataFrame({"x": range(30), "y": ["a"] * 30}), "y", "single class"),
    (pd.DataFrame({"x": range(5), "y": [0, 1, 0, 1, 0]}), "y", "at least"),
    (pd.DataFrame({"x": range(30), "y": [None] * 30}), "y", "at least"),
    (pd.DataFrame({"x": range(30), "y": range(30)}), "missing", "not in the dataset"),
])
def test_invalid_inputs_fail_gracefully(df, target, match):
    with pytest.raises(modeling.ModelingError, match=match):
        modeling.train_models(df, target)


def test_missing_targets_are_dropped_with_a_warning(reg_df):
    df = reg_df.copy()
    df.loc[:9, "y"] = np.nan
    r = modeling.train_models(df, "y", cv_folds=0, model_names=["Ridge Regression"]).results
    assert r["n_rows"] == len(df) - 10 and any("missing target" in w for w in r["warnings"])


# Credibility test 7 — reproducibility under fixed seeds
def test_training_is_reproducible(clf_df):
    a = modeling.train_models(clf_df, "label", cv_folds=3, seed=7).results
    b = modeling.train_models(clf_df, "label", cv_folds=3, seed=7).results
    assert [m["name"] for m in a["models"]] == [m["name"] for m in b["models"]]
    for ma, mb in zip(a["models"], b["models"]):
        assert ma["metrics"] == mb["metrics"] and ma["cv"] == mb["cv"]


# Credibility test 1 — importance comes from the fitted model and evaluation data
def test_permutation_importance_is_computed_not_hardcoded(clf_df):
    out = modeling.train_models(clf_df, "label", cv_folds=0, model_names=["Random Forest"])
    r = out.results
    imp = ex.held_out_permutation_importance(out.pipelines["Random Forest"], out.X_test, out.y_test, r["task"], "f1", 2, 5, 42, "Random Forest")
    feats = {f["feature"]: f for f in imp["features"]}
    assert set(feats) == set(out.X_test.columns)
    assert imp["features"][0]["feature"] == "signal" and feats["signal"]["mean"] > 0.1
    assert abs(feats["noise"]["mean"]) < feats["signal"]["mean"] / 3
    assert imp["n_test"] == len(out.X_test) and "held-out" in imp["evaluated_on"]
    # recomputing with sklearn directly on the same data gives the same numbers
    from sklearn.inspection import permutation_importance
    ref = permutation_importance(out.pipelines["Random Forest"], out.X_test, out.y_test, scoring="f1", n_repeats=5, random_state=42)
    assert feats["signal"]["mean"] == pytest.approx(ref.importances_mean[list(out.X_test.columns).index("signal")])
    # shuffling the labels destroys the signal -> importance changes accordingly
    y_perm = np.random.default_rng(0).permutation(np.asarray(out.y_test))
    imp2 = ex.held_out_permutation_importance(out.pipelines["Random Forest"], out.X_test, y_perm, r["task"], "f1", 2, 5, 42)
    assert {f["feature"]: f["mean"] for f in imp2["features"]}["signal"] != pytest.approx(feats["signal"]["mean"])


def test_negative_importances_are_kept(reg_df):
    out = modeling.train_models(reg_df, "y", cv_folds=0, model_names=["Random Forest Regressor"])
    imp = ex.held_out_permutation_importance(out.pipelines["Random Forest Regressor"], out.X_test, out.y_test, "regression", "r2", 2, 5)
    assert len(imp["features"]) == 3  # nothing filtered out, whatever the sign


def test_importance_unavailable_without_model_or_data(clf_df):
    with pytest.raises(ex.ImportanceUnavailable, match="Feature importance unavailable"):
        ex.held_out_permutation_importance(None, None, None, "classification", "f1")
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    with pytest.raises(ex.ImportanceUnavailable):
        ex.held_out_permutation_importance(Pipeline([("model", LogisticRegression())]), clf_df[["signal"]], clf_df["label"], "classification", "f1")
