"""Notebook export: structure, no app imports, and top-to-bottom execution that reproduces the app's results."""
from __future__ import annotations

import io
import re
import zipfile

import nbformat
import numpy as np
import pytest

from core import cleaning, clustering, modeling
from core import data_loader as dl
from core import explainability as ex
from core import notebook as nbx
from core.charts import ChartSpec


def _apply(df, steps):
    history = []
    for op, params in steps:
        df, rec = cleaning.preview_operation(df, op, params)
        history.append(rec)
    return df, history


def _ctx(meta, raw, cleaned, history, charts=(), training=None, train_config=None, importance=None, cl=None, findings=None):
    return {"dataset": meta.to_dict(), "history": history, "expected_hash": dl.frame_hash(cleaned),
            "expected_shape": list(cleaned.shape), "charts": [{"title": s.title or s.chart, "spec": s.to_dict()} for s in charts],
            "training": training, "train_config": train_config, "importance": importance, "clustering": cl,
            "findings": findings or [{"title": "Example", "observation": "Recorded observation.", "limitation": "Example limitation."}]}


def _churn_case():
    base, _ = dl.load_sample("churn")
    rng = np.random.default_rng(0)
    base = base.copy()
    base.loc[rng.choice(len(base), 40, replace=False), "TotalCharges"] = np.nan
    base.loc[rng.choice(len(base), 25, replace=False), "Contract"] = base["Contract"].str.upper()
    base = base.iloc[list(range(len(base))) + list(range(15))].reset_index(drop=True)  # 15 duplicate rows
    raw, meta = dl.load_upload("churn_upload.csv", dl.to_csv_bytes(base))
    cleaned, history = _apply(raw, [
        ("impute", {"columns": ["TotalCharges"], "strategy": "median"}),
        ("handle_outliers", {"columns": ["MonthlyCharges", "TotalCharges"], "method": "iqr", "threshold": 1.0, "action": "cap"}),
        ("drop_duplicates", {"columns": [], "keep": "first"}),
        ("normalize_categories", {"columns": ["Contract"], "case": "title", "strip": True, "mapping": {}}),
        ("encode", {"columns": ["Gender"], "method": "onehot", "drop_first": False}),
    ])
    charts = [ChartSpec("bar", x="Contract", y="Churn", agg="rate", title="Churn rate by contract"),
              ChartSpec("histogram", x="Tenure", title="Tenure distribution")]
    cfg = {"target": "Churn", "task": "classification", "features": None, "models": ["Logistic Regression", "Decision Tree"],
           "primary": "f1", "test_size": 0.2, "cv": 3, "seed": 7, "tuning": "none", "imbalance": "none", "scale": True}
    out = modeling.train_models(cleaned, "Churn", None, "classification", cfg["models"], 0.2, 7, 3, "none", "none", True, "f1")
    imp = ex.held_out_permutation_importance(out.pipelines["Decision Tree"], out.X_test, out.y_test, "classification", "f1",
                                             2, 3, 7, "Decision Tree")
    cl = clustering.run_clustering(cleaned, ["Tenure", "MonthlyCharges", "Contract"], "kmeans", {"n_clusters": 3}, seed=3)
    return raw, meta, cleaned, _ctx(meta, raw, cleaned, history, charts, out.results, cfg, imp,
                                    {"config": cl["config"], "n_clusters": cl["n_clusters"], "sizes": cl["sizes"]})


def _bike_case():
    raw, meta = dl.load_sample("bike")
    cleaned, history = _apply(raw, [
        ("drop_missing_rows", {"columns": [], "how": "any"}),
        ("convert_type", {"columns": ["dteday"], "to": "datetime"}),
        ("scale", {"columns": ["temp", "hum"], "method": "standard"}),
    ])
    feats = ["dteday", "season", "temp", "hum", "windspeed", "weathersit"]
    cfg = {"target": "cnt", "task": "regression", "features": feats, "models": ["Ridge Regression", "Decision Tree Regressor"],
           "primary": "rmse", "test_size": 0.25, "cv": 3, "seed": 1, "tuning": "random", "imbalance": "none", "scale": True}
    out = modeling.train_models(cleaned, "cnt", feats, "regression", cfg["models"], 0.25, 1, 3, "random", "none", True, "rmse")
    return raw, meta, cleaned, _ctx(meta, raw, cleaned, history, [ChartSpec("line", x="dteday", y="cnt", agg="sum", freq="MS")],
                                    out.results, cfg)


def _iris_case():
    raw, meta = dl.load_sample("iris")
    charts = [ChartSpec("bar", x="species"), ChartSpec("box", x="species", y="petal length (cm)"),
              ChartSpec("heatmap", columns=[c for c in raw.columns if c != "species"]),
              ChartSpec("scatter", x="sepal length (cm)", y="petal length (cm)", color="species",
                        filters=[{"column": "species", "op": "!=", "value": "setosa"}])]
    return raw, meta, raw, _ctx(meta, raw, raw, [], charts)


def _code(nb):
    return "\n".join(c.source for c in nb.cells if c.cell_type == "code")


@pytest.mark.parametrize("case", [_churn_case, _bike_case, _iris_case])
def test_structure_valid_and_self_contained(case):
    raw, meta, cleaned, ctx = case()
    nb = nbx.build_notebook(ctx)
    nbformat.validate(nb)
    code = _code(nb)
    assert not re.search(r"^\s*(from|import)\s+core\b", code, re.M)
    assert "DATA_PATH = Path(" in code
    md = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    for pkg in nbx.PACKAGES:
        assert pkg in md
    if meta.synthetic:
        assert "Synthetic Dataset" in md
    if meta.provenance_category == dl.PROV_UPLOAD:
        assert "provenance unverified" in md


def test_bundle_contents():
    raw, meta, cleaned, ctx = _iris_case()
    data = dl.to_csv_bytes(raw)
    z = zipfile.ZipFile(io.BytesIO(nbx.bundle_zip(ctx, data)))
    names = set(z.namelist())
    assert {"analysis.ipynb", "requirements.txt", "README.txt", "data/iris.csv"} <= names
    assert z.read("data/iris.csv") == data
    assert "scikit-learn==" in z.read("requirements.txt").decode()


def _execute(tmp_path, raw, ctx):
    pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    from nbclient import NotebookClient

    zipfile.ZipFile(io.BytesIO(nbx.bundle_zip(ctx, dl.to_csv_bytes(raw)))).extractall(tmp_path)
    nb = nbformat.read(tmp_path / "analysis.ipynb", as_version=4)
    NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(tmp_path)}}).execute()
    text = []
    for c in nb.cells:
        for o in c.get("outputs", []):
            if o.get("output_type") == "stream":
                text.append(o.get("text", ""))
            elif "data" in o and "text/plain" in o["data"]:
                text.append(o["data"]["text/plain"])
    return "\n".join(text)


def test_execute_churn_reproduces_app(tmp_path):
    raw, meta, cleaned, ctx = _churn_case()
    out = _execute(tmp_path, raw, ctx)
    assert "Data file matches" in out
    assert "hash match" in out and "MISMATCH" not in out
    assert "metrics match" in out
    assert "importance match" in out
    assert "cluster sizes match" in out


def test_execute_bike_regression(tmp_path):
    raw, meta, cleaned, ctx = _bike_case()
    out = _execute(tmp_path, raw, ctx)
    assert "hash match" in out and "MISMATCH" not in out
    assert "metrics match" in out


def test_execute_iris_charts_only(tmp_path):
    raw, meta, cleaned, ctx = _iris_case()
    out = _execute(tmp_path, raw, ctx)
    assert "hash match" in out
