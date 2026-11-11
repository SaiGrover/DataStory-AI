"""Clustering: preprocessing, metric availability, determinism, labels export and the page."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from core import clustering as cl
from core import data_loader as dl
from core.profiler import infer_kinds



@pytest.fixture
def iris():
    df, _ = dl.load_sample("iris")
    return df


def _num(df):
    return [c for c in df.columns if c != "species"]


def test_candidates_exclude_identifiers_and_text():
    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "row": np.arange(n),
        "note": [f"free text comment number {i} with plenty of words to be long enough" for i in range(n)],
        "when": pd.date_range("2024-01-01", periods=n),
        "const": 1,
        "x": rng.normal(size=n), "y": rng.normal(size=n),
        "grp": rng.choice(list("abc"), n),
    })
    out = cl.cluster_feature_candidates(df, infer_kinds(df))
    assert set(out["usable"]) == {"x", "y", "grp"}
    reasons = {e["column"]: e["reason"] for e in out["excluded"]}
    assert "identifier" in reasons["customer_id"] and "identifier" in reasons["row"]
    assert reasons["note"] == "free text" and "datetime" in reasons["when"] and "constant" in reasons["const"]


def test_kmeans_on_iris(iris):
    res = cl.run_clustering(iris, _num(iris), "kmeans", {"n_clusters": 3})
    assert res["n_clusters"] == 3 and sum(res["sizes"].values()) == len(iris)
    assert 0 < res["metrics"]["silhouette"] <= 1 and res["metrics"]["davies_bouldin"] > 0
    assert len(res["labels"]) == len(iris) and len(res["descriptions"]) == 3
    assert res["config"]["sklearn_version"] and res["projection"]["explained_variance"][0] > 0
    import json
    json.dumps(res)  # JSON-serialisable


def test_dbscan_degenerate_metrics_undefined(iris):
    one = cl.run_clustering(iris, _num(iris), "dbscan", {"eps": 50, "min_samples": 2})
    assert one["n_clusters"] == 1 and all(v is None for v in one["metrics"].values())
    assert any("fewer than two" in n for n in one["metric_notes"])
    noise = cl.run_clustering(iris, _num(iris), "dbscan", {"eps": 0.001, "min_samples": 5})
    assert noise["n_clusters"] == 0 and noise["noise_count"] == len(iris)
    assert all(v is None for v in noise["metrics"].values())


def test_agglomerative_and_gmm(iris):
    a = cl.run_clustering(iris, _num(iris), "agglomerative", {"n_clusters": 3, "linkage": "ward"})
    g = cl.run_clustering(iris, _num(iris), "gmm", {"n_components": 3})
    assert a["n_clusters"] == 3 and g["n_clusters"] == 3


def test_deterministic_with_seed(iris):
    a = cl.run_clustering(iris, _num(iris), "kmeans", {"n_clusters": 4}, seed=7)
    b = cl.run_clustering(iris, _num(iris), "kmeans", {"n_clusters": 4}, seed=7)
    assert a["labels"] == b["labels"]
    st = cl.stability(iris, _num(iris), "kmeans", {"n_clusters": 3}, n_runs=3)
    assert 0 <= st["ari_min"] <= 1


def test_labels_frame_names(iris):
    res = cl.run_clustering(iris, _num(iris), "kmeans", {"n_clusters": 2})
    lf = cl.labels_frame(iris, res, {"0": "Group A"})
    assert "cluster" in lf and set(lf["cluster"]) == {"Group A", "Cluster 1"}
    noise = cl.run_clustering(iris, _num(iris), "dbscan", {"eps": 0.001})
    assert set(cl.labels_frame(iris, noise)["cluster"]) == {"Noise"}


def test_suggest_k(iris):
    scan = cl.suggest_k(iris, _num(iris), 2, 5)
    assert [r["k"] for r in scan] == [2, 3, 4, 5] and all(r["silhouette"] is not None for r in scan)


def test_categorical_and_missing_values(clf_df):
    res = cl.run_clustering(clf_df, ["signal", "noise", "group"], "kmeans", {"n_clusters": 3})
    assert res["config"]["categorical"] == ["group"] and res["n_clusters"] == 3
    assert any("group=" in d for d in res["descriptions"]) or res["descriptions"]


def test_errors():
    df = pd.DataFrame({"x": range(10)})
    with pytest.raises(cl.ClusteringError):
        cl.run_clustering(df, [], "kmeans")
    with pytest.raises(cl.ClusteringError):
        cl.run_clustering(df, ["x"], "kmeans", {"n_clusters": 50})
    with pytest.raises(cl.ClusteringError):
        cl.run_clustering(df, ["x"], "nope")


_PAGE = """
import runpy, sys
sys.path.insert(0, {root!r})
from core import state, data_loader as dl
state.init()
if not state.has_data():
    state.set_dataset(*dl.load_sample("iris"))
runpy.run_path({page!r}, run_name="__main__")
"""


def test_clustering_page_renders():
    root = Path(__file__).resolve().parents[1]
    at = AppTest.from_string(_PAGE.format(root=str(root), page=str(root / "views" / "clustering.py")), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    next(b for b in at.button if b.label == "Run clustering").click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["clustering"]["result"]["n_clusters"] == 3
    next(b for b in at.button if b.label.startswith("Elbow")).click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
