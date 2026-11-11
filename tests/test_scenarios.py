"""End-to-end acceptance scenarios A–E from the upgrade specification.

The real Netflix titles file is not bundled (licence), so Scenario A uses a
generated frame with the same column layout (tests/fixtures.py).
"""
from pathlib import Path

import nbformat
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from ai import copilot
from core import charts as ch
from core import cleaning, clustering, comparison, modeling, notebook, overview, profiler, quality_metrics, reporting, session_io, targets
from core import data_loader as dl
from core import explainability as ex
from core.results import build_results
from fixtures import netflix_like

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _execute(nb_bytes: bytes, workdir: Path, csv: bytes, name: str) -> str:
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    nb = nbformat.reads(nb_bytes.decode("utf-8"), as_version=4)
    (workdir / "data").mkdir(exist_ok=True)
    (workdir / "data" / name).write_bytes(csv)
    nbclient.NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(workdir)}}).execute()
    return "\n".join(o.get("text", "") for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "stream")


def test_scenario_a_netflix(tmp_path):
    raw = dl.to_csv_bytes(netflix_like(8807))
    df, meta = dl.load_upload("netflix_titles.csv", raw)
    assert meta.provenance_category == dl.PROV_UPLOAD and not meta.synthetic  # 2. provenance
    p = profiler.profile_dataset(df)
    assert (p["rows"], p["columns"]) == (8807, 12)  # 3. statistics
    assert "show_id" in p["id_candidates"]  # 4. identifier
    tg = targets.assess_targets(df, p["kinds"])
    sid = next(c for c in tg["candidates"] if c["column"] == "show_id")
    assert sid["verdict"] == "blocked" and tg["best"] != "show_id"  # 5. not a target
    assert not targets.validate_target(df, "show_id", None, p["kinds"])["ok"]
    recs = ch.recommend(df, p, limit=8)  # 6. meaningful charts
    assert {"type", "release_year", "date_added"} <= {c for r in recs for c in r["columns"]}
    assert p["kinds"]["date_added"] == "datetime"  # 7. dates
    new, rec = cleaning.preview_operation(df, "handle_outliers", {"columns": ["release_year"], "action": "cap"})  # 8-9
    assert rec["impact"]["cells_changed"] == rec["flagged"] and rec["impact"]["cells_changed"] < len(df)
    cmp = comparison.compare(df, new)  # 10
    assert cmp["impact"]["cells_changed"] == rec["impact"]["cells_changed"]
    r = build_results(new, meta, original=df, history=[rec])
    a = copilot.ask("Compare movie and TV show counts by country", copilot.Ctx(new, r, r["profile"]["kinds"]), use_llm=False)  # 11
    assert a["tool"] == "make_chart" and a["spec"]["chart"] == "stacked_bar"
    charts = [{"id": "c1", "title": "Titles by type", "spec": ch.ChartSpec("bar", x="type").to_dict(), "in_report": True}]
    r = build_results(new, meta, original=df, history=[rec], charts=charts)
    secs = reporting.build_report(r, new, None, {"mode": "Exploratory data analysis report"})  # 12
    assert reporting.to_pdf(r, sections=secs).startswith(b"%PDF")
    md = reporting.to_markdown(r, sections=secs)
    assert "Titles by type" in md and "provenance unverified" in md
    ctx = {"dataset": meta.to_dict(), "history": [rec], "expected_hash": dl.frame_hash(new), "expected_shape": list(new.shape),
           "charts": [{"title": c["title"], "spec": c["spec"]} for c in charts], "findings": r["insights"]}
    out = _execute(notebook.notebook_bytes(ctx), tmp_path, raw, Path(notebook.default_data_path(ctx)).name)  # 13
    assert "hash match" in out


def test_scenario_b_churn_classification():
    df, meta = dl.load_sample("churn")
    assert meta.synthetic  # 2
    tg = targets.assess_targets(df)
    assert tg["best"] == "Churn"  # 3
    p = profiler.profile_dataset(df, "Churn")
    assert p["target"]["imbalance_ratio"] > 1  # 4
    from core import ml_readiness

    rd = ml_readiness.assess_readiness(df, "Churn", [c for c in df.columns if c not in ("Churn", "CustomerID")], "classification")
    assert rd["status"] != "blocked" and rd["recommendations"]["metric"] in ("f1", "roc_auc", "accuracy")  # 5
    out = modeling.train_models(df, "Churn", [c for c in df.columns if c not in ("Churn", "CustomerID")], "classification",
                                ["Logistic Regression", "Random Forest"], cv_folds=3, primary_metric="f1")  # 6
    best = out.results["models"][0]
    d = best["diagnostics"]  # 7
    assert "pr_curve" in str(d.keys()) or any("pr" in k for k in d)
    imp = ex.held_out_permutation_importance(out.pipelines[best["name"]], out.X_test, out.y_test, "classification", "f1", 2, 5, 42,
                                             best["name"])  # 8
    assert imp["evaluated_on"].startswith("held-out") or "test" in imp["evaluated_on"]
    r = build_results(df, meta, target="Churn", modeling=out.results, importance=imp)
    assert any(i["category"] in ("Modeling", "Model") or "model" in i["title"].lower() for i in r["insights"])  # 9
    secs = reporting.build_report(r, df, None, {"mode": "Machine learning report"})  # 10
    md = reporting.to_markdown(r, sections=secs)
    assert best["name"] in md and "Synthetic Dataset" in md


def test_scenario_c_regression(tmp_path):
    df, meta = dl.load_sample("bike")
    assert targets.assess_targets(df)["best"] == "cnt"
    feats = [c for c in df.columns if c != "cnt"]
    out = modeling.train_models(df, "cnt", feats, "regression", ["Linear Regression", "Random Forest Regressor"], cv_folds=3)
    m = out.results["models"][0]
    assert "rmse" in m["metrics"] and m["diagnostics"]  # actual vs predicted / residuals stored
    cfg = {"target": "cnt", "task": "regression", "features": feats, "models": ["Linear Regression", "Random Forest Regressor"],
           "primary": "r2", "test_size": 0.2, "cv": 3, "seed": 42, "tuning": "none", "imbalance": "none", "scale": True}
    ctx = {"dataset": meta.to_dict(), "history": [], "expected_hash": dl.frame_hash(df), "expected_shape": list(df.shape),
           "train_config": cfg, "training": out.results}
    text = _execute(notebook.notebook_bytes(ctx), tmp_path, dl.to_csv_bytes(df), Path(notebook.default_data_path(ctx)).name)
    assert "metrics match" in text


def test_scenario_d_unsupervised():
    df = netflix_like(1500)[["type", "release_year", "rating", "duration"]].copy()
    df["duration_min"] = df["duration"].str.extract(r"(\d+)").astype(float)
    df = df.drop(columns=["duration"])
    p = profiler.profile_dataset(df)
    o = overview.build_overview(df, p, quality_metrics.assess_quality(df))
    assert o["clustering_suitable"] and any(r["title"] == "Find segments" for r in o["recommendations"])  # 2
    cand = clustering.cluster_feature_candidates(df, p["kinds"])  # 3
    res = clustering.run_clustering(df, cand["usable"], "kmeans", {"n_clusters": 3}, seed=7)  # 4
    assert res["n_clusters"] == 3 and res["metrics"]["silhouette"] is not None  # 5
    assert res["profiles"] and res["descriptions"] and res["projection"]  # 6
    r = build_results(df, dl.DatasetMeta("segments.csv", "upload"), clustering={"algorithm": "kmeans", **{k: res[k] for k in ("n_clusters", "sizes", "metrics")},
                                                                                 "noise_count": 0, "descriptions": res["descriptions"],
                                                                                 "features": cand["usable"], "config": res["config"]})
    md = reporting.to_markdown(r, sections=reporting.build_report(r, df, None, {"sections": ["Clustering and segments"]}))  # 7
    assert "Cluster" in md and "not business personas" in md
    labelled = clustering.labels_frame(df, res, {"0": "Group A"})
    assert "cluster" in labelled.columns and "Group A" in set(labelled["cluster"])


def test_scenario_e_session_restoration():
    df, meta = dl.load_sample("churn")
    new, rec = cleaning.preview_operation(df, "drop_columns", {"columns": ["Gender"]})
    out = modeling.train_models(new, "Churn", None, "classification", ["Logistic Regression"], cv_folds=0)
    snap = {"raw_df": df, "df": new, "meta": meta, "history": [rec], "rules": [], "completeness_exclude": [], "target": "Churn",
            "train_config": {"target": "Churn", "task": "classification", "features": [], "models": ["Logistic Regression"], "primary": "f1",
                             "test_size": 0.2, "cv": 0, "seed": 42, "tuning": "none", "imbalance": "none", "scale": True},
            "training": out, "importance": None, "clustering": None,
            "charts": [{"id": "c1", "title": "Churn by contract", "spec": ch.ChartSpec("bar", x="Contract").to_dict(), "created_at": "",
                        "data_version": "v1", "in_report": True}],
            "chat": [{"role": "user", "content": "hi"}], "narrative": None, "report_config": None, "domain_override": None}
    pkg = session_io.export_session(snap)  # 5
    restored = session_io.import_session(pkg)  # 6-7
    st_ = restored["state"]
    assert dl.frame_hash(st_["df"]) == dl.frame_hash(dl.load_upload("x.csv", dl.to_csv_bytes(new))[0])  # 8
    assert st_["target"] == "Churn" and len(st_["history"]) == 1 and len(st_["charts"]) == 1
    assert st_["training"] is None and any("retrain" in x.lower() for x in restored["report"]["not_restored"])  # 9


def test_app_renders_netflix_like_upload_on_every_page():
    """Inject an uploaded Netflix-shaped dataset into a real app session and render every page."""
    from components.navigation import STEPS

    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    df, meta = dl.load_upload("netflix_titles.csv", dl.to_csv_bytes(netflix_like(1200)))
    at.session_state["raw_df"] = df
    at.session_state["df"] = df
    at.session_state["meta"] = meta
    at.session_state["_version"] = 1
    for step in STEPS:
        at.switch_page(step.path)
        at.run()
        assert not at.exception, (step.path, [e.value for e in at.exception])
    at.switch_page("views/eda.py")
    at.session_state["_eda_tab"] = "Chart builder"
    at.run()
    at.text_input(key="nl_q").set_value("Compare movie and TV show counts by country")
    next(b for b in at.button if "Generate" in b.label).click()
    at.run()
    assert not at.exception and at.session_state["_nl_spec"]["chart"] == "stacked_bar"
