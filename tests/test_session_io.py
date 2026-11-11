"""Session package export / import: round trip, integrity checks and malicious packages."""
import io
import json
import zipfile

import pytest

from core import cleaning, clustering, modeling, session_io
from core import data_loader as dl
from core.charts import ChartSpec


def _snap(include_model=True):
    raw, meta = dl.load_sample("churn")
    df, hist = raw, []
    num = [c for c in raw.columns if raw[c].dtype.kind in "if" and c != "Churn"]
    for op, params in [("impute", {"columns": [c for c in raw.columns if raw[c].isna().any() and raw[c].dtype.kind in "if"] or num[:1],
                                   "strategy": "median"}),
                       ("drop_duplicates", {}),
                       ("handle_outliers", {"columns": num[:2], "method": "iqr", "threshold": 1.5, "action": "remove"})]:
        df, rec = cleaning.preview_operation(df, op, params)
        hist.append(rec)
    training, cfg = None, None
    if include_model:
        training = modeling.train_models(df, "Churn", model_names=["Logistic Regression"], cv_folds=0)
        cfg = {"target": "Churn", "task": "classification", "features": [], "models": ["Logistic Regression"], "primary": "f1",
               "test_size": 0.2, "cv": 0, "seed": 42, "tuning": "none", "imbalance": "none", "scale": True}
    cl = clustering.run_clustering(df, num[:3], "kmeans", {"n_clusters": 3})
    charts = [{"id": "c1", "title": "Churn by Contract", "spec": ChartSpec("bar", x="Churn").to_dict(),
               "created_at": "now", "data_version": 3, "in_report": True}]
    chat = [{"role": "user", "content": "hi"},
            {"role": "assistant", "answer": "Hello", "figure": object(), "table": object(), "sources": ["x"]}]
    return {"raw_df": raw, "df": df, "meta": meta, "history": hist, "rules": [{"type": "range", "column": num[0], "min": 0}],
            "completeness_exclude": [], "target": "Churn", "train_config": cfg, "training": training,
            "importance": {"model": "x"} if include_model else None,
            "clustering": {"result": cl, "names": {"0": "A"}, "version": 4}, "charts": charts, "chat": chat,
            "narrative": None, "report_config": {"mode": "executive"}, "domain_override": None}


@pytest.fixture(scope="module")
def snap():
    return _snap()


def test_round_trip_restores_state(snap):
    pkg = session_io.export_session(snap)
    info = session_io.inspect_package(pkg)
    assert info["include_data"] and info["contents"]["cleaning_steps"] == 3
    out = session_io.import_session(pkg)
    st, rep = out["state"], out["report"]
    assert dl.frame_hash(st["raw_df"]) == dl.frame_hash(session_io._roundtrip(snap["raw_df"]))
    assert dl.frame_hash(st["df"]) == dl.frame_hash(cleaning.replay(st["raw_df"], snap["history"]))
    assert st["df"].shape == snap["df"].shape
    assert [h["operation"] for h in st["history"]] == ["impute", "drop_duplicates", "handle_outliers"]
    assert st["target"] == "Churn" and len(st["charts"]) == 1 and st["rules"]
    assert st["training"] is None and st["previous_training_results"]["best_model"] == "Logistic Regression"
    assert session_io.RETRAIN_NOTE in rep["not_restored"] and session_io.IMPORTANCE_NOTE in rep["not_restored"]
    assert st["clustering"]["result"]["n_clusters"] == 3
    assert st["chat"][1]["answer"] == "Hello" and "figure" not in st["chat"][1]
    assert st["meta"].provenance_category == snap["meta"].provenance_category
    assert any("Restored from session package" in n for n in st["meta"].notes)
    assert not rep["warnings"]


def test_export_without_data_requires_matching_dataset(snap):
    pkg = session_io.export_session(snap, include_data=False)
    with zipfile.ZipFile(io.BytesIO(pkg)) as z:
        assert session_io.DATA not in z.namelist()
    with pytest.raises(session_io.SessionError, match="does not contain the dataset"):
        session_io.import_session(pkg)
    good = dl.to_csv_bytes(snap["raw_df"])
    out = session_io.import_session(pkg, good, "customer_churn.csv")
    assert out["state"]["df"].shape == snap["df"].shape
    other, _ = dl.load_sample("iris")
    with pytest.raises(session_io.SessionError, match="not the one this session was saved with"):
        session_io.import_session(pkg, dl.to_csv_bytes(other), "iris.csv")


def _rebuild(pkg, mutate):
    with zipfile.ZipFile(io.BytesIO(pkg)) as z:
        files = {n: z.read(n) for n in z.namelist()}
    mutate(files)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n, b in files.items():
            z.writestr(n, b)
    return buf.getvalue()


def test_tampered_state_rejected(snap):
    pkg = session_io.export_session(snap)

    def mut(f):
        st = json.loads(f["state.json"])
        st["target"] = "tenure"
        f["state.json"] = json.dumps(st).encode()
    with pytest.raises(session_io.SessionError, match="Checksum mismatch"):
        session_io.import_session(_rebuild(pkg, mut))


@pytest.mark.parametrize("name", ["evil.py", "../x", "model.pkl", "data/other.csv", "/abs.json"])
def test_unexpected_members_rejected(snap, name):
    pkg = session_io.export_session(snap)
    with pytest.raises(session_io.SessionError):
        session_io.import_session(_rebuild(pkg, lambda f: f.__setitem__(name, b"print('x')")))


def test_non_zip_and_oversized_rejected(snap, monkeypatch):
    with pytest.raises(session_io.SessionError, match="not a ZIP"):
        session_io.import_session(b"hello world")
    pkg = session_io.export_session(snap)
    monkeypatch.setattr(session_io, "MAX_PACKAGE_MB", 0.001)
    with pytest.raises(session_io.SessionError, match="larger than"):
        session_io.import_session(pkg)


def test_zip_bomb_guard(snap, monkeypatch):
    pkg = session_io.export_session(snap)
    monkeypatch.setattr(session_io, "MAX_UNCOMPRESSED_MB", 0.01)
    with pytest.raises(session_io.SessionError, match="exceed"):
        session_io.import_session(pkg)


def test_unsupported_schema_rejected(snap):
    pkg = session_io.export_session(snap)

    def mut(f):
        m = json.loads(f["manifest.json"])
        m["schema_version"] = 99
        f["manifest.json"] = json.dumps(m).encode()
    with pytest.raises(session_io.SessionError, match="Unsupported session schema"):
        session_io.import_session(_rebuild(pkg, mut))


def test_invalid_chart_spec_dropped_with_report(snap):
    s = dict(snap)
    s["charts"] = snap["charts"] + [{"id": "bad", "title": "Ghost", "spec": ChartSpec("bar", x="no_such_column").to_dict(),
                                     "created_at": "now", "data_version": 1, "in_report": False}]
    out = session_io.import_session(session_io.export_session(s))
    assert len(out["state"]["charts"]) == 1
    assert any("Ghost" in line for line in out["report"]["not_restored"])


def test_unknown_operation_rejected(snap):
    s = dict(snap)
    s["history"] = snap["history"] + [{"operation": "os.system", "params": {}, "label": "x", "detail": "x"}]
    with pytest.raises(session_io.SessionError, match="unknown operation"):
        session_io.import_session(session_io.export_session(s))


def test_compare_configs_detects_changes(snap):
    saved = session_io.import_session(session_io.export_session(snap))["state"]
    cur = dict(snap)
    cur["target"] = "tenure"
    cur["history"] = snap["history"][:1]
    items = {d["item"] for d in session_io.compare_configs(saved, cur)}
    assert {"Target", "Cleaning steps", "Cleaning operations"} <= items
    assert not session_io.compare_configs(saved, saved)
