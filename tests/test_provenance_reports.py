"""Provenance, formatting, overview/domain inference, evidence register, report builder and theme hand-off."""
import pytest

from core import cleaning, handoff, overview, profiler, quality_metrics, reporting
from core import data_loader as dl
from core.formatting import num, pct, share
from core.results import build_results
from fixtures import employee_attrition, housing, netflix_like, sales


# ------------------------------------------------------------ provenance
@pytest.mark.parametrize("sid", ["titanic", "churn", "bike", "student"])
def test_synthetic_samples_labelled(sid):
    _, m = dl.load_sample(sid)
    assert m.synthetic and m.provenance_category == dl.PROV_SYNTHETIC and m.label == dl.SYNTHETIC_LABEL


@pytest.mark.parametrize("sid", ["iris", "wine", "breast_cancer", "diabetes", "digits", "linnerud"])
def test_public_samples_verified(sid):
    _, m = dl.load_sample(sid)
    assert not m.synthetic and m.provenance_category == dl.PROV_PUBLIC and m.license and m.source_url


@pytest.mark.parametrize("name", ["titanic.csv", "synthetic_data.csv", "real_customers.csv"])
def test_uploads_never_inferred_from_filename(name):
    df, m = dl.load_upload(name, b"a,b\n1,2\n3,4\n")
    assert not m.synthetic and m.provenance_category == dl.PROV_UPLOAD and m.file_hash and m.original_filename == name


def test_provenance_survives_transformations_and_reaches_reports():
    df, m = dl.load_upload("data.csv", dl.to_csv_bytes(employee_attrition()))
    new, rec = cleaning.preview_operation(df, "drop_columns", {"columns": ["Age"]})
    r = build_results(new, m, original=df, history=[rec])
    assert r["lineage"]["original_category"] == dl.PROV_UPLOAD
    assert r["lineage"]["active_category"] == dl.PROV_DERIVED and r["data_version"] == "v1"
    md = reporting.to_markdown(r)
    assert "provenance unverified" in md and "Synthetic Dataset" not in md
    assert any(i["title"].startswith("User-uploaded") for i in r["insights"])


def test_meta_round_trip():
    _, m = dl.load_sample("iris")
    assert dl.DatasetMeta.from_dict(m.to_dict()).file_hash == m.file_hash


# ------------------------------------------------------------ formatting
def test_percentages_never_hide_small_nonzero_values():
    assert pct(0.03) == "<0.1%" and pct(0) == "0%" and pct(100) == "100%" and pct(99.97) == ">99.9%"
    assert pct(12.345) == "12.3%" and pct(None) == "Unavailable"
    assert share(1, 8807) == "<0.1%" and share(0, 10) == "0%"
    assert num(19720.4) == "19,720" and num(2021) == "2,021" and num(None) == "Unavailable"


def test_profile_text_uses_safe_rounding():
    from core import eda

    p = profiler.profile_dataset(netflix_like(3000))
    txt = eda.interpret_missing(p)
    assert "0.0%" not in txt


# -------------------------------------------------------------- overview
@pytest.mark.parametrize("frame,domain", [
    (netflix_like(800), "Entertainment & media"), (sales(), "Retail & sales"),
    (employee_attrition(), "Human resources"), (housing(), "Real estate"),
])
def test_domain_inference(frame, domain):
    assert overview.infer_domain(frame)["domain"] == domain


def test_unclear_domain_is_general():
    import pandas as pd

    df = pd.DataFrame({"a": range(50), "b": range(50), "c": list("xy") * 25})
    assert overview.infer_domain(df)["domain"] == overview.GENERAL


def test_overview_netflix_recommends_descriptive_and_clustering():
    df = netflix_like(2000)
    p = profiler.profile_dataset(df)
    o = overview.build_overview(df, p, quality_metrics.assess_quality(df))
    assert o["targets"]["best"] is None
    titles = [r["title"] for r in o["recommendations"]]
    assert "Find segments" in titles and not any(t.startswith("Predict") for t in titles)
    assert any("show_id" in r["title"] for r in o["risks"])


def test_overview_churn_recommends_prediction():
    df, m = dl.load_sample("churn")
    p = profiler.profile_dataset(df)
    o = overview.build_overview(df, p, quality_metrics.assess_quality(df))
    rec = next(r for r in o["recommendations"] if r["title"].startswith("Predict"))
    assert rec["preset"]["target"] == "Churn" and rec["page"] == "views/ml.py"


# ------------------------------------------------------- evidence/reports
def test_every_finding_has_evidence_record():
    df, m = dl.load_sample("churn")
    r = build_results(df, m, target="Churn")
    ids = [i["id"] for i in r["insights"]]
    assert ids == [e["id"] for e in r["evidence"]] and len(set(ids)) == len(ids)
    assert all(e["dataset_version"] == "v0" and e["computation"] for e in r["evidence"])


def test_report_builder_modes_order_and_user_commentary():
    df, m = dl.load_sample("bike")
    charts = [{"id": "c1", "title": "Rows per month", "spec": {"chart": "line", "x": "dteday", "agg": "count", "freq": "MS"}, "in_report": True}]
    r = build_results(df, m, target="cnt", charts=charts)
    cfg = {"mode": "Exploratory data analysis report", "sections": ["Saved charts", "Dataset overview", "Evidence register"],
           "notes": {"Dataset overview": "Collected by the city."}, "summary": "Ridership grows."}
    secs = reporting.build_report(r, df, None, cfg)
    assert [s["title"] for s in secs] == ["Summary (analyst-written)", "Saved charts", "Dataset overview", "Evidence register"]
    md = reporting.to_markdown(r, sections=secs, title="Bike report")
    assert md.startswith("# Bike report") and reporting.USER_PREFIX + "Collected by the city." in md
    assert "Rows per month" in md and "Model evaluation" not in md
    assert reporting.to_pdf(r, sections=secs, title="Bike report").startswith(b"%PDF")
    assert "<html" in reporting.to_html(r, sections=secs).lower()


def test_report_never_fabricates_unavailable_results():
    df, m = dl.load_sample("iris")
    r = build_results(df, m)
    secs = reporting.build_report(r, df, None, {"mode": "Comprehensive technical report"})
    md = reporting.to_markdown(r, sections=secs)
    assert "Estimated Lift" not in md and "No clustering was run" in md
    assert quality_metrics.NOT_EVALUATED.split(" —")[0] in md


# --------------------------------------------------------------- hand-off
def test_handoff_is_one_time_and_expires(monkeypatch):
    tok = handoff.park({"x": 1})
    assert handoff.claim(tok) == {"x": 1} and handoff.claim(tok) is None
    tok = handoff.park({"y": 2})
    monkeypatch.setattr(handoff.time, "time", lambda: 10 ** 12)
    assert handoff.claim(tok) is None
    assert handoff.claim("") is None and handoff.claim("x" * 100) is None
