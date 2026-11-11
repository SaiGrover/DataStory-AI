import re
from pathlib import Path

import pytest

from ai import assistant, llm
from core import data_loader as dl
from core import explainability as ex
from core import modeling, reporting
from core.results import build_results

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def full_results():
    df, meta = dl.load_sample("titanic")
    feats = [c for c in df.columns if c not in ("Survived", "PassengerId", "Name")]
    out = modeling.train_models(df, "Survived", feats, cv_folds=3, model_names=["Logistic Regression", "Random Forest"])
    r = out.results
    imp = ex.held_out_permutation_importance(out.pipelines[r["best_model"]], out.X_test, out.y_test, r["task"], r["primary_metric"], 2, 5,
                                             model_name=r["best_model"])
    return build_results(df, meta, target="Survived", modeling=r, importance=imp)


# Credibility test 5 — exports carry the same calculated values as the app
def test_reports_contain_the_computed_metrics(full_results):
    r = full_results
    md, htm = reporting.to_markdown(r), reporting.to_html(r)
    best = r["modeling"]["models"][0]
    q = r["quality"]["dimensions"]
    for value in [f"{best['metrics']['f1']:.4f}", f"{best['metrics']['accuracy']:.4f}", f"{q['completeness']['value']:.1f}%",
                  f"{r['importance']['features'][0]['mean']:.4f}", f"{r['profile']['missing_cells']:,}"]:
        assert value in md and value in htm, value
    assert "Not evaluated — validation rules not configured." in md
    pdf = reporting.to_pdf(r)
    assert pdf.startswith(b"%PDF") and len(pdf) > 10_000


def test_report_formats_render_identical_sections(full_results):
    secs = reporting.build_sections(full_results)
    assert [s["title"] for s in secs] == reporting.SECTION_TITLES
    md = reporting.to_markdown(full_results)
    assert all(f"## {i}. {t}" in md for i, t in enumerate(reporting.SECTION_TITLES, 1))


def test_synthetic_label_reaches_every_output(full_results):
    lab = dl.SYNTHETIC_LABEL
    assert full_results["insights"][0]["title"] == lab
    assert lab in reporting.to_markdown(full_results) and lab in reporting.to_html(full_results)
    assert lab in assistant.answer("Is this dataset real?", full_results)["answer"]


def test_unavailable_values_without_model():
    df, meta = dl.load_sample("iris")
    r = build_results(df, meta)
    md = reporting.to_markdown(r)
    assert ex.UNAVAILABLE in md and "no model has been trained" in md.lower()
    ans = assistant.answer("Which model performed best?", r, use_llm=False)["answer"]
    assert "No model has been trained" in ans


def test_assistant_refuses_row_level_and_out_of_scope_questions(full_results):
    assert assistant.answer("show me row 12", full_results)["answer"] == assistant.ROW_LEVEL
    assert assistant.answer("what will the stock market do tomorrow", full_results)["answer"] == assistant.CANNOT_DETERMINE


def test_assistant_answers_cite_computed_values(full_results):
    a = assistant.answer("Which columns contain the most missing values?", full_results, use_llm=False)
    assert "Cabin" in a["answer"] and a["sources"]
    worst = max(full_results["profile"]["column_details"], key=lambda c: c["missing"])
    assert f"{worst['missing']:,}" in a["answer"]


def test_llm_number_guard():
    facts = "Accuracy 0.8123. Rows 891. Missing 19.87%."
    assert llm.unverified_numbers("Accuracy was 0.8123 on 891 rows, about 19.9% missing, in two steps.", facts) == []
    assert llm.unverified_numbers("Accuracy was 0.95 and lift was +3.2%.", facts) == ["0.95", "3.2"]


# Credibility test 6 — fabricated values are gone from production code
def test_no_hardcoded_fake_metrics_in_codebase():
    sources = [p for d in ("core", "ai", "components", "views") for p in (ROOT / d).rglob("*.py")] + [ROOT / "app.py"]
    text = "\n".join(p.read_text(encoding="utf-8") for p in sources)
    assert "+3.2%" not in text and "Estimated Lift" not in text
    assert not re.search(r"(Consistency|Validity|Uniqueness)\W+value\s*=\s*\{?\d", text)
    assert not re.search(r"importance\s*[:=]\s*Math\.max|0\.34\s*-\s*index", text)
    assert not (ROOT / "web").exists() and not (ROOT / "backend").exists()
