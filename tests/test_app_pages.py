"""End-to-end smoke test: drive the real Streamlit app through every page headlessly."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from components.navigation import STEPS

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _run(at, page=None):
    if page:
        at.switch_page(page)
    at.run(timeout=120)
    assert not at.exception, [e.value for e in at.exception]
    assert not [e for e in at.error if "went wrong" in e.value], [e.value for e in at.error]
    return at


def test_every_page_renders_without_data():
    at = _run(AppTest.from_file(APP, default_timeout=120))
    for step in STEPS[1:]:
        _run(at, step.path)


@pytest.mark.parametrize("sample", ["titanic", "bike"])
def test_full_workflow(sample):
    at = _run(AppTest.from_file(APP, default_timeout=120))
    at.button(key=f"load_{sample}").click()
    _run(at)
    assert at.session_state["df"] is not None
    for step in STEPS:
        _run(at, step.path)
    # train with defaults, compute importance, generate report
    _run(at, "views/ml.py")
    next(b for b in at.button if b.label == "Train models").click()
    _run(at)
    assert at.session_state["training"] is not None
    _run(at, "views/model_results.py")
    next(b for b in at.button if b.label == "Compute importance").click()
    _run(at)
    assert at.session_state["importance"]["features"]
    _run(at, "views/insights.py")
    next(b for b in at.button if b.label == "Write narrative").click()
    _run(at)
    _run(at, "views/assistant.py")
    at.chat_input[0].set_value("Which model performed best?").run()
    assert at.session_state["chat"][-1]["answer"]
    _run(at, "views/reports.py")
    next(b for b in at.button if b.label == "Generate report").click()
    _run(at)
    rep = at.session_state["_report"]
    assert rep["pdf"].startswith(b"%PDF") and at.session_state["training"].results["best_model"] in rep["md"]
    # cleaning invalidates trained models
    _run(at, "views/cleaning.py")
    at.selectbox(key="cl_op").set_value("drop_duplicates" if sample == "bike" else "impute")
    _run(at)
    next(b for b in at.button if b.label == "Preview changes").click()
    _run(at)


def test_ml_page_blocks_identifier_target_and_shows_diagnostics():
    at = _run(AppTest.from_file(APP, default_timeout=120))
    at.button(key="load_churn").click()
    _run(at)
    _run(at, "views/ml.py")
    assert at.session_state["target"] == "Churn"
    assert any("Readiness assessment" in s.value for s in at.subheader)
    at.selectbox(key="ml_target").set_value("CustomerID")
    _run(at)
    train = next(b for b in at.button if b.label == "Train models")
    assert train.disabled
    at.selectbox(key="ml_target").set_value("Churn")
    _run(at)
    next(b for b in at.button if b.label == "Train models").click()
    _run(at)
    assert at.session_state["training"] is not None
    _run(at, "views/model_results.py")
    at.toggle(key="mr_cm_norm").set_value(True)
    _run(at)
