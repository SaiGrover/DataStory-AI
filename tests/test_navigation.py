"""Flowchart navigation: real page links, data-driven completion, locked prerequisites, rerun stability."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

from components.navigation import STEPS

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _links(at):
    """Map each stage's short label to its link (rail labels read 'Step NN  **Label**  *metric*')."""
    import re

    out = {}
    for p in at.get("page_link"):
        m = re.search(r"\*\*(.+?)\*\*", p.proto.label)
        name = m.group(1) if m else p.proto.label.split(" · ")[0]
        out.setdefault(name, p.proto)
    return out


def _app():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    return at


def test_every_stage_has_a_real_page_link():
    at = _app()
    links = _links(at)
    for s in STEPS:
        assert s.label in links, s.label
        assert links[s.label].page_script_hash  # resolves to a registered st.Page


def test_without_data_only_dataset_is_available():
    links = _links(_app())
    assert not links["Dataset"].disabled
    for s in STEPS[1:]:
        assert links[s.label].disabled, s.label
        assert "Load a dataset first" in links[s.label].help


def test_visiting_does_not_complete_a_stage_and_locks_follow_prerequisites():
    at = _app()
    at.button(key="load_churn").click()
    at.run()
    for page in ("views/eda.py", "views/insights.py", "views/assistant.py", "views/reports.py"):
        at.switch_page(page)
        at.run()
        assert not at.exception
    st_ = at.session_state
    assert st_["eda_generated"] == 0 and st_["narrative"] is None and st_["chat"] == [] and st_["_report"] is None
    links = _links(at)
    assert links["Results"].disabled and "Train a model first" in links["Results"].help
    assert links["Compare"].disabled and "cleaning step" in links["Compare"].help
    assert not links["Explore"].disabled and "completed" not in links["Explore"].help


def test_stage_completion_follows_real_operations():
    at = _app()
    at.button(key="load_bike").click()
    at.run()
    at.switch_page("views/cleaning.py")
    at.run()
    at.selectbox(key="cl_op").set_value("drop_duplicates")
    at.run()
    next(b for b in at.button if b.label == "Preview changes").click()
    at.run()
    apply = next(b for b in at.button if b.label == "Apply to dataset")
    if not apply.disabled:
        apply.click()
        at.run()
        links = _links(at)
        assert "completed" in links["Clean"].help and not links["Compare"].disabled
    # navigation keeps working across reruns
    for s in STEPS:
        at.switch_page(s.path)
        at.run()
        assert not at.exception, s.path
