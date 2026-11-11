"""App chrome: brand header with dataset context, the workflow flowchart and session utilities.

The flowchart is built from native ``st.page_link`` elements (so every node
navigates through Streamlit and survives reruns). Each node sits in a keyed
container whose key encodes its state (active / done / todo / locked); the
connectors and their moving highlight are pure CSS on those containers.
"""
from __future__ import annotations

import html
from dataclasses import dataclass

import streamlit as st

from components.styling import BRANCH, LOGO
from components.theme import theme_toggle
from core import state


@dataclass(frozen=True)
class Step:
    key: str
    path: str
    title: str
    label: str
    icon: str
    url: str
    main: bool = True


STEPS = [
    Step("dataset", "views/dataset.py", "Dataset", "Dataset", "database", ""),
    Step("quality", "views/quality.py", "Data Health", "Health", "health_and_safety", "health"),
    Step("cleaning", "views/cleaning.py", "Data Cleaning", "Clean", "auto_fix_high", "clean"),
    Step("eda", "views/eda.py", "Exploratory Analysis", "Explore", "insights", "explore"),
    Step("insights", "views/insights.py", "AI Insights", "Insights", "lightbulb", "insights"),
    Step("ml", "views/ml.py", "Machine Learning", "Model", "model_training", "model"),
    Step("results", "views/model_results.py", "Model Results", "Results", "leaderboard", "results"),
    Step("assistant", "views/assistant.py", "AI Copilot", "Agent", "forum", "agent"),
    Step("reports", "views/reports.py", "Report Builder", "Report", "description", "report"),
    Step("compare", "views/compare.py", "Original vs Cleaned", "Compare", "compare_arrows", "compare", main=False),
    Step("clustering", "views/clustering.py", "Clustering & Segments", "Segments", "bubble_chart", "segments", main=False),
]
BY_PATH = {s.path: s for s in STEPS}


def build_pages() -> list:
    return [st.Page(s.path, title=s.title, icon=f":material/{s.icon}:", url_path=s.url or None, default=(s.key == "dataset"))
            for s in STEPS]


def _page(pages, step: Step):
    return next(p for p in pages if p.title == step.title)


def header(pages) -> None:
    with st.container(key="ds_header", horizontal=True, vertical_alignment="center"):
        st.html(f'<div class="ds-brand">{LOGO}<div><span class="w">DataStory <b>AI</b></span>'
                '<span class="d">Evidence-based analytics workspace</span></div></div>')
        if state.has_data():
            m, r = state.meta(), state.results()
            p, h = r["profile"], r["quality"]["health"]
            health = f"{h['value']:.0f}/100" if h.get("value") is not None else "n/a"
            status = "Cleaned" if r["cleaning"]["history"] else "Original"
            bits = [f'<span><span class="ds-dot"></span><b class="name" title="{html.escape(m.name)}">{html.escape(m.name)}</b></span>',
                    f"<span><b>{p['rows']:,}</b> rows</span>", f"<span><b>{p['columns']:,}</b> cols</span>",
                    f"<span>Health <b>{health}</b></span>", f"<span>{state.data_version()} · {status}</span>"]
            if state.target():
                bits.append(f"<span>Target <b>{html.escape(state.target())}</b></span>")
            pill = ('<span class="ds-pill syn">Synthetic demo</span>' if m.synthetic else
                    '<span class="ds-pill ok">Verified public</span>' if m.provenance_category == "verified_public" else
                    '<span class="ds-pill info">Provenance unverified</span>')
            bits.append(pill)
            st.html(f'<div class="ds-context" role="status" aria-label="Active dataset">{"".join(bits)}</div>', width="stretch")
        else:
            st.html('<div class="ds-context">No dataset loaded</div>', width="stretch")
        _session_menu(pages)
        theme_toggle([s.url for s in STEPS])


def _session_menu(pages) -> None:
    from core import session_io

    with st.popover("Session", icon=":material/folder_open:", help="Save, resume or start a new analysis"):
        st.markdown("**Analysis session**")
        st.caption("Sessions live in this browser tab's memory only. Save a package to keep your work; nothing is stored on the server.")
        if state.has_data():
            inc = st.checkbox("Include the dataset in the package", value=True, key="sess_inc",
                              help="Untick to share configuration only; you'll need the original file to resume.")
            if st.button("Prepare session package", icon=":material/inventory_2:", width="stretch", key="sess_prep"):
                try:
                    st.session_state["_sess_pkg"] = (session_io.export_session(state.snapshot(), include_data=inc), state.data_key(), inc)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Could not build the package: {exc}")
            pkg = st.session_state.get("_sess_pkg")
            if pkg and pkg[1] == state.data_key():
                stem = state.meta().name.rsplit(".", 1)[0].replace(" ", "_").lower()
                st.download_button("Download .datastory.zip", pkg[0], f"{stem}.datastory.zip", "application/zip",
                                   icon=":material/download:", width="stretch", type="primary", key="sess_dl")
        st.page_link(_page(pages, STEPS[0]), label="Resume a saved session…", icon=":material/upload_file:")
        if state.has_data():
            if st.button("New analysis", icon=":material/restart_alt:", width="stretch", key="sess_new",
                         help="Clears the dataset, cleaning, models, charts and chat from this tab."):
                st.session_state["_confirm_new"] = True
            if st.session_state.get("_confirm_new"):
                st.warning("This clears everything in this tab. Save a session package first if you need it.")
                c1, c2 = st.columns(2)
                if c1.button("Clear", type="primary", key="sess_clear", width="stretch"):
                    state.clear()
                    st.switch_page(_page(pages, STEPS[0]))
                if c2.button("Cancel", key="sess_cancel", width="stretch"):
                    st.session_state["_confirm_new"] = False
                    st.rerun()


def _node_state(step: Step, current: Step, status: dict) -> str:
    s = status[step.key]
    if step.key == current.key:
        return "active"
    if not s["available"]:
        return "locked"
    return "done" if s["done"] else "todo"


def flowchart(pages, current_page) -> None:
    current = next((s for s in STEPS if s.title == current_page.title), STEPS[0])
    status = state.stage_status()
    main = [s for s in STEPS if s.main]
    done = sum(1 for s in main if status[s.key]["done"])
    with st.container(key="ds_flow", horizontal=True):
        for s in main:
            ns = _node_state(s, current, status)
            with st.container(key=f"flow_{ns}_{s.key}"):
                tip = status[s.key]["why"] + (" — completed." if status[s.key]["done"] else "")
                st.page_link(_page(pages, s), label=s.label, icon=f":material/{s.icon}:", disabled=ns == "locked", help=tip)
    with st.container(key="ds_branch", horizontal=True):
        st.html(f'<span class="ds-branch-label">{BRANCH} Optional paths</span>')
        for s in [x for x in STEPS if not x.main]:
            ns = _node_state(s, current, status)
            with st.container(key=f"branch_{ns}_{s.key}"):
                st.page_link(_page(pages, s), label=s.label, icon=f":material/{s.icon}:", disabled=ns == "locked",
                             help=status[s.key]["why"])
        st.html(f'<span class="ds-progress"><b>{done}</b> of {len(main)} stages complete · '
                f'now: <b>{html.escape(current.title)}</b></span>', width="stretch")
    # compact selector for small screens (CSS shows it below 860px)
    with st.container(key="ds_flow_mobile"):
        idx = next((i for i, s in enumerate(STEPS) if s.key == current.key), 0) + 1
        with st.popover(f"Step {idx} of {len(STEPS)} · {current.label} — {done}/{len(main)} complete",
                        icon=":material/route:", width="stretch"):
            for s in STEPS:
                ns = _node_state(s, current, status)
                mark = "✓ " if ns == "done" else ("● " if ns == "active" else "")
                st.page_link(_page(pages, s), label=f"{mark}{s.label}" + ("" if s.main else " (optional)"),
                             icon=f":material/{s.icon}:", disabled=ns == "locked", help=status[s.key]["why"])
