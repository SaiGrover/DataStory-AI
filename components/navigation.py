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

from components.styling import BRANCH, MARK
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
    accent: str = "sage"  # stage colour token (--ds-c-<accent>), drawn from the DataStory palette


STEPS = [
    Step("dataset", "views/dataset.py", "Dataset", "Dataset", "database", "", accent="sage"),
    Step("quality", "views/quality.py", "Data Health", "Health", "health_and_safety", "health", accent="teal"),
    Step("cleaning", "views/cleaning.py", "Data Cleaning", "Clean", "auto_fix_high", "clean", accent="amber"),
    Step("eda", "views/eda.py", "Exploratory Analysis", "Explore", "insights", "explore", accent="olive"),
    Step("insights", "views/insights.py", "AI Insights", "Insights", "lightbulb", "insights", accent="clay"),
    Step("ml", "views/ml.py", "Machine Learning", "Model", "model_training", "model", accent="teal"),
    Step("results", "views/model_results.py", "Model Results", "Results", "leaderboard", "results", accent="sage"),
    Step("assistant", "views/assistant.py", "AI Copilot", "Agent", "forum", "agent", accent="olive"),
    Step("reports", "views/reports.py", "Report Builder", "Report", "description", "report", accent="moss"),
    Step("compare", "views/compare.py", "Original vs Cleaned", "Compare", "compare_arrows", "compare", main=False, accent="amber"),
    Step("clustering", "views/clustering.py", "Clustering & Segments", "Segments", "bubble_chart", "segments", main=False, accent="clay"),
]
BY_PATH = {s.path: s for s in STEPS}


def build_pages() -> list:
    return [st.Page(s.path, title=s.title, icon=f":material/{s.icon}:", url_path=s.url or None, default=(s.key == "dataset"))
            for s in STEPS]


def _page(pages, step: Step):
    return next(p for p in pages if p.title == step.title)


def header(pages) -> None:
    with st.container(key="ds_header", horizontal=True, vertical_alignment="center"):
        st.html(f'<div class="ds-brand">{MARK}<div><span class="w">DataStory AI</span>'
                '<span class="d">Evidence-based <b>analytics</b> workspace</span></div></div>')
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


def _metric(key: str) -> str:
    """One live, computed line per stage (never a fixed number)."""
    ss = st.session_state
    if not state.has_data():
        return "load a dataset" if key == "dataset" else "waiting for data"
    r = state.results()
    p = r["profile"]
    if key == "dataset":
        return f"{p['rows']:,} rows · {p['columns']} cols"
    if key == "quality":
        h = r["quality"]["health"].get("value")
        return f"health {h:.0f}/100" if h is not None else "health n/a"
    if key == "cleaning":
        n = len(ss["history"])
        return f"{n} step{'s' if n != 1 else ''} · {state.data_version()}" if n else "no steps yet"
    if key == "eda":
        n = len(ss.get("charts") or [])
        return f"{n} saved chart{'s' if n != 1 else ''}" if n else ("charts explored" if ss.get("eda_generated") else "no charts yet")
    if key == "insights":
        return f"{len(r['insights'])} findings" + (" · story" if ss.get("narrative") else "")
    if key == "ml":
        m = r.get("modeling")
        if not m:
            return "not trained"
        best = m["models"][0]
        v = best["metrics"].get(m["primary_metric"])
        return f"{m['primary_metric']} {v:.3f}" if v is not None else m["best_model"]
    if key == "results":
        return "importance ready" if ss.get("importance") else ("needs importance" if r.get("modeling") else "needs a model")
    if key == "assistant":
        n = sum(1 for m in ss.get("chat", []) if m.get("role") == "assistant")
        return f"{n} answer{'s' if n != 1 else ''}" if n else "ask anything"
    if key == "reports":
        return "report ready" if status_cache().get("reports", {}).get("done") else "not built"
    if key == "compare":
        return f"{len(ss['history'])} change(s)" if ss["history"] else ""
    if key == "clustering":
        cl = state.clustering()
        return f"{cl['result']['n_clusters']} clusters" if cl else ""
    return ""


def status_cache() -> dict:
    return st.session_state.get("_stage_status") or {}


def _accent_css() -> str:
    rules = []
    for s in STEPS:
        for ns in ("active", "done", "todo", "locked"):
            pre = "flow" if s.main else "branch"
            rules.append(f".st-key-{pre}_{ns}_{s.key}{{--c:var(--ds-c-{s.accent})}}")
    return "<style>" + "".join(rules) + "</style>"


def flowchart(pages, current_page) -> None:
    current = next((s for s in STEPS if s.title == current_page.title), STEPS[0])
    status = state.stage_status()
    st.session_state["_stage_status"] = status
    main = [s for s in STEPS if s.main]
    done = sum(1 for s in main if status[s.key]["done"])
    st.html(_accent_css())
    with st.container(key="ds_rail"):
        with st.container(key="ds_flow", horizontal=True):
            for i, s in enumerate(main, 1):
                ns = _node_state(s, current, status)
                with st.container(key=f"flow_{ns}_{s.key}"):
                    tip = status[s.key]["why"] + (" — completed." if status[s.key]["done"] else "")
                    label = f"Step {i:02d}  \n**{s.label}**  \n*{_metric(s.key)}*"
                    st.page_link(_page(pages, s), label=label, icon=f":material/{s.icon}:", disabled=ns == "locked", help=tip,
                                 width="stretch")
        with st.container(key="ds_branch", horizontal=True, vertical_alignment="center"):
            st.html(f'<span class="ds-branch-label">{BRANCH} Optional paths</span>')
            for s in [x for x in STEPS if not x.main]:
                ns = _node_state(s, current, status)
                extra = _metric(s.key)
                with st.container(key=f"branch_{ns}_{s.key}"):
                    st.page_link(_page(pages, s), label=s.label + (f" · *{extra}*" if extra else ""), icon=f":material/{s.icon}:",
                                 disabled=ns == "locked", help=status[s.key]["why"])
            pct_done = int(100 * done / len(main))
            st.html(f'<span class="ds-progress"><span class="bar"><i style="width:{pct_done}%"></i></span>'
                    f'<b>{done}</b> of {len(main)} stages complete · now: <b>{html.escape(current.title)}</b></span>', width="stretch")
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


def ask_dock(pages, current_page) -> None:
    """Floating shortcut to the copilot, Data Vortex style; hidden on the copilot page itself."""
    if not state.has_data() or current_page.title == "AI Copilot":
        return
    with st.container(key="askdock"):
        st.page_link(_page(pages, next(s for s in STEPS if s.key == "assistant")), label="Ask the copilot", icon=":material/forum:")
