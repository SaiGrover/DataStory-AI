"""App chrome: brand header, dataset strip and the workflow stepper."""
from __future__ import annotations

import html

import streamlit as st

from components.styling import LOGO
from core import state
from core.data_loader import SYNTHETIC_LABEL

# (file, title, short nav label, material icon)
STEPS = [
    ("views/dataset.py", "Dataset", "Upload", "cloud_upload"),
    ("views/quality.py", "Data Quality", "Data Health", "database"),
    ("views/cleaning.py", "Data Cleaning", "Clean Data", "auto_fix_high"),
    ("views/eda.py", "Exploratory Analysis", "EDA", "show_chart"),
    ("views/insights.py", "AI Insights", "Insights", "lightbulb"),
    ("views/ml.py", "Machine Learning", "Modeling", "science"),
    ("views/model_results.py", "Model Results", "Results", "trophy"),
    ("views/assistant.py", "AI Assistant", "Agent", "smart_toy"),
    ("views/reports.py", "Reports", "AI Report", "description"),
]


def build_pages() -> list:
    return [st.Page(path, title=title, icon=f":material/{icon}:", default=(i == 0))
            for i, (path, title, _, icon) in enumerate(STEPS)]


def header() -> None:
    st.html(f'<div class="ds-header"><div class="ds-brand">{LOGO}<span>DataStory AI</span></div>'
            '<div class="ds-tagline">Your Junior Data Analyst</div></div>')
    if state.has_data():
        m, r = state.meta(), state.results()
        p, h = r["profile"], r["quality"]["health"]
        health = f"{h['value']:.0f}/100" if h.get("value") is not None else "unavailable"
        bits = [f"<strong>{html.escape(m.name)}</strong>", f"{p['rows']:,} rows", f"{p['columns']:,} columns",
                f"Health {health}"]
        if state.target():
            bits.append(f"Target: <strong>{html.escape(state.target())}</strong>")
        if r["cleaning"]["history"]:
            bits.append(f"{len(r['cleaning']['history'])} cleaning step(s)")
        if m.synthetic:
            bits.append(f'<span class="ds-pill syn">{SYNTHETIC_LABEL}</span>')
        st.html('<div class="ds-strip">' + "".join(f"<span>{b}</span>" for b in bits) + "</div>")


def stepper(pages: list, current) -> None:
    with st.container(key="ds_nav"):
        cols = st.columns(len(pages))
        for i, (col, page, (_, _, label, _)) in enumerate(zip(cols, pages, STEPS), 1):
            with col:
                if page.title == current.title:
                    st.html(f'<div class="ds-step-active"><span class="n">{i}</span>{label}</div>')
                else:
                    st.page_link(page, label=label, icon=page.icon, width="stretch")
