"""Small shared UI pieces so every page reads as one product."""
from __future__ import annotations

import html
from typing import Any

import streamlit as st

from core import state
from core.data_loader import SYNTHETIC_LABEL

NONE = "— none —"


def page_header(eyebrow: str, title: str, subtitle: str) -> None:
    st.html(f'<span class="ds-eyebrow">{html.escape(eyebrow)}</span><div class="ds-title">{html.escape(title)}</div>'
            f'<div class="ds-sub">{html.escape(subtitle)}</div>')


def callout(text: str, kind: str = "read") -> None:
    """kind: read (interpretation), syn (synthetic disclosure), na (unavailable)."""
    st.html(f'<div class="ds-callout {kind}">{text}</div>')


def interpretation(text: str) -> None:
    import re

    safe = html.escape(text)
    safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
    callout(f"<strong>Reading this chart.</strong> {safe}", "read")


def synthetic_notice() -> None:
    if state.has_data() and state.meta().synthetic:
        callout(f"<strong>{SYNTHETIC_LABEL}.</strong> {html.escape(state.meta().provenance)}", "syn")


def empty_state(title: str, body: str) -> None:
    st.html(f'<div class="ds-empty"><h3>{html.escape(title)}</h3><p>{html.escape(body)}</p></div>')


def require_data(what: str) -> bool:
    """Render the shared empty state and return False when no dataset is loaded."""
    if state.has_data():
        return True
    empty_state("No dataset loaded yet", f"Upload a file or pick a sample dataset to {what}.")
    st.page_link("views/dataset.py", label="Go to Upload", icon=":material/cloud_upload:")
    return False


def target_picker(label: str = "Target column", key: str = "target_pick", help: str | None = None) -> str | None:
    cols = [NONE] + list(state.df().columns)
    cur = state.target()
    choice = st.selectbox(label, cols, index=cols.index(cur) if cur in cols else 0, key=key, help=help)
    state.set_target(None if choice == NONE else choice)
    return state.target()


def chart(fig: Any, key: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, key=key, config={"displaylogo": False})


def safe(fn, *args, **kwargs):
    """Run a chart/analysis callable; show a readable message instead of a traceback."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - last line of defence for the UI
        st.error(f"This view could not be computed for the current selection: {exc}")
        return None
