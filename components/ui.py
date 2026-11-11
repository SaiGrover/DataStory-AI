"""Shared UI pieces so every page reads as one product."""
from __future__ import annotations

import html
import re
from typing import Any

import streamlit as st

from components.styling import TOKENS
from components.theme import is_dark
from core import state
from core.data_loader import PROV_PUBLIC, PROV_SYNTHETIC, PROV_UPLOAD, SYNTHETIC_LABEL

NONE = "— none —"


def page_header(eyebrow: str, title: str, subtitle: str) -> None:
    st.html(f'<span class="ds-eyebrow">{html.escape(eyebrow)}</span><div class="ds-title">{html.escape(title)}</div>'
            f'<div class="ds-sub">{html.escape(subtitle)}</div>')


def section(title: str, note: str | None = None) -> None:
    st.html(f'<div class="ds-section">{html.escape(title)}' + (f'<span class="n">{html.escape(note)}</span>' if note else "") + "</div>")


def callout(text: str, kind: str = "read") -> None:
    """kind: read (interpretation), syn (synthetic), na (unavailable), prov (provenance), err."""
    st.html(f'<div class="ds-callout {kind}">{text}</div>')


def md_inline(text: str) -> str:
    safe = html.escape(text)
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)


def interpretation(text: str, label: str = "Reading this chart.") -> None:
    callout(f"<strong>{html.escape(label)}</strong> {md_inline(text)}", "read")


def provenance_notice() -> None:
    """Provenance banner for the active dataset — accurate for every source category."""
    if not state.has_data():
        return
    m = state.meta()
    lin = state.results()["lineage"]
    derived = f" Active version {lin['active_version']} has {len(lin['steps'])} cleaning step(s) applied." if lin["steps"] else ""
    if m.provenance_category == PROV_SYNTHETIC:
        callout(f"<strong>{SYNTHETIC_LABEL}.</strong> {html.escape(m.provenance)}{derived}", "syn")
    elif m.provenance_category == PROV_UPLOAD and lin["steps"]:
        callout(f"<strong>User-uploaded dataset — provenance unverified.</strong>{derived}", "prov")
    elif m.provenance_category == PROV_PUBLIC and lin["steps"]:
        callout(f"<strong>Verified bundled public dataset.</strong>{derived}", "prov")


synthetic_notice = provenance_notice  # backwards-compatible name used by several pages


def empty_state(title: str, body: str) -> None:
    st.html(f'<div class="ds-empty"><h3>{html.escape(title)}</h3><p>{html.escape(body)}</p></div>')


def require_data(what: str) -> bool:
    """Render the shared empty state and return False when no dataset is loaded."""
    if state.has_data():
        return True
    empty_state("No dataset loaded yet", f"Upload a file or pick a sample dataset to {what}.")
    st.page_link("views/dataset.py", label="Go to Dataset", icon=":material/database:")
    return False


def target_picker(label: str = "Target column", key: str = "target_pick", help: str | None = None) -> str | None:
    cols = [NONE] + list(state.df().columns)
    cur = state.target()
    choice = st.selectbox(label, cols, index=cols.index(cur) if cur in cols else 0, key=key, help=help)
    state.set_target(None if choice == NONE else choice)
    return state.target()


def themed(fig: Any) -> Any:
    """Adapt a DataStory-styled Plotly figure to the active light/dark theme."""
    t = TOKENS["dark" if is_dark() else "light"]
    fig.update_layout(font=dict(family="Plus Jakarta Sans, system-ui, sans-serif", color=t["text"]),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      hoverlabel=dict(bgcolor=t["elevated"], font=dict(color=t["text"], family="Plus Jakarta Sans, sans-serif"),
                                      bordercolor=t["border"]),
                      legend=dict(font=dict(color=t["text-2"])))
    if fig.layout.title and fig.layout.title.text:
        fig.update_layout(title=dict(font=dict(family="Manrope, sans-serif", size=15, color=t["text"])))
    fig.update_xaxes(gridcolor=t["border"], zerolinecolor=t["border"], linecolor=t["border"], tickfont=dict(color=t["text-2"]),
                     title_font=dict(color=t["text-2"]), automargin=True)
    fig.update_yaxes(gridcolor=t["border"], zerolinecolor=t["border"], linecolor=t["border"], tickfont=dict(color=t["text-2"]),
                     title_font=dict(color=t["text-2"]), automargin=True)
    return fig


def chart(fig: Any, key: str | None = None) -> None:
    st.plotly_chart(themed(fig), width="stretch", theme=None, key=key,
                    config={"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2}})


def spec_chart(spec_dict: dict[str, Any], key: str, title: str | None = None, source: str = "EDA",
               show_table: bool = True, toolbar: bool = True) -> Any:
    """Render a ChartSpec with its interpretation and the shared chart toolbar. Returns the ChartOutput or None."""
    from core import charts as ch

    try:
        spec = ch.ChartSpec.from_dict(spec_dict)
        out = ch.render(state.analysis_df(), spec, state.results()["profile"]["kinds"])
    except ch.ChartSpecError as exc:
        callout(html.escape(str(exc)), "na")
        return None
    except Exception as exc:  # noqa: BLE001
        st.error(f"This chart could not be computed: {exc}")
        return None
    chart(out.figure, f"fig_{key}")
    for n in out.notes:
        st.caption(n)
    interpretation(out.interpretation)
    if toolbar:
        chart_toolbar(spec, out, key, title or (out.figure.layout.title.text if out.figure.layout.title else spec.describe()), source,
                      show_table)
    return out


def chart_toolbar(spec: Any, out: Any, key: str, title: str, source: str, show_table: bool = True) -> None:
    saved = {c["id"] for c in state.charts()}
    with st.container(horizontal=True, key=f"tb_{key}"):
        is_saved = spec.key() in saved or any(c["spec"] == spec.to_dict() for c in state.charts())
        if st.button("Saved to analysis" if is_saved else "Save & add to report", icon=":material/bookmark_added:" if is_saved else ":material/bookmark_add:",
                     key=f"save_{key}", disabled=is_saved, help="Stores the chart specification; it is re-computed from the data whenever it is shown."):
            state.save_chart(spec.to_dict(), title, source)
            st.toast("Chart saved to this analysis and added to the report.", icon=":material/bookmark_added:")
            st.rerun()
        st.download_button("Data (CSV)", out.table.to_csv(index=False).encode("utf-8"), f"{key}.csv", "text/csv",
                           icon=":material/table_view:", key=f"dl_{key}", help="The aggregated values behind this chart.")
        if st.button("Ask the copilot", icon=":material/forum:", key=f"ask_{key}"):
            st.session_state["_copilot_prefill"] = {"question": f"Explain this chart: {title}", "spec": spec.to_dict()}
            st.switch_page("views/assistant.py")
    if show_table:
        with st.expander("Values behind this chart"):
            st.dataframe(out.table, width="stretch", hide_index=True, height=min(360, 38 * (len(out.table) + 1)))


def safe(fn, *args, **kwargs):
    """Run a chart/analysis callable; show a readable message instead of a traceback."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - last line of defence for the UI
        st.error(f"This view could not be computed for the current selection: {exc}")
        return None
