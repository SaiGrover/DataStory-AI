"""Step 9 — export the analysis as PDF, Markdown or HTML. All formats render the same verified values."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from components import ui
from core import reporting, state

if not ui.require_data("generate a report"):
    st.stop()

ui.page_header("Step 9 · AI report", "Report export",
               "Every figure in the report comes from the same results the dashboard shows. Anything not computed is marked unavailable or left out.")
ui.synthetic_notice()

r = state.results()
story = st.session_state.get("narrative")

c = st.columns(4)
c[0].metric("Cleaning steps", len(r["cleaning"]["history"]))
c[1].metric("Models trained", sum(1 for m in r["modeling"]["models"] if "error" not in m) if r.get("modeling") else 0)
c[2].metric("Feature importance", "Computed" if r.get("importance") else "Not computed")
c[3].metric("Findings", len(r["insights"]))

missing = []
if not r.get("modeling"):
    missing.append("no model has been trained, so methodology and evaluation sections will say so")
if not r.get("importance"):
    missing.append("feature importance has not been computed and will be reported as unavailable")
if not story:
    missing.append("no narrative has been written on the Insights step; the report will list the structured findings only")
if missing:
    ui.callout("<strong>Heads-up.</strong> " + "; ".join(missing).capitalize() + ".", "na")

include = st.multiselect("Sections to include", reporting.SECTION_TITLES, default=reporting.SECTION_TITLES, key="rep_sections")
if not include:
    st.warning("Select at least one section.", icon=":material/warning:")
    st.stop()

sig = (st.session_state["_version"], tuple(include), bool(story))
if st.button("Generate report", type="primary", icon=":material/description:"):
    try:
        with st.status("Building report…", expanded=True) as status:
            narrative = story["text"] if story else None
            st.write("Rendering Markdown…")
            md = reporting.to_markdown(r, narrative, include)
            st.write("Rendering HTML with charts…")
            htm = reporting.to_html(r, narrative, include)
            st.write("Rendering PDF…")
            pdf = reporting.to_pdf(r, narrative, include)
            status.update(label="Report ready", state="complete", expanded=False)
        st.session_state["_report"] = {"sig": sig, "md": md, "html": htm, "pdf": pdf}
    except Exception as exc:  # noqa: BLE001
        st.error(f"The report could not be generated: {type(exc).__name__}: {exc}", icon=":material/error:")

rep = st.session_state.get("_report")
if not rep:
    ui.empty_state("No report generated yet", "Choose the sections you need and select “Generate report”.")
    st.stop()
if rep["sig"] != sig:
    st.warning("The analysis or section selection changed after this report was generated. Generate it again to include the latest results.",
               icon=":material/history:")

stem = f"datastory_report_{Path(state.meta().name).stem}".replace(" ", "_").lower()
d = st.columns(3)
d[0].download_button("Download PDF", rep["pdf"], f"{stem}.pdf", "application/pdf", icon=":material/picture_as_pdf:", width="stretch", type="primary")
d[1].download_button("Download Markdown", rep["md"], f"{stem}.md", "text/markdown", icon=":material/article:", width="stretch")
d[2].download_button("Download HTML", rep["html"], f"{stem}.html", "text/html", icon=":material/html:", width="stretch")

st.subheader("Preview")
with st.container(border=True, height=620):
    st.markdown(rep["md"])
st.caption("Preview of the Markdown export. The PDF and HTML versions add charts drawn from the same values.")
