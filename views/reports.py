"""Step 9 — report builder (PDF / Markdown / HTML) and reproducible notebook export."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from components import ui
from core import data_loader as dl
from core import notebook, reporting, state

if not ui.require_data("build a report"):
    st.stop()

ui.page_header("Step 9 · Report", "Report builder",
               "Choose a report type, arrange sections, add your own commentary and export. Computed figures come from the same verified "
               "results as the dashboard; your commentary is labelled as user-authored.")
ui.provenance_notice()

r = state.results()
story = st.session_state.get("narrative")
cfg = st.session_state.get("report_config") or {}
stem = f"datastory_{Path(state.meta().name).stem}".replace(" ", "_").lower()

c = st.columns(5)
c[0].metric("Dataset version", state.data_version())
c[1].metric("Saved charts", len(state.charts()))
c[2].metric("Models trained", sum(1 for m in r["modeling"]["models"] if "error" not in m) if r.get("modeling") else 0)
c[3].metric("Feature importance", "Computed" if r.get("importance") else "Not computed")
c[4].metric("Evidence records", len(r["evidence"]))

tab_build, tab_nb = st.tabs(["Report", "Reproducible notebook"])
with tab_build:
    left, right = st.columns([1, 1.35], gap="large")
    with left, st.container(border=True):
        st.markdown("**1 · Configure**")
        modes = list(reporting.REPORT_MODES)
        mode = st.selectbox("Report type", modes, index=modes.index(cfg.get("mode", "Comprehensive technical report")), key="rep_mode",
                            help="Sets the default sections; you can change them below.")
        st.caption(reporting.REPORT_MODES[mode]["desc"])
        title = st.text_input("Title", cfg.get("title") or f"DataStory AI Report — {state.meta().name}", key="rep_title")
        default_secs = cfg.get("sections") if cfg.get("mode") == mode and cfg.get("sections") else reporting.REPORT_MODES[mode]["sections"]
        table = pd.DataFrame({"Include": [t in default_secs for t in reporting.ALL_SECTIONS], "Section": reporting.ALL_SECTIONS,
                              "Order": [default_secs.index(t) + 1 if t in default_secs else 99 for t in reporting.ALL_SECTIONS]})
        table = table.sort_values(["Include", "Order"], ascending=[False, True]).reset_index(drop=True)
        edited = st.data_editor(table, hide_index=True, width="stretch", key=f"rep_secs_{mode}", disabled=["Section"],
                                column_config={"Order": st.column_config.NumberColumn("Order", min_value=1, max_value=99, step=1,
                                                                                      help="Lower numbers come first.")},
                                height=min(560, 36 * (len(table) + 1)))
        sections = edited[edited["Include"]].sort_values("Order")["Section"].tolist()
        charts = [ch for ch in state.charts() if ch.get("in_report", True)]
        chart_ids = [ch["id"] for ch in charts]
        if state.charts():
            picked = st.multiselect("Charts to include", [ch["id"] for ch in state.charts()], default=cfg.get("chart_ids") or chart_ids,
                                    format_func=lambda cid: next(ch["title"] for ch in state.charts() if ch["id"] == cid), key="rep_charts")
            chart_ids = picked
        else:
            st.caption("No saved charts yet — use “Save & add to report” under any chart on Explore or in the copilot.")
        with st.expander("Your commentary (labelled as user-authored)", icon=":material/edit_note:"):
            summary = st.text_area("Opening summary", cfg.get("summary", ""), key="rep_summary", height=90)
            notes = dict(cfg.get("notes") or {})
            sec_for_note = st.selectbox("Add a note to section", sections or ["—"], key="rep_note_sec")
            if sections:
                notes[sec_for_note] = st.text_area("Note", notes.get(sec_for_note, ""), key=f"rep_note_{sec_for_note}", height=80)
            notes = {k: v for k, v in notes.items() if v and k in sections}
            if notes:
                st.caption("Notes on: " + ", ".join(notes))
        new_cfg = {"mode": mode, "title": title, "sections": sections, "chart_ids": chart_ids, "summary": summary, "notes": notes}
        if new_cfg != cfg:
            st.session_state["report_config"] = new_cfg
        missing = []
        if not r.get("modeling") and any(s in sections for s in ("Model evaluation", "Feature importance", "Machine-learning methodology")):
            missing.append("no model is trained, so model sections will say so")
        if not story and "Analytical narrative" in sections:
            missing.append("no data story has been written on Insights; the structured findings are listed instead")
        if missing:
            ui.callout("<strong>Heads-up.</strong> " + "; ".join(missing).capitalize() + ".", "na")
        sig = (st.session_state["_version"], repr(new_cfg), bool(story))
        gen = st.button("Generate report", type="primary", icon=":material/description:", width="stretch", disabled=not sections)

    if gen:
        try:
            with st.status("Building report…", expanded=True) as status:
                narrative = story["text"] if story else None
                secs = reporting.build_report(r, state.analysis_df(), narrative, new_cfg)
                st.write("Rendering Markdown…")
                md = reporting.to_markdown(r, narrative, sections=secs, title=title)
                st.write("Rendering HTML with charts…")
                htm = reporting.to_html(r, narrative, sections=secs, title=title)
                st.write("Rendering PDF…")
                pdf = reporting.to_pdf(r, narrative, sections=secs, title=title)
                status.update(label="Report ready", state="complete", expanded=False)
            st.session_state["_report"] = {"sig": sig, "md": md, "html": htm, "pdf": pdf}
        except Exception as exc:  # noqa: BLE001
            st.error(f"The report could not be generated: {type(exc).__name__}: {exc}", icon=":material/error:")

    with right:
        rep = st.session_state.get("_report")
        if not rep:
            ui.empty_state("No report generated yet", "Configure the report and choose “Generate report”. The preview appears here.")
        else:
            if rep["sig"] != sig:
                st.warning("The analysis or report settings changed after this report was generated. Generate again so it reflects the "
                           "current results — stale reports are never updated silently.", icon=":material/history:")
            d = st.columns(3)
            d[0].download_button("PDF", rep["pdf"], f"{stem}.pdf", "application/pdf", icon=":material/picture_as_pdf:", width="stretch", type="primary")
            d[1].download_button("Markdown", rep["md"], f"{stem}.md", "text/markdown", icon=":material/article:", width="stretch")
            d[2].download_button("HTML", rep["html"], f"{stem}.html", "text/html", icon=":material/html:", width="stretch")
            with st.container(border=True, height=720):
                st.markdown(rep["md"])
            st.caption("Preview of the Markdown export. PDF and HTML add charts drawn from the same values.")

with tab_nb:
    st.markdown("Export a Jupyter notebook that **re-runs** this analysis — loading, profiling, every cleaning step, saved charts, model "
                "training with the same pipeline and seed, permutation importance and clustering — and checks its results against the values "
                "recorded here.")
    tr = state.training()
    cl = state.clustering()
    ctx = {
        "dataset": state.meta().to_dict(), "history": r["cleaning"]["history"],
        "expected_hash": dl.frame_hash(state.df()), "expected_shape": list(state.df().shape),
        "charts": [{"title": ch["title"], "spec": ch["spec"]} for ch in state.charts()],
        "train_config": st.session_state.get("train_config") if tr is not None else None,
        "training": tr.results if tr is not None else None, "importance": state.importance(),
        "clustering": cl["result"] if cl else None, "findings": r["insights"],
    }
    if len(state.analysis_df()) < len(state.df()) and tr is not None:
        st.warning("Models were trained on a seeded sample of a large dataset; the notebook trains on all rows, so its metrics will differ.",
                   icon=":material/warning:")
    inc = st.radio("Dataset handling", ["Bundle: notebook + dataset + requirements (zip)", "Notebook only (load the data from a path you set)"],
                   key="nb_mode", help="The bundle contains your data; only share it where that is allowed.")
    if st.button("Build notebook", type="primary", icon=":material/code_blocks:", key="nb_build"):
        try:
            with st.spinner("Generating notebook…"):
                if inc.startswith("Bundle"):
                    st.session_state["_nb"] = ("zip", notebook.bundle_zip(ctx, dl.to_csv_bytes(state.raw_df())), state.data_key())
                else:
                    st.session_state["_nb"] = ("ipynb", notebook.notebook_bytes(ctx), state.data_key())
        except Exception as exc:  # noqa: BLE001
            st.error(f"The notebook could not be generated: {exc}", icon=":material/error:")
    nb = st.session_state.get("_nb")
    if nb and nb[2] == state.data_key():
        if nb[0] == "zip":
            st.download_button("Download analysis bundle (.zip)", nb[1], f"{stem}_analysis.zip", "application/zip",
                               icon=":material/folder_zip:", type="primary")
        else:
            st.download_button("Download notebook (.ipynb)", nb[1], f"{stem}.ipynb", "application/x-ipynb+json", icon=":material/download:",
                               type="primary")
            st.caption(f"Set DATA_PATH in the first cell to your copy of the original file (sha256 {state.meta().file_hash[:12]}…).")
    st.caption("The notebook imports only pandas, numpy, scikit-learn, imbalanced-learn, scipy and plotly — never DataStory itself. "
               "Fitted models are re-trained, not unpickled.")
