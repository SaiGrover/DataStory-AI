"""Step 1 — load data (upload, sample or saved session), then a dataset-aware overview."""
from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import streamlit as st

from components import ui
from core import data_loader as dl
from core import overview as ov
from core import session_io, state
from core.formatting import pct

ASSETS = Path(__file__).resolve().parents[1] / "assets"
TASK = {"titanic": "Classification", "churn": "Classification", "bike": "Regression", "student": "Regression", "iris": "Classification",
        "wine": "Classification", "breast_cancer": "Classification", "diabetes": "Regression", "digits": "Classification",
        "linnerud": "Regression"}


@st.cache_data(show_spinner=False)
def _sample_dims() -> dict[str, tuple[int, int]]:
    out = {}
    for sid, s in dl.SAMPLES.items():
        try:
            with open(dl.SAMPLES_DIR / s.file, encoding="utf-8") as fh:
                header = fh.readline()
                rows = sum(1 for _ in fh)
            out[sid] = (rows, header.count(",") + 1)
        except OSError:
            out[sid] = (0, 0)
    return out


def _load_upload(file) -> None:
    try:
        with st.spinner(f"Reading {file.name}…"):
            frame, meta = dl.load_upload(file.name, file.getvalue())
    except dl.DataLoadError as exc:
        st.error(str(exc), icon=":material/error:")
        return
    state.set_dataset(frame, meta)
    st.session_state["_loaded_upload"] = (file.name, file.size)
    st.toast(f"Loaded {meta.name}: {len(frame):,} rows × {frame.shape[1]} columns", icon=":material/check_circle:")
    st.rerun()


def _upload_panel() -> None:
    with st.container(border=True):
        st.markdown("#### Upload your data")
        file = st.file_uploader("Drop a CSV, TSV or Excel file", type=[e.lstrip(".") for e in dl.ALLOWED_EXTENSIONS], key="uploader",
                                help=f"Up to {dl.MAX_UPLOAD_MB} MB. Delimiter and text encoding are detected automatically.")
        if file is not None and st.session_state.get("_loaded_upload") != (file.name, file.size):
            _load_upload(file)
        st.caption(f"CSV · TSV · TXT · XLSX · XLS — max {dl.MAX_UPLOAD_MB} MB. Files stay in this browser session's memory and are never "
                   "written to the server. Uploaded data is labelled “provenance unverified” — DataStory never guesses whether it is real or synthetic.")


def _samples_panel() -> None:
    st.markdown("#### Or start from a sample")
    dims = _sample_dims()
    ids = list(dl.SAMPLES)
    syn = [i for i in ids if dl.SAMPLES[i].synthetic]
    real = [i for i in ids if not dl.SAMPLES[i].synthetic]
    for group, label in ((real, "Verified public datasets (bundled with scikit-learn)"), (syn, f"Synthetic demonstration datasets — {dl.SYNTHETIC_LABEL}")):
        st.caption(label)
        for start in range(0, len(group), 5):
            cols = st.columns(5)
            for col, sid in zip(cols, group[start:start + 5]):
                s = dl.SAMPLES[sid]
                with col, st.container(border=True):
                    pill = ('<span class="ds-pill syn">Synthetic · demo only</span>' if s.synthetic
                            else '<span class="ds-pill ok">Verified · scikit-learn</span>')
                    st.html(f'<div class="ds-sample"><div class="name">{html.escape(s.name)}</div>'
                            f'<div class="desc">{html.escape(s.description)}</div>{pill}'
                            f'<div class="dims">{dims[sid][0]:,} rows · {dims[sid][1]} cols · {TASK.get(sid, "")}</div></div>')
                    if st.button("Load", key=f"load_{sid}", width="stretch", help=s.provenance):
                        try:
                            frame, meta = dl.load_sample(sid)
                        except dl.DataLoadError as exc:
                            st.error(str(exc))
                        else:
                            state.set_dataset(frame, meta)
                            st.rerun()


def _resume_panel() -> None:
    with st.container(border=True):
        st.markdown("#### Resume a saved analysis")
        st.caption("Import a .datastory.zip package exported from the Session menu. Packages are validated (checksums, schema version, "
                   "allowed files only) before anything loads; no code is ever executed from them.")
        pkg = st.file_uploader("Session package", type=["zip"], key="sess_upload")
        if pkg is None:
            return
        try:
            info = session_io.inspect_package(pkg.getvalue())
        except session_io.SessionError as exc:
            st.error(f"This package was rejected: {exc}", icon=":material/gpp_bad:")
            return
        st.markdown(f"**{html.escape(str(info.get('dataset_name')))}** · saved {info.get('created_at')} · "
                    f"{'dataset included' if info.get('include_data') else 'dataset not included'}")
        contents = info.get("contents") or {}
        if contents:
            st.caption(" · ".join(f"{k.replace('_', ' ')}: {v}" for k, v in contents.items()))
        data_file = None
        if not info.get("include_data"):
            data_file = st.file_uploader("The original dataset file (checked against the saved hash)", key="sess_data",
                                         type=[e.lstrip(".") for e in dl.ALLOWED_EXTENSIONS])
        if st.button("Restore this session", type="primary", icon=":material/restore:", key="sess_restore",
                     disabled=not info.get("include_data") and data_file is None):
            try:
                with st.spinner("Validating and restoring…"):
                    out = session_io.import_session(pkg.getvalue(), data_file.getvalue() if data_file else None,
                                                    data_file.name if data_file else None)
            except session_io.SessionError as exc:
                st.error(f"Could not restore: {exc}", icon=":material/error:")
                return
            state.restore(out["state"])
            st.session_state["_restore_report"] = out["report"]
            st.rerun()


def _hero() -> None:
    left, right = st.columns([1.15, 1], gap="large", vertical_alignment="center")
    with left:
        st.html('<span class="ds-eyebrow">Evidence-based analytics</span>'
                '<div class="hero-title">Data analysis made <span class="hero-script">simple</span>,<br>'
                'insights made <span class="hero-script">clear</span>.</div>'
                '<p class="hero-lead">Load a dataset and DataStory AI works like a careful junior analyst: it profiles the data, '
                'recommends analyses, cleans transparently, charts what matters, trains and explains models, finds segments and '
                'writes reproducible reports — every number traceable to a computation on your data.</p>'
                '<div class="ds-hero-points"><span>Exact cleaning previews</span><span>Leakage-safe models</span>'
                '<span>Evidence-linked reports</span><span>Runnable notebooks</span></div>')
    with right:
        img = ASSETS / "hero_workspace.png"
        if img.exists():
            st.image(str(img), width="stretch")


def _restore_banner() -> None:
    rep = st.session_state.pop("_restore_report", None)
    if not rep:
        return
    st.success("Session restored. " + (f"Restored: {', '.join(rep['restored'])}." if rep.get("restored") else ""), icon=":material/restore:")
    for line in rep.get("not_restored", []):
        st.info(line, icon=":material/info:")
    for w in rep.get("warnings", []):
        st.warning(w, icon=":material/warning:")


def _overview() -> None:
    r = state.results()
    p, m, frame = r["profile"], state.meta(), state.df()
    ov_ = state.overview()
    ui.page_header("Dataset · Overview", m.name, m.description or "What DataStory understands about this dataset, what it supports and what to watch out for.")
    _restore_banner()
    ui.provenance_notice()
    for note in m.notes:
        st.info(note, icon=":material/info:")

    ui.section("Dataset snapshot")
    h = r["quality"]["health"]
    c = st.columns(6)
    c[0].metric("Rows", f"{p['rows']:,}")
    c[1].metric("Columns", f"{p['columns']:,}")
    c[2].metric("Missing cells", f"{p['missing_cells']:,}", help=f"{pct(p['missing_pct'], 2)} of all {p['cells']:,} cells")
    c[3].metric("Duplicate rows", f"{p['duplicate_rows']:,}", help=f"{pct(p['duplicate_pct'], 2)} of rows")
    c[4].metric("Health score", "n/a" if h.get("value") is None else f"{h['value']:.0f}/100", help=h.get("formula"))
    c[5].metric("Version", state.data_version(), help="v0 is the original upload; each applied cleaning step adds a version.")
    st.html(f'<div class="ds-meta">Provenance: <b>{html.escape(r["lineage"]["active_label"])}</b> · source file '
            f'{html.escape(str(m.original_filename or m.name))} · sha256 {html.escape((m.file_hash or "")[:16])}… · loaded {html.escape(m.loaded_at)}'
            + (f' · licence {html.escape(m.license)}' if m.license else "") + "</div>")

    left, right = st.columns([1.25, 1], gap="large")
    with left:
        ui.section("What DataStory understands")
        dom = ov_["domain"]
        st.html('<ul class="ds-list">' + "".join(f"<li>{ui.md_inline(x)}</li>" for x in ov_["understands"]) + "</ul>")
        with st.expander("Domain evidence & correction", icon=":material/category:"):
            st.caption(f"Suggested: **{dom['domain']}** ({dom['confidence']} confidence). " + (dom.get("note") or ""))
            if dom["evidence"]:
                st.caption("Evidence: " + "; ".join(dom["evidence"]))
            options = ["Use suggestion"] + list(ov.DOMAINS) + [ov.GENERAL]
            cur = st.session_state.get("domain_override")
            pick = st.selectbox("Domain", options, index=options.index(cur) if cur in options else 0, key="ov_domain")
            new = None if pick == "Use suggestion" else pick
            if new != cur:
                state.set_domain(new)
                st.rerun()
        ui.section("Recommended analyses")
        recs = ov_["recommendations"]
        cols = st.columns(2)
        for i, rec in enumerate(recs):
            with cols[i % 2], st.container(border=True):
                st.html(f'<div class="ds-rec"><div class="t">{html.escape(rec["title"])}</div><div class="w">{html.escape(rec["detail"])}</div></div>')
                if st.button("Open", key=f"ov_go_{i}", icon=f":material/{rec['icon']}:", width="stretch"):
                    pre = rec.get("preset") or {}
                    if pre.get("target"):
                        state.set_target(pre["target"])
                    if pre.get("eda_tab"):
                        st.session_state["_eda_tab"] = pre["eda_tab"]
                    st.switch_page(rec["page"])
    with right:
        ui.section("Potential risks", str(len(ov_["risks"])))
        if not ov_["risks"]:
            ui.callout("No notable data risks were detected by the profiler.", "na")
        for rk in ov_["risks"]:
            icon = ":material/warning:" if rk["level"] == "warning" else ":material/info:"
            (st.warning if rk["level"] == "warning" else st.info)(f"**{rk['title']}** — {rk['detail']}", icon=icon)
        ui.section("Target candidates")
        tg = ov_["targets"]
        rows = [{"Column": t["column"], "Verdict": t["verdict"], "Task": t["task"] or "—",
                 "Why": (t["reasons"] or t["warnings"] or [""])[0]} for t in tg["candidates"][:8]]
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=min(330, 36 * (len(rows) + 1)))
        st.caption(tg["recommendation"])
        ui.section("Suggested next steps")
        st.html('<ol class="ds-list">' + "".join(f"<li>{html.escape(x)}</li>" for x in ov_["next_steps"]) + "</ol>")

    ui.section("Data")
    tab_prev, tab_cols, tab_stats, tab_meta = st.tabs(["Preview", "Columns & types", "Statistics", "Provenance & downloads"])
    with tab_prev:
        n = st.slider("Rows to preview", 5, min(200, max(5, len(frame))), min(20, len(frame)), key="preview_n") if len(frame) > 5 else len(frame)
        st.dataframe(frame.head(n), width="stretch", height=min(420, 38 * (n + 1)))
        st.caption(f"Showing the first {n:,} of {len(frame):,} rows of version {state.data_version()}.")
    with tab_cols:
        table = pd.DataFrame([{
            "Column": c["name"], "Detected type": c["kind"], "Storage dtype": c["dtype"], "Missing": c["missing"],
            "Missing %": c["missing_pct"], "Distinct": c["unique"],
            "Flags": ", ".join(f for f, cond in (("identifier-like", c["name"] in p["id_candidates"]),
                                                 ("constant", c["name"] in p["constant_columns"]),
                                                 ("near-constant", c["name"] in p["near_constant_columns"])) if cond),
        } for c in p["column_details"]])
        st.dataframe(table, width="stretch", hide_index=True,
                     column_config={"Missing %": st.column_config.ProgressColumn("Missing %", format="%.1f%%", min_value=0, max_value=100)})
        st.caption("Types are inferred from values: numeric, categorical, datetime (parseable dates, including month names) or text "
                   "(long or mostly unique strings). Convert a mis-detected column on the Clean step.")
    with tab_stats:
        num = [c for c in p["column_details"] if c.get("mean") is not None]
        if num:
            st.markdown("**Numeric columns**")
            st.dataframe(pd.DataFrame([{k: c.get(k) for k in ("name", "mean", "std", "min", "q1", "median", "q3", "max", "skew")} for c in num])
                         .rename(columns={"name": "Column"}), width="stretch", hide_index=True)
        other = [c for c in p["column_details"] if c.get("mean") is None and c.get("top_value") is not None]
        if other:
            st.markdown("**Categorical, datetime and text columns**")
            st.dataframe(pd.DataFrame([{"Column": c["name"], "Type": c["kind"], "Distinct": c["unique"], "Most frequent": c["top_value"],
                                        "Count": c["top_count"], "Share %": c["top_share_pct"]} for c in other]),
                         width="stretch", hide_index=True, column_config={"Share %": st.column_config.NumberColumn(format="%.1f%%")})
    with tab_meta:
        a, b = st.columns([1.4, 1])
        lin = r["lineage"]
        with a:
            st.markdown(f"**Category:** {m.provenance_label}  \n**Provenance:** {m.provenance}  \n"
                        + (f"**Source:** {m.source_url}  \n" if m.source_url else "")
                        + f"**Original file:** {m.original_filename or m.name} ({m.file_size_kb:,.1f} KB) · sha256 `{m.file_hash}`  \n"
                        f"**Loaded:** {m.loaded_at} · **Dataset ID:** `{m.dataset_id}`  \n"
                        f"**Active version:** {lin['active_label']}")
            if lin["steps"]:
                st.markdown("**Lineage**\n" + "\n".join(f"- {s}" for s in lin["steps"]))
        with b:
            st.download_button("Original dataset (CSV)", dl.to_csv_bytes(state.raw_df()), f"{Path(m.name).stem}_original.csv",
                               "text/csv", width="stretch", icon=":material/download:")
            st.download_button(f"Current version {state.data_version()} (CSV)", dl.to_csv_bytes(frame),
                               f"{Path(m.name).stem}_{state.data_version()}.csv", "text/csv", width="stretch",
                               icon=":material/download:", type="primary", help="Identical to the original until cleaning steps are applied.")


if state.has_data():
    _overview()
    with st.expander("Load a different dataset or resume a session", icon=":material/swap_horiz:"):
        st.warning("Loading another dataset replaces the current analysis (cleaning, models, charts and chat).", icon=":material/warning:")
        a, b = st.columns(2, gap="large")
        with a:
            _upload_panel()
        with b:
            _resume_panel()
        _samples_panel()
else:
    _hero()
    a, b = st.columns([1.3, 1], gap="large")
    with a:
        _upload_panel()
    with b:
        _resume_panel()
    _samples_panel()
