"""Step 1 — upload or pick a dataset, then inspect it."""
from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import streamlit as st

from components import ui
from core import data_loader as dl
from core import state

ASSETS = Path(__file__).resolve().parents[1] / "assets"


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
        st.error(str(exc))
        return
    state.set_dataset(frame, meta)
    st.session_state["_loaded_upload"] = (file.name, file.size)
    st.toast(f"Loaded {meta.name}: {len(frame):,} rows × {frame.shape[1]} columns", icon=":material/check_circle:")
    st.rerun()


def _loader() -> None:
    left, right = st.columns([0.9, 1.6], gap="large")
    with left:
        st.subheader("Upload your data")
        file = st.file_uploader("CSV, TSV or Excel", type=[e.lstrip(".") for e in dl.ALLOWED_EXTENSIONS],
                                key="uploader", help=f"Up to {dl.MAX_UPLOAD_MB} MB. Delimiter and encoding are detected automatically.")
        if file is not None and st.session_state.get("_loaded_upload") != (file.name, file.size):
            _load_upload(file)
        st.caption("Files are processed in this session's memory only and are not stored on disk.")
    with right:
        st.subheader("Or try a sample dataset")
        dims = _sample_dims()
        ids = list(dl.SAMPLES)
        for start in range(0, len(ids), 5):
            cols = st.columns(5)
            for col, sid in zip(cols, ids[start:start + 5]):
                s = dl.SAMPLES[sid]
                with col, st.container(border=True):
                    pill = '<span class="ds-pill syn">Synthetic · demo only</span>' if s.synthetic else '<span class="ds-pill ok">scikit-learn bundled</span>'
                    st.html(f'<div class="ds-sample"><div class="name">{html.escape(s.name)}</div>'
                            f'<div class="desc">{html.escape(s.description)}</div>{pill}'
                            f'<div class="dims" style="margin-top:6px">{dims[sid][0]:,} rows · {dims[sid][1]} columns</div></div>')
                    if st.button("Load", key=f"load_{sid}", width="stretch"):
                        try:
                            frame, meta = dl.load_sample(sid)
                        except dl.DataLoadError as exc:
                            st.error(str(exc))
                        else:
                            state.set_dataset(frame, meta)
                            st.rerun()
        st.caption(f"Titanic, Customer Churn, Bike Sharing and Student Performance are each a **{dl.SYNTHETIC_LABEL}**: "
                   "generated values in a familiar column layout, not the real public datasets.")


def _hero() -> None:
    left, right = st.columns([1.15, 1], gap="large", vertical_alignment="center")
    with left:
        st.html('<span class="ds-eyebrow">AI-powered data analysis</span>'
                '<div class="hero-title">Data Analysis Made <span class="hero-script">Simple</span>,<br>'
                'Insights Made <span class="hero-script">Clear</span>.</div>'
                '<p class="hero-lead">Upload a dataset and DataStory AI profiles it, helps you clean it, explores it with '
                'interactive charts, trains and compares models, and writes a report — with every number computed '
                'from your data.</p>')
    with right:
        img = ASSETS / "hero_workspace.png"
        if img.exists():
            st.image(str(img), width="stretch")


def _overview() -> None:
    r = state.results()
    p, m, frame = r["profile"], state.meta(), state.df()
    ui.page_header("Step 1 · Dataset", m.name, m.description or "Preview, column types and basic statistics for the active dataset.")
    ui.synthetic_notice()
    for note in m.notes:
        st.info(note, icon=":material/info:")
    kc = p["kind_counts"]
    c = st.columns(6)
    c[0].metric("Rows", f"{p['rows']:,}")
    c[1].metric("Columns", f"{p['columns']:,}")
    c[2].metric("Missing cells", f"{p['missing_cells']:,}", help=f"{p['missing_pct']:.2f}% of all cells")
    c[3].metric("Duplicate rows", f"{p['duplicate_rows']:,}")
    c[4].metric("Numeric / categorical", f"{kc['numeric']} / {kc['categorical']}")
    c[5].metric("Datetime / text", f"{kc['datetime']} / {kc['text']}")

    tab_prev, tab_cols, tab_stats, tab_meta = st.tabs(["Preview", "Columns & types", "Basic statistics", "Metadata & downloads"])
    with tab_prev:
        n = st.slider("Rows to preview", 5, min(200, max(5, len(frame))), min(20, len(frame)), key="preview_n") if len(frame) > 5 else len(frame)
        st.dataframe(frame.head(n), width="stretch", height=min(420, 38 * (n + 1)))
        st.caption(f"Showing the first {n:,} of {len(frame):,} rows.")
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
        st.caption("Types are inferred from values: numeric, categorical, datetime (parseable dates) or text (long or mostly unique strings). "
                   "Convert a mis-detected column on the Data Cleaning step.")
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
        with a:
            st.markdown(f"**Source:** {'Built-in sample' if m.source == 'sample' else 'User upload'}  \n"
                        f"**Provenance:** {m.provenance}  \n"
                        f"**File size:** {m.file_size_kb:,.1f} KB · **In-memory size:** {p['memory_mb']:.2f} MB  \n"
                        f"**Storage dtypes:** " + ", ".join(f"{k} × {v}" for k, v in p["dtype_counts"].items()))
        with b:
            st.download_button("Download original dataset (CSV)", dl.to_csv_bytes(state.raw_df()), f"{Path(m.name).stem}_original.csv",
                               "text/csv", width="stretch", icon=":material/download:")
            st.download_button("Download current (cleaned) dataset (CSV)", dl.to_csv_bytes(frame), f"{Path(m.name).stem}_cleaned.csv",
                               "text/csv", width="stretch", icon=":material/download:", type="primary",
                               help="Identical to the original until you apply cleaning steps.")


if state.has_data():
    _overview()
    st.page_link("views/quality.py", label="Continue to Data Health", icon=":material/arrow_forward:")
    with st.expander("Load a different dataset", icon=":material/swap_horiz:"):
        st.warning("Loading another dataset replaces the current one and clears its cleaning history, models and chat.", icon=":material/warning:")
        _loader()
else:
    _hero()
    st.divider()
    _loader()
