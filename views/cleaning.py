"""Step 3 — configurable cleaning with preview, apply, history, undo and reset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from components import ui
from core import cleaning, state
from core.data_loader import to_csv_bytes

if not ui.require_data("clean it"):
    st.stop()

ui.page_header("Step 3 · Clean", "Data cleaning & transformation",
               "Choose an operation, preview exactly what it would change, then apply it. The original dataset is never modified.")
ui.synthetic_notice()

frame = state.df()
r = state.results()
p = r["profile"]
kinds = p["kinds"]
all_cols = list(frame.columns)
num_cols = [c for c in all_cols if kinds[c] == "numeric"]
cat_cols = [c for c in all_cols if kinds[c] in ("categorical", "text")]
missing_cols = [c["name"] for c in p["column_details"] if c["missing"]]

if r["cleaning"]["suggestions"]:
    with st.expander(f"Suggestions from the profile ({len(r['cleaning']['suggestions'])})", icon=":material/tips_and_updates:"):
        for s in r["cleaning"]["suggestions"]:
            st.markdown(f"- {s}")
        st.caption("Suggestions point at things worth reviewing. They are not claims that a step will improve any model.")

left, right = st.columns([1, 1.25], gap="large")

with left, st.container(border=True):
    st.subheader("1 · Configure")
    op = st.selectbox("Operation", list(cleaning.OPERATIONS), format_func=lambda k: cleaning.OPERATIONS[k][0], key="cl_op")
    params: dict = {}
    if op == "impute":
        params["strategy"] = st.selectbox("Strategy", ["median", "mean", "mode", "constant", "knn"],
                                          format_func={"median": "Median", "mean": "Mean", "mode": "Mode (most frequent)", "constant": "Constant value", "knn": "KNN (numeric only)"}.get)
        pool = [c for c in missing_cols if c in num_cols] if params["strategy"] in ("mean", "median", "knn") else missing_cols
        params["columns"] = st.multiselect("Columns with missing values", pool, default=pool, key=f"cl_cols_{params['strategy']}")
        if params["strategy"] == "constant":
            params["value"] = st.text_input("Fill value")
        if params["strategy"] == "knn":
            params["n_neighbors"] = st.slider("Neighbours (k)", 1, 15, 5)
            st.caption("Neighbours are found using all numeric columns. Can be slow on large datasets.")
        if not pool:
            st.info("No columns with missing values match this strategy.")
    elif op == "drop_missing_rows":
        params["columns"] = st.multiselect("Consider only these columns (empty = all)", all_cols, default=[])
        params["how"] = st.radio("Drop a row when", ["any", "all"], format_func={"any": "any selected value is missing", "all": "all selected values are missing"}.get, horizontal=True)
    elif op == "drop_sparse_columns":
        params["threshold_pct"] = float(st.slider("Drop columns with more than this % missing", 5, 95, 50, step=5))
    elif op == "drop_duplicates":
        params["columns"] = st.multiselect("Compare only these columns (empty = all)", all_cols, default=[])
        params["keep"] = st.radio("Keep", ["first", "last"], horizontal=True)
    elif op == "convert_type":
        params["columns"] = st.multiselect("Columns", all_cols)
        params["to"] = st.selectbox("Convert to", ["numeric", "integer", "string", "category", "datetime", "boolean"])
        st.caption("Values that cannot be converted become missing; the preview reports how many.")
    elif op == "parse_dates":
        params["columns"] = st.multiselect("Columns", [c for c in all_cols if kinds[c] != "numeric"])
        params["dayfirst"] = st.checkbox("Day comes first (e.g. 31/12/2024)")
    elif op == "normalize_categories":
        params["columns"] = st.multiselect("Columns", cat_cols)
        params["case"] = st.selectbox("Letter case", ["lower", "upper", "title", "keep"])
        params["strip"] = st.checkbox("Trim and collapse whitespace", True)
        mapping_text = st.text_area("Replace values (one `old => new` per line, optional)", height=80)
        params["mapping"] = {a.strip(): b.strip() for a, b in (line.split("=>", 1) for line in mapping_text.splitlines() if "=>" in line)}
    elif op == "clean_text":
        params["columns"] = st.multiselect("Columns", cat_cols)
        c1, c2 = st.columns(2)
        params["strip"] = c1.checkbox("Trim ends", True)
        params["collapse_spaces"] = c1.checkbox("Collapse repeated spaces", True)
        params["lower"] = c2.checkbox("Lowercase")
        params["remove_punctuation"] = c2.checkbox("Remove punctuation")
    elif op == "handle_outliers":
        params["columns"] = st.multiselect("Numeric columns", num_cols, default=[x["column"] for x in p["outlier_columns"]][:10])
        params["method"] = st.radio("Detection", ["iqr", "zscore"], format_func={"iqr": "IQR fences", "zscore": "Z-score"}.get, horizontal=True)
        params["threshold"] = st.number_input("Threshold (IQR multiplier or |z|)", 0.5, 10.0, 1.5 if params["method"] == "iqr" else 3.0, 0.5, key=f"cl_thr_{params['method']}")
        params["action"] = st.radio("Action", ["cap", "null", "remove"], format_func={"cap": "Cap at the bounds", "null": "Set to missing", "remove": "Remove rows"}.get, horizontal=True)
    elif op == "scale":
        params["columns"] = st.multiselect("Numeric columns", num_cols)
        params["method"] = st.selectbox("Method", ["standard", "minmax", "robust"], format_func={"standard": "Standard (z-score)", "minmax": "Min–max (0–1)", "robust": "Robust (median / IQR)"}.get)
        st.caption("Optional and mainly for export. The modelling step scales inside its pipeline using training data only, so do not scale here before training.")
    elif op == "encode":
        params["columns"] = st.multiselect("Categorical columns", [c for c in all_cols if kinds[c] == "categorical"])
        params["method"] = st.radio("Method", ["onehot", "ordinal"], format_func={"onehot": "One-hot", "ordinal": "Ordinal codes"}.get, horizontal=True)
        if params["method"] == "onehot":
            params["drop_first"] = st.checkbox("Drop first level")
        st.caption("Optional and mainly for export. The modelling step encodes categories itself.")
    elif op == "rename_columns":
        col = st.selectbox("Column", all_cols)
        params["mapping"] = {col: st.text_input("New name", key=f"cl_new_{col}")}
    elif op == "drop_columns":
        params["columns"] = st.multiselect("Columns to remove", all_cols)

    if st.button("Preview changes", type="primary", icon=":material/visibility:", width="stretch"):
        try:
            with st.spinner("Computing preview…"):
                new_df, record = cleaning.preview_operation(frame, op, params)
            st.session_state["_cl_preview"] = (new_df, record, st.session_state["_version"])
        except cleaning.CleaningError as exc:
            st.session_state.pop("_cl_preview", None)
            st.error(str(exc))

with right, st.container(border=True):
    st.subheader("2 · Preview & apply")
    prev = st.session_state.get("_cl_preview")
    if prev and prev[2] != st.session_state["_version"]:
        prev = None  # dataset changed since the preview was computed
    if not prev:
        ui.empty_state("Nothing to preview yet", "Configure an operation and choose “Preview changes”.")
    else:
        new_df, rec, _ = prev
        imp = rec["impact"]
        st.markdown(f"**{rec['label']}** — {rec['detail']}")
        c = st.columns(4)
        if rec.get("flagged") is not None:
            c[0].metric("Values flagged", f"{rec['flagged']:,}", help="What the operation targets (e.g. outliers detected, missing values).")
        else:
            c[0].metric("Cells examined", f"{imp['cells_examined']:,}")
        c[1].metric("Values modified", f"{imp['cells_changed']:,}", help="Cells whose value actually differs (null-safe, type-aware: 2019 and 2019.0 are equal).")
        c[2].metric("Rows affected", f"{imp['rows_affected']:,}")
        c[3].metric("Columns affected", f"{imp['columns_affected']:,}")
        c = st.columns(4)
        c[0].metric("Rows removed / added", f"{imp['rows_removed']:,} / {imp['rows_added']:,}")
        c[1].metric("Columns removed / added", f"{len(imp['columns_removed'])} / {len(imp['columns_added'])}")
        c[2].metric("Missing introduced", f"{imp['missing_introduced']:,}")
        c[3].metric("Missing resolved", f"{imp['missing_resolved']:,}")
        if imp["dtype_changes"]:
            st.caption("Type changes: " + "; ".join(f"{k}: {v['before']} → {v['after']}" for k, v in imp["dtype_changes"].items()))
        if imp["columns_added"]:
            st.caption(f"New columns: {', '.join(imp['columns_added'][:8])}{'…' if len(imp['columns_added']) > 8 else ''}")
        if imp["rows_removed"]:
            st.warning(f"This removes {imp['rows_removed']:,} rows. Removed rows are not counted as modified cells; the remaining rows keep their identity.",
                       icon=":material/warning:")
        if not rec["changed"]:
            st.info("This operation would not change the dataset.", icon=":material/info:")
        from core.impact import changed_mask

        changed_rows = [r_ for r_ in frame.index if str(r_) in set(imp["changed_row_labels_sample"])][:8]
        show_rows = changed_rows or list(new_df.index[:8])
        show_cols = [c_ for c_ in new_df.columns if c_ in frame.columns]
        if changed_rows and show_cols:
            before = frame.loc[changed_rows, show_cols]
            after = new_df.loc[changed_rows, show_cols]
            mask = changed_mask(frame, new_df, pd.Index(changed_rows), show_cols)
            st.markdown("**Changed values** (only modified cells are highlighted; rows shown are a sample of affected rows)")
            st.dataframe(after.style.apply(lambda col: ["background-color: rgba(201,139,58,.28)" if m else "" for m in mask[col.name]], axis=0),
                         width="stretch", height=min(330, 38 * (len(changed_rows) + 1)))
            with st.expander("Same rows before the change"):
                st.dataframe(before, width="stretch")
        else:
            st.dataframe(new_df.loc[show_rows[:8]], width="stretch", height=250)
            st.caption("First rows after the operation.")
        if st.button("Apply to dataset", type="primary", icon=":material/check:", disabled=not rec["changed"], width="stretch"):
            had_models = state.training() is not None
            state.apply_cleaning(new_df, rec)
            st.session_state.pop("_cl_preview", None)
            st.toast("Applied." + (" Trained models were cleared because the data changed." if had_models else ""), icon=":material/check_circle:")
            st.rerun()

st.subheader("Before and after")
st.page_link("views/compare.py", label="Open the full original-vs-cleaned comparison", icon=":material/compare_arrows:", disabled=not r["cleaning"]["history"])
o, cur = r["cleaning"]["original"], r["cleaning"]["current"]
st.dataframe(pd.DataFrame([{"": "Original", **o}, {"": "Current", **cur},
                           {"": "Change", **{k: cur[k] - o[k] for k in o}}]).rename(
    columns={"rows": "Rows", "columns": "Columns", "missing_cells": "Missing cells", "duplicate_rows": "Duplicate rows"}),
    width="stretch", hide_index=True)

ui.section("Transformation pipeline", sub="every applied step, in order, with its measured impact")
hist = r["cleaning"]["history"]
if not hist:
    ui.callout("No transformations applied. The working dataset is identical to the original (v0).", "na")
else:
    left_f, right_f = st.columns([1.35, 1], gap="large")
    with left_f:
        steps = []
        for i, h in enumerate(hist, 1):
            imp = h.get("impact") or {}
            chips = [f"v{i}"]
            if imp.get("cells_changed"):
                chips.append(f"{imp['cells_changed']:,} values modified")
            if imp.get("rows_removed"):
                chips.append(f"{imp['rows_removed']:,} rows removed")
            if imp.get("columns_removed") or imp.get("columns_added"):
                chips.append(f"{len(imp.get('columns_removed', []))} cols removed / {len(imp.get('columns_added', []))} added")
            if imp.get("missing_resolved"):
                chips.append(f"{imp['missing_resolved']:,} missing resolved")
            steps.append((h["label"], h["detail"], chips))
        ui.method_flow(steps)
    with right_f:
        o, cur = r["cleaning"]["original"], r["cleaning"]["current"]
        ui.io_panel(
            [("Original (v0)", f"{o['rows']:,} × {o['columns']}"), ("Missing cells", f"{o['missing_cells']:,}"),
             ("Duplicate rows", f"{o['duplicate_rows']:,}")],
            f"{len(hist)}-step cleaning pipeline",
            [(f"Active ({state.data_version()})", f"{cur['rows']:,} × {cur['columns']}"), ("Missing cells", f"{cur['missing_cells']:,}"),
             ("Duplicate rows", f"{cur['duplicate_rows']:,}")],
            "Counts computed from both versions on this page load. The original is never modified.")
    with st.expander("History table"):
        st.dataframe(pd.DataFrame([{
            "#": i, "Operation": h["label"], "What changed": h["detail"],
            "Values modified": (h.get("impact") or {}).get("cells_changed", h.get("cells_changed")),
            "Rows removed": (h.get("impact") or {}).get("rows_removed"), "Applied": h["timestamp"],
        } for i, h in enumerate(hist, 1)]), width="stretch", hide_index=True)
    st.caption("History records what changed in the data. Whether a step helps a model is only known after retraining and comparing scores.")

c1, c2, c3 = st.columns(3)
if c1.button("Undo last step", icon=":material/undo:", disabled=not hist, width="stretch"):
    state.undo_last()
    st.rerun()
if c2.button("Reset to original", icon=":material/restart_alt:", disabled=not hist, width="stretch"):
    state.reset_cleaning()
    st.rerun()
c3.download_button("Download cleaned dataset (CSV)", to_csv_bytes(frame), f"{Path(state.meta().name).stem}_cleaned.csv", "text/csv",
                   icon=":material/download:", width="stretch")
