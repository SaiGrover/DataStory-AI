"""Optional — original (v0) vs the active cleaned version."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from components import ui
from core import comparison, state
from core.formatting import num

if not ui.require_data("compare versions"):
    st.stop()

ui.page_header("Optional · Compare", "Original vs cleaned data",
               f"The original upload (v0) is never modified. This compares it with the active version {state.data_version()}.")
ui.provenance_notice()

hist = state.results()["cleaning"]["history"]
if not hist:
    ui.empty_state("Nothing to compare yet", "Apply at least one cleaning step; the original and current data are identical until then.")
    st.page_link("views/cleaning.py", label="Go to Clean", icon=":material/auto_fix_high:")
    st.stop()


@st.cache_data(show_spinner="Comparing versions…", max_entries=4)
def _compare(_orig: pd.DataFrame, _cur: pd.DataFrame, key: str):
    return comparison.compare(_orig, _cur)


orig, cur = state.raw_df(), state.df()
cmp = _compare(orig, cur, f"{state.data_key()}|{state.meta().dataset_id}")
imp = cmp["impact"]

c = st.columns(5)
c[0].metric("Rows", f"{len(cur):,}", f"{len(cur) - len(orig):,}" if len(cur) != len(orig) else None, delta_color="off")
c[1].metric("Values modified", f"{imp['cells_changed']:,}", help="Cells on rows and columns present in both versions whose value differs.")
c[2].metric("Rows removed", f"{imp['rows_removed']:,}")
c[3].metric("Missing introduced / resolved", f"{imp['missing_introduced']:,} / {imp['missing_resolved']:,}")
c[4].metric("Columns removed / added", f"{len(imp['columns_removed'])} / {len(imp['columns_added'])}")
st.caption("Row identity is preserved through cleaning, so removed rows are counted as removals, not as changed cells on shifted rows.")

tabs = st.tabs(["Summary", "Columns", "Distributions", "Missing values", "Audit trail"])
with tabs[0]:
    ov = cmp["overview"].copy()
    st.dataframe(ov, width="stretch", hide_index=True, column_config={
        "original": st.column_config.NumberColumn(format="%.2f"), "current": st.column_config.NumberColumn(format="%.2f"),
        "change": st.column_config.NumberColumn(format="%+.2f")})
    if imp["changed_by_column"]:
        t = pd.DataFrame({"column": list(imp["changed_by_column"]), "values modified": list(imp["changed_by_column"].values())})
        st.dataframe(t.sort_values("values modified", ascending=False), width="stretch", hide_index=True)
    if imp["dtype_changes"]:
        st.caption("Type changes: " + "; ".join(f"{k}: {v['before']} → {v['after']}" for k, v in imp["dtype_changes"].items()))
    if cmp["removed_columns"] or cmp["added_columns"]:
        st.caption(f"Removed columns: {', '.join(cmp['removed_columns']) or '—'} · Added columns: {', '.join(cmp['added_columns']) or '—'}")
with tabs[1]:
    cols = cmp["columns"]
    show = st.radio("Show", ["Changed columns only", "All columns"], horizontal=True, key="cmp_show")
    if show == "Changed columns only":
        changed = set(imp["changed_by_column"]) | set(imp["dtype_changes"])
        cols = cols[cols["column"].isin(changed) | (cols["missing before"] != cols["missing after"])]
    st.dataframe(cols, width="stretch", hide_index=True)
with tabs[2]:
    common = [c for c in orig.columns if c in cur.columns]
    col = st.selectbox("Column", common, key="cmp_col",
                       index=common.index(next(iter(imp["changed_by_column"]), common[0])) if common else 0)
    kb = cmp["kinds_before"].get(col)
    if kb == "numeric" and cmp["kinds_after"].get(col) == "numeric":
        fig = comparison.distribution_overlay(orig, cur, col)
        if fig:
            ui.chart(fig, "cmp_dist")
        a, b = pd.to_numeric(orig[col], errors="coerce"), pd.to_numeric(cur[col], errors="coerce")
        ui.interpretation(f"**{col}**: mean {num(a.mean())} → {num(b.mean())}, median {num(a.median())} → {num(b.median())}, "
                          f"standard deviation {num(a.std())} → {num(b.std())}, range [{num(a.min())}, {num(a.max())}] → [{num(b.min())}, {num(b.max())}].")
    else:
        fig, t = comparison.category_shift(orig, cur, col)
        ui.chart(fig, "cmp_cat")
        st.dataframe(t, width="stretch", hide_index=True)
with tabs[3]:
    fig = comparison.missing_comparison(orig, cur)
    if fig is None:
        ui.callout("Neither version has missing values.", "na")
    else:
        ui.chart(fig, "cmp_miss")
with tabs[4]:
    st.dataframe(pd.DataFrame([{
        "Version": f"v{i}", "Operation": h["label"], "What changed": h["detail"],
        "Values modified": (h.get("impact") or {}).get("cells_changed", h.get("cells_changed")),
        "Rows removed": (h.get("impact") or {}).get("rows_removed"), "Applied": h["timestamp"],
    } for i, h in enumerate(hist, 1)]), width="stretch", hide_index=True)
    c1, c2 = st.columns(2)
    if c1.button("Undo last step", icon=":material/undo:", width="stretch"):
        state.undo_last()
        st.rerun()
    if c2.button("Restore original (v0)", icon=":material/restart_alt:", width="stretch"):
        state.reset_cleaning()
        st.rerun()
