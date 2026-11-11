"""Step 4 — interactive exploratory analysis. Every chart is built from the active dataset."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from components import ui
from core import eda, state
from core.profiler import correlation_pairs

if not ui.require_data("explore it"):
    st.stop()

ui.page_header("Step 4 · EDA", "Exploratory data analysis",
               "Pick columns and chart options. Interpretations under each chart restate values computed from the data shown.")
ui.synthetic_notice()

full = state.df()
frame = state.analysis_df()
r = state.results()
p = r["profile"]
kinds = p["kinds"]
details = {c["name"]: c for c in p["column_details"]}
num = [c for c in frame.columns if kinds[c] == "numeric"]
cat = [c for c in frame.columns if kinds[c] == "categorical" or (kinds[c] == "numeric" and details[c]["unique"] <= 12)]
pure_cat = [c for c in frame.columns if kinds[c] == "categorical"]
dates = [c for c in frame.columns if kinds[c] == "datetime"]
NONE = ui.NONE
if len(frame) < len(full):
    st.info(f"Charts use a seeded random sample of {len(frame):,} of {len(full):,} rows.", icon=":material/info:")


def opt(label, options, key):
    v = st.selectbox(label, [NONE] + options, key=key)
    return None if v == NONE else v


tabs = st.tabs(["Univariate", "Bivariate", "Multivariate", "Missing & outliers", "Time series", "Target relationships"])

# ------------------------------------------------------------- univariate
with tabs[0]:
    c1, c2 = st.columns([1, 3], gap="large")
    with c1:
        col = st.selectbox("Column", list(frame.columns), key="u_col")
        k = kinds[col]
        if k == "numeric":
            view = st.radio("Chart", ["Histogram", "Box plot", "Violin plot"], key="u_view")
            bins = st.slider("Bins", 5, 100, 30, key="u_bins") if view == "Histogram" else 30
            kde = st.checkbox("Overlay KDE (density)", key="u_kde") if view == "Histogram" else False
        else:
            top_n = st.slider("Show top categories", 3, 40, 15, key="u_top")
    with c2:
        if k == "numeric":
            fig = ui.safe(eda.histogram, frame, col, bins, None, kde) if view == "Histogram" else ui.safe(
                eda.box_or_violin, frame, col, None, "violin" if view == "Violin plot" else "box")
            if fig:
                ui.chart(fig, "u_fig")
                ui.interpretation(eda.interpret_numeric(details[col]))
        elif k == "datetime":
            d = pd.to_datetime(frame[col], errors="coerce", format="mixed").dropna()
            if d.empty:
                ui.callout("No parseable dates in this column.", "na")
            else:
                ui.chart(eda.style(eda.px.histogram(x=d, nbins=40, color_discrete_sequence=[eda.SAGE]), f"Records over {col}"), "u_fig_dt")
                ui.interpretation(f"**{col}** spans {d.min():%Y-%m-%d} to {d.max():%Y-%m-%d} across {len(d):,} dated rows.")
        else:
            out = ui.safe(eda.category_counts, frame, col, top_n)
            if out:
                ui.chart(out[0], "u_fig_cat")
                ui.interpretation(eda.interpret_counts(col, out[1], details[col]["unique"]))

# -------------------------------------------------------------- bivariate
with tabs[1]:
    mode = st.radio("Compare", ["Numeric vs numeric", "Numeric by category", "Grouped aggregation", "Category vs category"], horizontal=True, key="b_mode")
    if mode == "Numeric vs numeric":
        if len(num) < 2:
            ui.callout("At least two numeric columns are needed for a scatter plot.", "na")
        else:
            c1, c2, c3 = st.columns(3)
            x = c1.selectbox("X", num, key="b_x")
            y = c2.selectbox("Y", [c for c in num if c != x], key="b_y")
            color = c3.selectbox("Colour by", [NONE] + cat, key="b_color")
            fig = ui.safe(eda.scatter, frame, x, y, None if color == NONE else color)
            if fig:
                ui.chart(fig, "b_scatter")
                ui.interpretation(eda.interpret_scatter(frame, x, y))
    elif mode == "Numeric by category":
        if not num or not cat:
            ui.callout("This view needs one numeric column and one categorical (or low-cardinality) column.", "na")
        else:
            c1, c2, c3 = st.columns(3)
            y = c1.selectbox("Numeric column", num, key="b_ny")
            g = c2.selectbox("Group by", [c for c in cat if c != y] or cat, key="b_ng")
            kind = c3.radio("Chart", ["box", "violin"], horizontal=True, key="b_nkind")
            if frame[g].nunique() > 30:
                ui.callout(f"'{g}' has {frame[g].nunique()} groups; choose a column with at most 30.", "na")
            else:
                fig = ui.safe(eda.box_or_violin, frame, y, g, kind)
                if fig:
                    ui.chart(fig, "b_box")
                    out = ui.safe(eda.grouped_aggregation, frame, g, y, "median", 30)
                    if out:
                        ui.interpretation(eda.interpret_groups(out[1], g, y, "median"))
    elif mode == "Grouped aggregation":
        if not num or not cat:
            ui.callout("This view needs one numeric column and one categorical (or low-cardinality) column.", "na")
        else:
            c1, c2, c3, c4 = st.columns(4)
            g = c1.selectbox("Group by", cat, key="b_gg")
            v = c2.selectbox("Value", [c for c in num if c != g] or num, key="b_gv")
            agg = c3.selectbox("Aggregation", eda.AGGREGATIONS, key="b_gagg")
            top = c4.slider("Max groups", 3, 40, 20, key="b_gtop")
            out = ui.safe(eda.grouped_aggregation, frame, g, v, agg, top)
            if out:
                ui.chart(out[0], "b_group")
                ui.interpretation(eda.interpret_groups(out[1], g, v, agg))
                st.dataframe(out[1], width="stretch", hide_index=True)
    else:
        if len(cat) < 2:
            ui.callout("At least two categorical (or low-cardinality) columns are needed.", "na")
        else:
            c1, c2 = st.columns(2)
            a = c1.selectbox("Rows (category)", cat, key="b_ca")
            b = c2.selectbox("Split by", [c for c in cat if c != a], key="b_cb")
            if frame[b].nunique() > 20:
                ui.callout(f"'{b}' has {frame[b].nunique()} values; choose a column with at most 20 to split by.", "na")
            else:
                out = ui.safe(eda.category_vs_class, frame, a, b)
                if out:
                    ui.chart(out[0], "b_catcat")
                    st.dataframe(out[1].round(1), width="stretch")
                    st.caption(f"Each row shows the percentage mix of {b} within that {a} value.")

# ----------------------------------------------------------- multivariate
with tabs[2]:
    usable = [c for c in num if c not in p["id_candidates"] and details[c]["unique"] > 1]
    if len(usable) < 2:
        ui.callout("At least two numeric columns with variation are needed for correlation analysis.", "na")
    else:
        c1, c2 = st.columns([3, 1])
        sel = c1.multiselect("Numeric columns", usable, default=usable[:12], key="m_cols")
        method = c2.selectbox("Method", ["pearson", "spearman"], key="m_method")
        if len(sel) < 2:
            ui.callout("Select at least two columns.", "na")
        else:
            out = ui.safe(eda.correlation_heatmap, frame, sel, method)
            if out:
                ui.chart(out[0], "m_heat")
                if method == "pearson":
                    pairs = correlation_pairs(frame[sel], {c: "numeric" for c in sel})
                    ui.interpretation(eda.interpret_correlations(pairs))
                else:
                    ui.interpretation("Spearman correlation measures monotonic (rank) association. " + eda.CAUSAL_NOTE)
            st.markdown("**Scatter matrix**")
            sm = st.multiselect("Columns (2–5)", sel, default=sel[:4], max_selections=5, key="m_sm")
            color = opt("Colour by", pure_cat, "m_smcolor")
            if len(sm) >= 2:
                fig = ui.safe(eda.scatter_matrix, frame, sm, color)
                if fig:
                    ui.chart(fig, "m_sm_fig")

# ---------------------------------------------------- missing & outliers
with tabs[3]:
    fig = eda.missing_bar(p)
    if fig is None:
        st.success("No missing values in any column.", icon=":material/check_circle:")
    else:
        a, b = st.columns(2)
        with a:
            ui.chart(fig, "mo_bar")
        with b:
            ui.chart(eda.missing_matrix(frame), "mo_matrix")
        ui.interpretation(eda.interpret_missing(p))
    st.markdown("**Outlier analysis**")
    st.caption(p["outlier_method"])
    if not p["outlier_columns"]:
        ui.callout("No numeric column has values outside the Tukey fences.", "na")
    else:
        t = pd.DataFrame(p["outlier_columns"]).head(20)
        figo = eda.px.bar(t.iloc[::-1], x="outlier_pct", y="column", orientation="h", color_discrete_sequence=["#A65D57"], hover_data={"outliers": ":,"})
        figo.update_layout(xaxis_title="% of non-missing values flagged", yaxis_title=None)
        ui.chart(eda.style(figo, "Share of values outside the Tukey fences", height=max(260, 28 * len(t) + 90)), "mo_out")
        top = p["outlier_columns"][0]
        ui.interpretation(f"**{top['column']}** has the most flagged values: {top['outliers']:,} ({top['outlier_pct']:.1f}%). "
                          "Flagged values may be influential on means and linear models; they are not necessarily errors.")

# ------------------------------------------------------------ time series
with tabs[4]:
    if not dates:
        ui.callout("No datetime column was detected. If a column holds dates stored as text in an unusual format, use “Parse dates” on the Data Cleaning step.", "na")
    elif not num:
        ui.callout("A numeric column is needed to plot over time.", "na")
    else:
        c1, c2, c3, c4 = st.columns(4)
        dcol = c1.selectbox("Date column", dates, key="t_date")
        v = c2.selectbox("Value", num, key="t_val")
        freq = c3.selectbox("Resample to", ["D", "W", "MS", "QS", "YS"], index=2, format_func={"D": "Day", "W": "Week", "MS": "Month", "QS": "Quarter", "YS": "Year"}.get, key="t_freq")
        agg = c4.selectbox("Aggregation", ["mean", "sum", "median", "min", "max", "count"], key="t_agg")
        out = ui.safe(eda.time_series, frame, dcol, v, freq, agg)
        if out and len(out[1]):
            ui.chart(out[0], "t_fig")
            t = out[1]
            hi, lo = t.loc[t[v].idxmax()], t.loc[t[v].idxmin()]
            ui.interpretation(f"Over {len(t):,} periods from {t[dcol].min():%Y-%m-%d} to {t[dcol].max():%Y-%m-%d}, the {agg} of **{v}** peaks at "
                              f"{hi[v]:.4g} ({hi[dcol]:%Y-%m-%d}) and is lowest at {lo[v]:.4g} ({lo[dcol]:%Y-%m-%d}). "
                              "This describes the recorded periods only; it is not a forecast.")
        elif out:
            ui.callout("No dated numeric values are available for this selection.", "na")

# ------------------------------------------------------------------ target
with tabs[5]:
    target = ui.target_picker("Target column", key="e_target")
    if not target:
        ui.callout("Select a target column to explore how other columns relate to it.", "na")
    else:
        ts = state.results()["profile"]["target"]
        discrete = bool(ts.get("class_counts")) and len(ts["class_counts"]) <= 20
        c1, c2 = st.columns([1, 2], gap="large")
        with c1:
            if ts.get("class_counts"):
                ui.chart(eda.class_balance(ts["class_counts"], target), "e_balance")
            else:
                ui.chart(eda.histogram(frame, target), "e_thist")
            ui.interpretation(eda.interpret_balance(ts, target))
        with c2:
            feats = [c for c in frame.columns if c != target and kinds[c] in ("numeric", "categorical")]
            if not feats:
                ui.callout("No numeric or categorical feature is available to compare with the target.", "na")
            else:
                f = st.selectbox("Feature", feats, key="e_feat")
                f_num = kinds[f] == "numeric" and details[f]["unique"] > 12
                if discrete and f_num:
                    fig = ui.safe(eda.box_or_violin, frame, f, target, "box")
                    if fig:
                        ui.chart(fig, "e_rel")
                        out = ui.safe(eda.grouped_aggregation, frame, target, f, "median", 30)
                        if out:
                            ui.interpretation(eda.interpret_groups(out[1], target, f, "median") + " Differences between groups are observed associations, not causal effects.")
                elif discrete:
                    out = ui.safe(eda.category_vs_class, frame, f, target)
                    if out:
                        ui.chart(out[0], "e_rel")
                        ui.interpretation(f"Bars show the mix of **{target}** within each value of **{f}** (top values by frequency). "
                                          "Differences in mix are observed associations in this dataset, not causal effects.")
                elif f_num and kinds[target] == "numeric":
                    fig = ui.safe(eda.scatter, frame, f, target)
                    if fig:
                        ui.chart(fig, "e_rel")
                        ui.interpretation(eda.interpret_scatter(frame, f, target))
                elif kinds[target] == "numeric":
                    out = ui.safe(eda.grouped_aggregation, frame, f, target, "mean", 30)
                    if out:
                        ui.chart(out[0], "e_rel")
                        ui.interpretation(eda.interpret_groups(out[1], f, target, "mean"))
                else:
                    ui.callout(f"'{target}' has too many distinct values to chart against a feature. Choose a different target.", "na")
