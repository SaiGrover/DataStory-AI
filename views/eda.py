"""Step 4 — exploratory analysis: recommended charts, a natural-language chart builder and manual views."""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from components import ui
from core import charts as ch
from core import eda, nl_chart, state
from core.formatting import pct as _pct
from core.profiler import correlation_pairs

if not ui.require_data("explore it"):
    st.stop()

ui.page_header("Step 4 · Explore", "Exploratory data analysis",
               "Start from charts recommended for this dataset, describe a chart in plain language, or build one yourself. "
               "Every interpretation restates values computed from the data shown.")
ui.provenance_notice()

full = state.df()
frame = state.analysis_df()
r = state.results()
p = r["profile"]
kinds = p["kinds"]
details = {c["name"]: c for c in p["column_details"]}
ids = set(p["id_candidates"])
num = [c for c in frame.columns if kinds[c] == "numeric" and c not in ids]
cat = [c for c in frame.columns if c not in ids and (kinds[c] == "categorical" or (kinds[c] == "numeric" and details[c]["unique"] <= 12))]
pure_cat = [c for c in frame.columns if kinds[c] == "categorical" and c not in ids and details[c]["unique"] <= 30]
dates = [c for c in frame.columns if kinds[c] == "datetime"]
NONE = ui.NONE
if len(frame) < len(full):
    st.info(f"Charts use a seeded random sample of {len(frame):,} of {len(full):,} rows; profile statistics use all rows.", icon=":material/info:")

VIEWS = ["Recommended", "Chart builder", "Saved charts", "Univariate", "Bivariate", "Correlation", "Missing & outliers", "Time series", "Target"]
if "_eda_tab" in st.session_state:
    st.session_state["eda_view"] = st.session_state.pop("_eda_tab")
view = st.segmented_control("View", VIEWS, default="Recommended", key="eda_view", label_visibility="collapsed") or "Recommended"


def opt(label, options, key):
    v = st.selectbox(label, [NONE] + options, key=key)
    return None if v == NONE else v


# ----------------------------------------------------------- recommended
def recommended() -> None:
    recs = ch.recommend(frame, p, target=state.target(), domain=state.overview()["domain_used"], limit=8)
    if not recs:
        ui.empty_state("No recommendations", "There are no columns suitable for automatic charts. Try the chart builder.")
        return
    st.caption(f"{len(recs)} charts ranked by analytical value for this dataset's columns"
               + (f" and the target **{state.target()}**" if state.target() else "") + ". Generate the ones you need — nothing is drawn until you ask.")
    for i, rec in enumerate(recs):
        spec = rec["spec"]
        with st.container(border=True):
            a, b = st.columns([4, 1], vertical_alignment="center")
            with a:
                st.html(f'<div class="ds-rec"><div class="t">{html.escape(rec["title"])} <span class="ds-pill muted">{html.escape(rec["chart_type"])}</span></div>'
                        f'<div class="w">{html.escape(rec["why"])}</div>'
                        f'<div class="m">Columns: {html.escape(", ".join(rec["columns"]) or "all numeric")}'
                        + (f' · Needs: {html.escape("; ".join(rec["prerequisites"]))}' if rec["prerequisites"] else "")
                        + (f' · Limits: {html.escape("; ".join(rec["limitations"]))}' if rec["limitations"] else "") + "</div></div>")
            shown = st.session_state.setdefault("_rec_open", set())
            with b:
                if st.button("Hide chart" if rec["id"] in shown else "Generate chart", key=f"rec_btn_{rec['id']}", width="stretch",
                             type="secondary" if rec["id"] in shown else "primary", icon=":material/insert_chart:"):
                    (shown.discard if rec["id"] in shown else shown.add)(rec["id"])
                    if rec["id"] in shown:
                        state.mark_eda()
                    st.rerun()
            if rec["id"] in shown:
                ui.spec_chart(spec.to_dict(), key=f"rec_{rec['id']}", title=rec["title"], source="Recommendation")


# -------------------------------------------------------- chart builder
def builder() -> None:
    st.caption("Describe a chart — e.g. “average revenue by region”, “distribution of release years”, “attrition rate across departments”, "
               "“monthly revenue”, “correlation heatmap”. The request is parsed into a validated chart specification you can edit; no code is executed.")
    with st.form("nl_form", border=False):
        c1, c2 = st.columns([5, 1], vertical_alignment="bottom")
        q = c1.text_input("Chart request", key="nl_q", placeholder="Show the top 10 categories by sales")
        go_ = c2.form_submit_button("Generate", type="primary", icon=":material/auto_awesome:", width="stretch")
    if go_ and q.strip():
        prev = st.session_state.get("_nl_spec")
        res = nl_chart.parse(q, frame, kinds, state.target(), ch.ChartSpec.from_dict(prev) if prev else None)
        if res.spec is None:
            st.session_state["_nl_err"] = (res.errors, res.clarification)
        else:
            st.session_state["_nl_spec"] = res.spec.to_dict()
            st.session_state["_nl_assume"] = res.assumptions
            st.session_state.pop("_nl_err", None)
            state.mark_eda()
    err = st.session_state.get("_nl_err")
    if err:
        ui.callout("<strong>Couldn't build that chart.</strong> " + html.escape(" ".join(err[0])) + (f"<br>{html.escape(err[1])}" if err[1] else ""), "na")
    spec_d = st.session_state.get("_nl_spec")
    if not spec_d:
        return
    for a in st.session_state.get("_nl_assume", []):
        st.caption(f"Assumption: {a}")
    left, right = st.columns([1, 2.6], gap="large")
    with left, st.container(border=True):
        st.markdown("**Chart specification**")
        cols = [None] + list(frame.columns)
        fmt = lambda c: NONE if c is None else c  # noqa: E731
        new = dict(spec_d)
        new["chart"] = st.selectbox("Chart type", list(ch.CHART_TYPES), index=list(ch.CHART_TYPES).index(spec_d["chart"]),
                                    format_func=ch.CHART_TYPES.get, key="sp_chart")
        new["x"] = st.selectbox("X / category", cols, index=cols.index(spec_d.get("x")), format_func=fmt, key="sp_x")
        new["y"] = st.selectbox("Value (Y)", cols, index=cols.index(spec_d.get("y")), format_func=fmt, key="sp_y")
        new["color"] = st.selectbox("Split / colour", cols, index=cols.index(spec_d.get("color")), format_func=fmt, key="sp_c")
        new["agg"] = st.selectbox("Aggregation", ch.AGGS, index=ch.AGGS.index(spec_d.get("agg", "count")), key="sp_agg")
        if new["chart"] == "line":
            freqs = [None] + list(ch.FREQS)
            new["freq"] = st.selectbox("Time grain", freqs, index=freqs.index(spec_d.get("freq")), key="sp_f",
                                       format_func=lambda f: "auto" if f is None else ch.FREQS[f])
        if new["chart"] in ("bar", "stacked_bar", "box", "violin"):
            new["top_n"] = st.slider("Top-N categories", 3, 50, int(spec_d.get("top_n") or 15), key="sp_top")
            new["other"] = st.checkbox("Group the rest into “Other”", spec_d.get("other", True), key="sp_other")
            new["sort"] = st.radio("Sort", ["desc", "asc", "none"], index=["desc", "asc", "none"].index(spec_d.get("sort", "desc")),
                                   horizontal=True, key="sp_sort")
        if new["chart"] == "histogram":
            new["bins"] = st.slider("Bins", 5, 100, int(spec_d.get("bins", 30)), key="sp_bins")
        with st.expander("Filters" + (f" ({len(spec_d.get('filters') or [])})" if spec_d.get("filters") else "")):
            ftable = pd.DataFrame(spec_d.get("filters") or [], columns=["column", "op", "value"])
            edited = st.data_editor(ftable, num_rows="dynamic", width="stretch", key="sp_filters", column_config={
                "column": st.column_config.SelectboxColumn("Column", options=list(frame.columns), required=True),
                "op": st.column_config.SelectboxColumn("Operator", options=ch.FILTER_OPS, required=True),
                "value": st.column_config.TextColumn("Value", required=True)})
            new["filters"] = [dict(r) for r in edited.dropna().to_dict("records")]
        new["title"] = st.text_input("Title (optional)", spec_d.get("title") or "", key="sp_title") or None
        if new != spec_d:
            errs = ch.validate(frame, ch.ChartSpec.from_dict(new), kinds)
            if errs:
                st.error(" ".join(errs), icon=":material/error:")
            elif st.button("Update chart", type="primary", width="stretch", key="sp_apply"):
                st.session_state["_nl_spec"] = new
                st.rerun()
    with right:
        ui.spec_chart(spec_d, key="nl", source="Chart builder")


# ---------------------------------------------------------- saved charts
def saved() -> None:
    items = state.charts()
    if not items:
        ui.empty_state("No saved charts", "Use “Save & add to report” under any chart. Saved charts are stored as specifications and re-computed "
                                          "from the current data whenever shown.")
        return
    for c in items:
        with st.container(border=True):
            a, b, d = st.columns([5, 1.2, 1], vertical_alignment="center")
            a.markdown(f"**{c['title']}**  \n<span class='ds-meta'>{c['source']} · saved {c['created_at']} on data {c['data_version']}"
                       + (f" — data is now {state.data_version()}, so values are recomputed" if c["data_version"] != state.data_version() else "")
                       + "</span>", unsafe_allow_html=True)
            inc = b.toggle("In report", value=c.get("in_report", True), key=f"inrep_{c['id']}")
            if inc != c.get("in_report", True):
                c["in_report"] = inc
            if d.button("Remove", key=f"rm_{c['id']}", icon=":material/delete:"):
                state.remove_chart(c["id"])
                st.rerun()
            ui.spec_chart(c["spec"], key=f"saved_{c['id']}", title=c["title"], toolbar=False, show_table=False)


# ------------------------------------------------------------ univariate
def univariate() -> None:
    c1, c2 = st.columns([1, 3], gap="large")
    with c1:
        col = st.selectbox("Column", list(frame.columns), key="u_col", format_func=lambda c: f"{c} (identifier)" if c in ids else c)
        k = kinds[col]
        if k == "numeric":
            view_ = st.radio("Chart", ["Histogram", "Box plot", "Violin plot"], key="u_view")
            bins = st.slider("Bins", 5, 100, 30, key="u_bins") if view_ == "Histogram" else 30
        elif k != "datetime":
            top_n = st.slider("Show top categories", 3, 40, 15, key="u_top")
            other = st.checkbox("Group the rest into “Other”", True, key="u_other")
    with c2:
        reason = ch.chart_suitability(frame, col, p) if k != "numeric" else (
            f"'{col}' looks like an identifier (one value per row); its distribution carries no analytical meaning." if col in ids else None)
        if reason:
            d = details[col]
            ui.callout(f"<strong>No frequency chart for this column.</strong> {html.escape(reason)} "
                       f"Instead: {d['unique']:,} distinct values, {d['missing']:,} missing"
                       + (f", e.g. '{html.escape(str(d.get('top_value')))}'" if d.get("top_value") else "") + ".", "na")
            return
        if k == "numeric":
            spec = ch.ChartSpec("histogram", x=col, bins=bins) if view_ == "Histogram" else ch.ChartSpec(
                "violin" if view_ == "Violin plot" else "box", y=col)
            out = ui.spec_chart(spec.to_dict(), key="uni", source="Univariate", show_table=False)
            if out:
                st.caption(eda.interpret_numeric(details[col]).replace("**", ""))
        elif k == "datetime":
            ui.spec_chart(ch.ChartSpec("line", x=col, agg="count", freq="MS").to_dict(), key="uni_dt", source="Univariate")
        else:
            spec = ch.ChartSpec("bar", x=col, top_n=top_n, other=other, explode=bool(ch.multi_value_separator(frame[col])))
            ui.spec_chart(spec.to_dict(), key="uni_cat", source="Univariate")


# -------------------------------------------------------------- bivariate
def bivariate() -> None:
    mode = st.radio("Compare", ["Numeric vs numeric", "Numeric by category", "Grouped aggregation", "Category vs category"], horizontal=True, key="b_mode")
    if mode == "Numeric vs numeric":
        if len(num) < 2:
            return ui.callout("At least two numeric columns are needed for a scatter plot.", "na")
        c1, c2, c3 = st.columns(3)
        x = c1.selectbox("X", num, key="b_x")
        y = c2.selectbox("Y", [c for c in num if c != x], key="b_y")
        color = c3.selectbox("Colour by", [NONE] + pure_cat, key="b_color")
        ui.spec_chart(ch.ChartSpec("scatter", x=x, y=y, color=None if color == NONE else color).to_dict(), key="biv_sc", source="Bivariate")
    elif mode == "Numeric by category":
        if not num or not cat:
            return ui.callout("This view needs one numeric column and one categorical (or low-cardinality) column.", "na")
        c1, c2, c3, c4 = st.columns(4)
        y = c1.selectbox("Numeric column", num, key="b_ny")
        g = c2.selectbox("Group by", [c for c in cat if c != y] or cat, key="b_ng")
        kind = c3.radio("Chart", ["box", "violin"], horizontal=True, key="b_nkind")
        top = c4.slider("Max groups", 3, 30, 12, key="b_ntop")
        ui.spec_chart(ch.ChartSpec(kind, x=g, y=y, top_n=top).to_dict(), key="biv_box", source="Bivariate")
    elif mode == "Grouped aggregation":
        if not num or not cat:
            return ui.callout("This view needs one numeric column and one categorical (or low-cardinality) column.", "na")
        c1, c2, c3, c4 = st.columns(4)
        g = c1.selectbox("Group by", cat, key="b_gg")
        v = c2.selectbox("Value", [c for c in num if c != g] or num, key="b_gv")
        agg = c3.selectbox("Aggregation", ["mean", "median", "sum", "min", "max"], key="b_gagg")
        top = c4.slider("Max groups", 3, 40, 15, key="b_gtop")
        ui.spec_chart(ch.ChartSpec("bar", x=g, y=v, agg=agg, top_n=top, other=agg == "sum").to_dict(), key="biv_grp", source="Bivariate")
    else:
        if len(pure_cat) < 2:
            return ui.callout("At least two categorical columns with ≤30 values are needed.", "na")
        c1, c2, c3 = st.columns(3)
        a = c1.selectbox("Rows (category)", pure_cat, key="b_ca")
        b = c2.selectbox("Split by", [c for c in pure_cat if c != a], key="b_cb")
        norm = c3.radio("Show", ["Shares", "Counts"], horizontal=True, key="b_cnorm") == "Shares"
        ui.spec_chart(ch.ChartSpec("stacked_bar", x=a, color=b, normalize=norm).to_dict(), key="biv_cc", source="Bivariate")


# ----------------------------------------------------------- correlation
def correlation() -> None:
    usable = [c for c in num if details[c]["unique"] > 1]
    if len(usable) < 2:
        return ui.callout("At least two numeric columns with variation are needed for correlation analysis.", "na")
    c1, c2 = st.columns([3, 1])
    sel = c1.multiselect("Numeric columns", usable, default=usable[:12], key="m_cols")
    method = c2.selectbox("Method", ["pearson", "spearman"], key="m_method")
    if len(sel) < 2:
        return ui.callout("Select at least two columns.", "na")
    if method == "pearson":
        ui.spec_chart(ch.ChartSpec("heatmap", columns=sel).to_dict(), key="corr", source="Correlation")
    else:
        out = ui.safe(eda.correlation_heatmap, frame, sel, method)
        if out:
            ui.chart(out[0], "m_heat_sp")
            ui.interpretation("Spearman correlation measures monotonic (rank) association. " + eda.CAUSAL_NOTE)
    with st.expander("Scatter matrix"):
        sm = st.multiselect("Columns (2–5)", sel, default=sel[:4], max_selections=5, key="m_sm")
        color = opt("Colour by", pure_cat, "m_smcolor")
        if len(sm) >= 2:
            fig = ui.safe(eda.scatter_matrix, frame, sm, color)
            if fig:
                ui.chart(fig, "m_sm_fig")


# ---------------------------------------------------- missing & outliers
def missing_outliers() -> None:
    if not p["missing_cells"]:
        st.success("No missing values in any column.", icon=":material/check_circle:")
    else:
        a, b = st.columns(2)
        with a:
            ui.spec_chart(ch.ChartSpec("missing").to_dict(), key="miss", source="Missing values", show_table=False)
        with b:
            fig = eda.missing_matrix(frame)
            if fig:
                ui.chart(fig, "mo_matrix")
        ui.interpretation(eda.interpret_missing(p))
    ui.section("Outlier analysis")
    st.caption(p["outlier_method"])
    if not p["outlier_columns"]:
        return ui.callout("No numeric column has values outside the Tukey fences.", "na")
    t = pd.DataFrame(p["outlier_columns"]).head(20)
    figo = eda.px.bar(t.iloc[::-1], x="outlier_pct", y="column", orientation="h", color_discrete_sequence=["#A65D57"], hover_data={"outliers": ":,"})
    figo.update_layout(xaxis_title="% of non-missing values flagged", yaxis_title=None)
    ui.chart(eda.style(figo, "Share of values outside the Tukey fences", height=max(260, 28 * len(t) + 90)), "mo_out")
    top = p["outlier_columns"][0]
    ui.interpretation(f"**{top['column']}** has the most flagged values: {top['outliers']:,} ({_pct(top['outlier_pct'], 1)}). "
                      "Flagged values may be influential on means and linear models; they are not necessarily errors.")


# ------------------------------------------------------------ time series
def time_series() -> None:
    years = [c for c in num if "year" in c.lower() and ch.is_yearlike(frame[c])]
    axes = dates + years
    if not axes:
        return ui.callout("No date or year column was detected. If dates are stored as text in an unusual format, use “Parse dates” on the Clean step.", "na")
    c1, c2, c3, c4 = st.columns(4)
    dcol = c1.selectbox("Time column", axes, key="t_date")
    v = c2.selectbox("Value", ["(row count)"] + [c for c in num if c != dcol], key="t_val")
    agg = c3.selectbox("Aggregation", ["sum", "mean", "median", "min", "max"], key="t_agg", disabled=v == "(row count)")
    freq = c4.selectbox("Resample to", list(ch.FREQS), index=2, format_func=ch.FREQS.get, key="t_freq", disabled=dcol not in dates)
    spec = ch.ChartSpec("line", x=dcol, y=None if v == "(row count)" else v, agg="count" if v == "(row count)" else agg,
                        freq=freq if dcol in dates else None)
    ui.spec_chart(spec.to_dict(), key="ts", source="Time series")


# ------------------------------------------------------------------ target
def target_view() -> None:
    target = ui.target_picker("Target column", key="e_target")
    if not target:
        return ui.callout("Select a target column to explore how other columns relate to it.", "na")
    if target in ids:
        return ui.callout(f"'{html.escape(target)}' looks like an identifier, so relating other columns to it is not meaningful. Choose another target.", "err")
    ts = state.results()["profile"]["target"]
    discrete = bool(ts.get("class_counts")) and len(ts["class_counts"]) <= 20
    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        if ts.get("class_counts") and len(ts["class_counts"]) <= 30:
            ui.chart(eda.class_balance(ts["class_counts"], target), "e_balance")
            ui.interpretation(eda.interpret_balance(ts, target))
        elif kinds[target] == "numeric":
            ui.spec_chart(ch.ChartSpec("histogram", x=target).to_dict(), key="tgt_hist", toolbar=False, show_table=False)
        else:
            ui.callout(f"'{html.escape(target)}' has {ts.get('unique', 0):,} distinct values — too many to chart as classes.", "na")
    with c2:
        feats = [c for c in frame.columns if c != target and c not in ids and kinds[c] in ("numeric", "categorical")]
        if not feats:
            return ui.callout("No numeric or categorical feature is available to compare with the target.", "na")
        f = st.selectbox("Feature", feats, key="e_feat")
        f_num = kinds[f] == "numeric" and details[f]["unique"] > 12
        if discrete and f_num:
            spec = ch.ChartSpec("box", x=target, y=f, top_n=20)
        elif discrete and len(ts["class_counts"]) == 2:
            spec = ch.ChartSpec("bar", x=f, y=target, agg="rate", top_n=15)
        elif discrete:
            spec = ch.ChartSpec("stacked_bar", x=f, color=target)
        elif f_num and kinds[target] == "numeric":
            spec = ch.ChartSpec("scatter", x=f, y=target)
        elif kinds[target] == "numeric":
            spec = ch.ChartSpec("bar", x=f, y=target, agg="mean", top_n=20)
        else:
            return ui.callout(f"'{html.escape(target)}' has too many distinct values to chart against a feature.", "na")
        ui.spec_chart(spec.to_dict(), key="tgt_rel", source="Target relationships")


{"Recommended": recommended, "Chart builder": builder, "Saved charts": saved, "Univariate": univariate, "Bivariate": bivariate,
 "Correlation": correlation, "Missing & outliers": missing_outliers, "Time series": time_series, "Target": target_view}[view]()
