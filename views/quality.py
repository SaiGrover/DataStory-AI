"""Step 2 — data-quality dimensions with explicit definitions and user-configured rules."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from components import ui
from core import eda, state
from core import quality_metrics as qm

if not ui.require_data("assess its quality"):
    st.stop()

ui.page_header("Step 2 · Data health", "Data quality assessment",
               "Each score below is calculated from the active dataset. Dimensions that need rules stay unevaluated until you define them.")
ui.synthetic_notice()

frame = state.df()
r = state.results()
p, q = r["profile"], r["quality"]
dims, health = q["dimensions"], q["health"]

cols = st.columns(5)
for col, key in zip(cols, ("completeness", "uniqueness", "validity", "consistency")):
    d = dims[key]
    with col:
        if d.get("value") is not None:
            st.metric(key.title(), f"{d['value']:.1f}%", help=f"{d['passed']:,} of {d['applicable']:,}. {d['definition']}")
        else:
            st.metric(key.title(), "—", help=d.get("definition"))
            st.caption(d["message"])
with cols[4]:
    if health.get("value") is not None:
        st.metric("Data health score", f"{health['value']:.0f} / 100", help=health["formula"])
        st.caption("Based on " + ", ".join(health["included"]) + ".")
    else:
        st.metric("Data health score", "—")
        st.caption(health["message"])

with st.expander("How these scores are defined", icon=":material/menu_book:"):
    for k, text in q["definitions"].items():
        st.markdown(f"**{k.title()}** — {text}")
    st.markdown(f"**Data health score** — {qm.HEALTH_FORMULA}")
    excl = st.multiselect("Columns to exclude from completeness", list(frame.columns), default=st.session_state["completeness_exclude"],
                          help="Use for columns that are legitimately optional. Excluded columns are listed in reports.")
    state.set_completeness_exclude(excl)

tab_rules, tab_missing, tab_dups, tab_struct, tab_out, tab_target = st.tabs(
    ["Validation rules", "Missing values", "Duplicates", "Cardinality & structure", "Outliers", "Target distribution"])

# ----------------------------------------------------------------- rules
with tab_rules:
    st.markdown("Validity and consistency are measured only against rules you configure here. Nothing is assumed about what a valid value looks like.")
    rules = list(st.session_state["rules"])
    a, b = st.columns(2, gap="large")
    with a, st.container(border=True):
        st.markdown("**Add a validity rule** (single column)")
        vt = st.selectbox("Rule type", list(qm.VALIDITY_TYPES), format_func=qm.VALIDITY_TYPES.get, key="vt")
        vcol = st.selectbox("Column", list(frame.columns), key="vcol")
        rule = {"dimension": "validity", "type": vt, "column": vcol}
        ok = True
        if vt == "range":
            num = pd.to_numeric(frame[vcol], errors="coerce")
            c1, c2 = st.columns(2)
            use_min = c1.checkbox("Minimum", True, key="v_use_min")
            use_max = c2.checkbox("Maximum", True, key="v_use_max")
            lo = c1.number_input("Min value", value=float(num.min()) if num.notna().any() else 0.0, key=f"vmin_{vcol}", disabled=not use_min)
            hi = c2.number_input("Max value", value=float(num.max()) if num.notna().any() else 1.0, key=f"vmax_{vcol}", disabled=not use_max)
            rule.update(min=lo if use_min else None, max=hi if use_max else None)
            ok = (use_min or use_max) and not (use_min and use_max and lo > hi)
            if not ok:
                st.caption("Set at least one bound, with minimum ≤ maximum.")
        elif vt == "allowed":
            opts = sorted(frame[vcol].dropna().astype(str).unique())[:200]
            vals = st.multiselect("Allowed values", opts, key=f"vallowed_{vcol}")
            extra = st.text_input("Additional allowed values (comma-separated)", key="vextra")
            rule["values"] = vals + [v.strip() for v in extra.split(",") if v.strip()]
            ok = bool(rule["values"])
        if st.button("Add validity rule", disabled=not ok, key="add_v"):
            state.set_rules(rules + [rule])
            st.rerun()
    with b, st.container(border=True):
        st.markdown("**Add a consistency rule** (cross-field or format)")
        ct = st.selectbox("Rule type", list(qm.CONSISTENCY_TYPES), format_func=qm.CONSISTENCY_TYPES.get, key="ct")
        ccol = st.selectbox("Column", list(frame.columns), key="ccol")
        rule = {"dimension": "consistency", "type": ct, "column": ccol}
        ok = True
        if ct == "compare":
            c1, c2 = st.columns([1, 2])
            rule["op"] = c1.selectbox("Operator", ["<=", "<", ">=", ">", "==", "!="], key="cop")
            others = [c for c in frame.columns if c != ccol]
            rule["other"] = c2.selectbox("Other column", others, key="cother") if others else None
            ok = rule["other"] is not None
        elif ct == "format":
            rule["pattern"] = st.text_input("Regular expression (must match the whole value)", placeholder=r"\d{4}-\d{2}-\d{2}", key="cpat")
            ok = bool(rule["pattern"])
        else:
            st.caption("Passes when a value has only one spelling regardless of case — e.g. 'Male' and 'male' both present fails.")
        if st.button("Add consistency rule", disabled=not ok, key="add_c"):
            state.set_rules(rules + [rule])
            st.rerun()

    results_rows = [x for k in ("validity", "consistency") for x in dims[k].get("rules", [])]
    if results_rows:
        st.markdown("**Configured rules**")
        st.dataframe(pd.DataFrame([{
            "Dimension": x["dimension"].title(), "Rule": x["rule"],
            "Passed": x.get("passed"), "Applicable": x.get("applicable"), "Pass rate %": x.get("value"),
            "Status": x.get("error", "Evaluated"),
        } for x in results_rows]), width="stretch", hide_index=True,
            column_config={"Pass rate %": st.column_config.NumberColumn(format="%.2f%%")})
        c1, c2 = st.columns([3, 1])
        idx = c1.selectbox("Remove a rule", range(len(rules)), format_func=lambda i: f"{rules[i]['dimension'].title()}: {qm.describe_rule(rules[i])}", key="rm_rule")
        c2.write("")
        if c2.button("Remove", key="rm_btn", width="stretch"):
            state.set_rules([x for i, x in enumerate(rules) if i != idx])
            st.rerun()
    else:
        ui.callout(qm.NOT_EVALUATED + " Add a rule above to evaluate validity or consistency.", "na")

# --------------------------------------------------------------- missing
with tab_missing:
    fig = eda.missing_bar(p)
    if fig is None:
        st.success("No missing values in any column.", icon=":material/check_circle:")
    else:
        a, b = st.columns(2)
        with a:
            ui.chart(fig, "q_missing_bar")
        with b:
            ui.chart(eda.missing_matrix(state.analysis_df()), "q_missing_matrix")
        ui.interpretation(eda.interpret_missing(p))
        st.dataframe(pd.DataFrame([{"Column": c["name"], "Missing": c["missing"], "Missing %": c["missing_pct"]}
                                   for c in p["column_details"] if c["missing"]]).sort_values("Missing", ascending=False),
                     width="stretch", hide_index=True, column_config={"Missing %": st.column_config.NumberColumn(format="%.2f%%")})

# ------------------------------------------------------------ duplicates
with tab_dups:
    if p["duplicate_rows"] == 0:
        st.success("No exact duplicate rows.", icon=":material/check_circle:")
    else:
        st.markdown(f"**{p['duplicate_rows']:,}** rows ({p['duplicate_pct']:.2f}%) are exact copies of an earlier row.")
        st.dataframe(frame[frame.duplicated(keep=False)].sort_values(list(frame.columns)[:3]).head(200), width="stretch")
        st.caption("All copies of duplicated rows are shown (up to 200). Remove them on the Data Cleaning step if they are not genuine repeats.")

# ------------------------------------------------------------- structure
with tab_struct:
    c1, c2, c3 = st.columns(3)
    c1.metric("Constant columns", len(p["constant_columns"]), help="A single distinct value (or entirely missing).")
    c2.metric("Near-constant columns", len(p["near_constant_columns"]), help="One value covers at least 95% of non-missing rows.")
    c3.metric("Identifier-like columns", len(p["id_candidates"]), help="Unique for every row and integer-sequential, ID-named, or short strings.")
    for label, key in (("Constant", "constant_columns"), ("Near-constant", "near_constant_columns"), ("Identifier-like", "id_candidates")):
        if p[key]:
            st.markdown(f"**{label}:** " + ", ".join(f"`{c}`" for c in p[key]))
    st.markdown("**Cardinality**")
    st.dataframe(pd.DataFrame([{"Column": c["name"], "Type": c["kind"], "Distinct values": c["unique"], "Distinct % of non-missing": c["unique_pct"]}
                               for c in p["column_details"]]).sort_values("Distinct values", ascending=False),
                 width="stretch", hide_index=True, column_config={"Distinct % of non-missing": st.column_config.NumberColumn(format="%.1f%%")})
    st.markdown("**Storage data types:** " + ", ".join(f"{k} × {v}" for k, v in p["dtype_counts"].items()))

# -------------------------------------------------------------- outliers
with tab_out:
    st.caption(p["outlier_method"])
    if not p["outlier_columns"]:
        st.success("No numeric column has values outside the Tukey fences.", icon=":material/check_circle:")
    else:
        st.dataframe(pd.DataFrame(p["outlier_columns"]).rename(columns={"column": "Column", "outliers": "Outliers", "outlier_pct": "% of non-missing"}),
                     width="stretch", hide_index=True, column_config={"% of non-missing": st.column_config.NumberColumn(format="%.2f%%")})
        pick = st.selectbox("Inspect a column", [x["column"] for x in p["outlier_columns"]], key="q_out_col")
        ui.chart(eda.box_or_violin(state.analysis_df(), pick), "q_out_box")
        ui.callout("A flagged value is statistically unusual, not necessarily wrong. Skewed columns routinely have many flagged values.", "read")

# ---------------------------------------------------------------- target
with tab_target:
    t = ui.target_picker("Target column", key="q_target", help="Optional. Used for class-distribution analysis and modelling.")
    if not t:
        ui.callout("Select a target column to see its distribution.", "na")
    else:
        ts = state.results()["profile"]["target"]
        if ts.get("class_counts"):
            ui.chart(eda.class_balance(ts["class_counts"], t), "q_balance")
        else:
            ui.chart(eda.histogram(state.analysis_df(), t), "q_target_hist")
        ui.interpretation(eda.interpret_balance(ts, t))
        if ts["missing"]:
            st.warning(f"{ts['missing']:,} rows have no target value; they are excluded from model training.", icon=":material/warning:")
