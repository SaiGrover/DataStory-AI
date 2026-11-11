"""Step 7 — leaderboard, held-out diagnostics, cross-validation and permutation importance."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components import ui
from core import diagnostics as dg
from core import eda, modeling, state
from core import explainability as ex
from core.formatting import count, metric, pct

if not ui.require_data("see model results"):
    st.stop()

ui.page_header("Step 7 · Results", "Model results & diagnostics",
               "Test scores come from the held-out test set; cross-validation scores come from the training data. "
               "Feature importance is computed from the fitted model, never assumed.")
ui.synthetic_notice()

tr = state.training()
if tr is None:
    ui.empty_state("No trained models", "Train models on the Modeling step to see evaluation metrics and diagnostics.")
    st.page_link("views/ml.py", label="Go to Modeling", icon=":material/science:")
    st.stop()

res = tr.results
ok = [m for m in res["models"] if "error" not in m]
L = modeling.METRIC_LABELS
primary = res["primary_metric"]
names = [m["name"] for m in ok]

# small-test-set limitations
limits = []
if res["n_test"] < 30:
    limits.append(f"The test set has only {count(res['n_test'])} rows, so every test metric below is highly uncertain.")
if res["task"] == "classification":
    d0 = ok[0].get("diagnostics") or {}
    small = [f"'{k}' ({v})" for k, v in (d0.get("test_class_counts") or {}).items() if v < 5]
    if small:
        limits.append("Some classes have fewer than 5 test rows — " + ", ".join(small[:6]) + " — so per-class metrics for them are unreliable.")
for msg in limits:
    st.warning(msg, icon=":material/science:")

tab_names = ["Leaderboard", "Classification diagnostics" if res["task"] == "classification" else "Regression diagnostics",
             "Cross-validation", "Feature importance"]
t_lb, t_diag, t_cv, t_imp = st.tabs(tab_names)

# ----------------------------------------------------------- leaderboard
with t_lb:
    board = modeling.leaderboard(res)
    st.dataframe(board, width="stretch", hide_index=True,
                 column_config={c: st.column_config.NumberColumn(format="%.4f") for c in board.columns if c.startswith("Test ")})
    st.caption(f"**Best model: {res['best_model']}** — ranked by {res['ranking_basis']}. "
               f"“Test” columns: held-out test set ({count(res['n_test'])} rows). “CV” column: training data only. "
               f"{count(res['n_train'])} training rows, seed {res['seed']}.")
    for w in res["warnings"]:
        st.warning(w, icon=":material/warning:")
    compare_metric = st.selectbox("Compare models on", res["metric_names"], index=res["metric_names"].index(primary),
                                  format_func=L.get, key="mr_metric")
    vals = [m["metrics"].get(compare_metric) for m in ok]
    if all(v is None for v in vals):
        ui.callout(f"{L[compare_metric]} is unavailable for these models.", "na")
    else:
        fig = go.Figure(go.Bar(x=names, y=vals, marker_color=[eda.SAGE if n == res["best_model"] else "#C9D6B8" for n in names],
                               text=[("n/a" if v is None else f"{v:.3f}") for v in vals], textposition="outside"))
        fig.update_layout(yaxis_title=f"Test {L[compare_metric]}" + (" (lower is better)" if compare_metric in modeling.LOWER_IS_BETTER else ""))
        ui.chart(eda.style(fig, f"Test {L[compare_metric]} by model (held-out)", height=340), "mr_compare")
        valid = [(n, v) for n, v in zip(names, vals) if v is not None]
        lo_better = compare_metric in modeling.LOWER_IS_BETTER
        top = (min if lo_better else max)(valid, key=lambda kv: kv[1])
        spread = max(v for _, v in valid) - min(v for _, v in valid)
        ui.interpretation(f"On the held-out test set, **{top[0]}** has the {'lowest' if lo_better else 'highest'} {L[compare_metric]} "
                          f"({metric(top[1])}); the spread across {len(valid)} model(s) is {metric(spread)}. "
                          "Small differences on one test split are not evidence that one model is generally better.")

pick = None
with t_diag:
    pick = st.selectbox("Model", names, key="mr_model")
    m = next(x for x in ok if x["name"] == pick)
    d = m.get("diagnostics") or {}
    cols = st.columns(len(res["metric_names"]))
    for col, k in zip(cols, res["metric_names"]):
        v = m["metrics"].get(k)
        col.metric(f"Test {L[k]}", "Unavailable" if v is None else f"{v:.4f}", help="Held-out test set.")
    notes = list(m["notes"])
    if m.get("roc_auc_note"):
        notes.append(m["roc_auc_note"])
    if m.get("best_params"):
        notes.append("Tuned hyperparameters: " + ", ".join(f"{k} = {v}" for k, v in m["best_params"].items())
                     + f" (best of {m['cv'].get('candidates', '?')} candidates by cross-validated {L[m['cv']['metric']]}).")
    for n in notes:
        st.caption(n)
    if d.get("error"):
        st.warning(d["error"], icon=":material/warning:")

    if res["task"] == "classification":
        classes = res["classes"]
        st.caption(f"Precision, recall and F1 use {m['averaging']} averaging.")
        a, b = st.columns(2, gap="large")
        with a:
            normalized = st.toggle("Normalise by actual class", key="mr_cm_norm")
            cm = np.array(m["confusion_matrix"])
            fig = dg.fig_confusion(d.get("confusion_matrix_normalized") if normalized else m["confusion_matrix"], classes, normalized)
            if fig:
                ui.chart(fig, "mr_cm")
            errs = d.get("errors") or {}
            txt = f"Of {count(res['n_test'])} test rows, **{pick}** classifies {count(int(np.trace(cm)))} correctly"
            if errs.get("top_confusions"):
                tc = errs["top_confusions"][0]
                txt += f"; the most common mistake is predicting '{tc['predicted']}' for actual '{tc['actual']}' ({count(tc['count'])} rows)"
            ui.interpretation(txt + ".")
        with b:
            fig = dg.fig_roc(m.get("roc_curve"), m["metrics"].get("roc_auc"), pick)
            if fig:
                ui.chart(fig, "mr_roc")
                ui.interpretation(f"ROC-AUC {metric(m['metrics']['roc_auc'], 3)}: the probability that a random positive test row is "
                                  "scored above a random negative one (0.5 = chance).")
            elif len(classes) > 2:
                ui.callout("ROC and PR curves are drawn for binary targets. For this multiclass target, one-vs-rest PR-AUC per class is shown below.", "na")
        a, b = st.columns(2, gap="large")
        with a:
            fig = dg.fig_pr(d.get("pr_curve"), pick)
            if fig:
                ui.chart(fig, "mr_pr")
                pr = d["pr_curve"]
                ui.interpretation(f"Average precision {metric(pr['average_precision'], 3)} for class '{pr['positive_class']}', against a "
                                  f"baseline of {metric(pr['baseline'], 3)} (the share of positives in the test set).")
            elif d.get("pr_auc_ovr"):
                st.markdown("**One-vs-rest PR-AUC per class (test set)**")
                st.dataframe(pd.DataFrame([{"Class": k, "PR-AUC": v} for k, v in d["pr_auc_ovr"].items()]), hide_index=True, width="stretch",
                             column_config={"PR-AUC": st.column_config.NumberColumn(format="%.4f")})
            elif d.get("pr_note") or d.get("probability_note"):
                ui.callout(d.get("pr_note") or d.get("probability_note"), "na")
        with b:
            fig = dg.fig_calibration(d.get("calibration"))
            if fig:
                ui.chart(fig, "mr_cal")
                ui.interpretation(f"Points near the diagonal mean predicted probabilities match observed frequencies. Brier score "
                                  f"{metric(d['calibration']['brier'], 4)} (lower is better). With {count(res['n_test'])} test rows, "
                                  "each bin holds few rows, so treat the curve as indicative.")
            else:
                fig = dg.fig_confidence(d.get("confidence_hist"))
                if fig:
                    ui.chart(fig, "mr_conf")
                    ui.interpretation("Calibration curves are shown for binary targets. This chart compares the model's confidence on "
                                      "correct and incorrect test predictions.")
        fig = dg.fig_probability(d.get("probability_hist"))
        if fig:
            ui.chart(fig, "mr_proba")
            ui.interpretation("Good separation shows as the two classes' probabilities piling up at opposite ends.")
        st.markdown("**Per-class performance (test set)**")
        rep = pd.DataFrame(m["class_report"]).T.rename(columns={"precision": "Precision", "recall": "Recall", "f1-score": "F1-score", "support": "Support"})
        rep["Support"] = rep["Support"].astype(int)
        st.dataframe(rep, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.4f") for c in ("Precision", "Recall", "F1-score")})
        worst = min(m["class_report"].items(), key=lambda kv: kv[1]["recall"])
        errs = d.get("errors") or {}
        ui.interpretation(f"The class with the lowest recall is '{worst[0]}' ({metric(worst[1]['recall'])} over {int(worst[1]['support'])} test rows). "
                          + (f"Overall error rate {pct(100 * errs['error_rate'])} ({count(errs['n_misclassified'])} rows)." if errs.get("error_rate") is not None else ""))
        if errs.get("top_confusions"):
            with st.expander("Most frequent misclassifications"):
                st.dataframe(pd.DataFrame(errs["top_confusions"]).rename(columns={"actual": "Actual", "predicted": "Predicted", "count": "Test rows"}),
                             hide_index=True, width="stretch")
    else:
        a, b = st.columns(2, gap="large")
        with a:
            fig = dg.fig_actual_vs_predicted(m.get("predictions"), res["target"])
            if fig:
                ui.chart(fig, "mr_pva")
            r2 = m["metrics"].get("r2")
            ui.interpretation(f"On {count(res['n_test'])} test rows, **{pick}** is off by {metric(m['metrics']['mae'])} on average (MAE) with RMSE "
                              f"{metric(m['metrics']['rmse'])}. " + ("R² is unavailable." if r2 is None else f"R² = {metric(r2)}"
                              + (", i.e. no better than predicting the mean." if r2 <= 0 else ".")))
        with b:
            fig = dg.fig_residual_hist(m.get("predictions"))
            if fig:
                ui.chart(fig, "mr_resid")
            if d.get("mean_residual") is not None:
                ui.interpretation(f"Mean residual {metric(d['mean_residual'])}: "
                                  + ("predictions are on average higher than actual values." if d["mean_residual"] > 0 else
                                     "predictions are on average lower than actual values." if d["mean_residual"] < 0 else "no average bias."))
        a, b = st.columns(2, gap="large")
        with a:
            fig = dg.fig_residuals_vs_predicted(d)
            if fig:
                ui.chart(fig, "mr_rvp")
                ui.interpretation("A funnel or curve in this cloud means errors depend on the predicted value (heteroscedasticity or a missed non-linearity).")
        with b:
            fig = dg.fig_abs_error(d)
            if fig:
                ui.chart(fig, "mr_abs")
                q = d["abs_error_quantiles"]
                ui.interpretation(f"Half of test predictions are within {metric(q['0.5'])} of the actual value; 90% are within {metric(q['0.9'])}.")
        groups = d.get("error_by_subgroup") or []
        if groups:
            g = st.selectbox("Error by subgroup", [x["column"] for x in groups], key="mr_group")
            grp = next(x for x in groups if x["column"] == g)
            ui.chart(dg.fig_error_by_group(grp), "mr_grp")
            hi, lo = grp["groups"][0], grp["groups"][-1]
            ui.interpretation(f"MAE is highest for {g} = '{hi['value']}' ({metric(hi['mae'])}, n = {hi['n']}) and lowest for "
                              f"'{lo['value']}' ({metric(lo['mae'])}, n = {lo['n']}). Groups with fewer than {dg.MIN_SUBGROUP} test rows are omitted.")
        else:
            ui.callout("Error by subgroup needs a categorical feature with at least two groups of 10+ test rows.", "na")

# ------------------------------------------------------- cross-validation
with t_cv:
    ui.callout("<strong>Training data, not held-out.</strong> These scores come from cross-validation folds of the training set. "
               "They are used to choose and compare models; the held-out test scores are on the Leaderboard tab.", "na")
    fig = dg.fig_cv_scores(ok, L[primary])
    if fig is None:
        ui.callout("Cross-validation was off for this run, so there are no fold scores. Enable it under Advanced settings on the Modeling step.", "na")
    else:
        ui.chart(fig, "mr_cv")
        rows = [{"Model": x["name"], "Folds": x["cv"]["folds"], "Mean": x["cv"]["mean"], "Std": x["cv"]["std"],
                 "Min": min(x["cv"]["scores"]), "Max": max(x["cv"]["scores"])} for x in ok if x.get("cv") and x["cv"].get("scores")]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.4f") for c in ("Mean", "Std", "Min", "Max")})
        widest = max(rows, key=lambda r: r["Max"] - r["Min"])
        ui.interpretation(f"{L[primary]} varies across folds; **{widest['Model']}** has the widest range "
                          f"({metric(widest['Min'])} to {metric(widest['Max'])}). Overlapping ranges mean the ranking may change with different data.")

# ------------------------------------------------------- explainability
with t_imp:
    st.markdown("**Held-out permutation importance**")
    c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="bottom")
    imp_model = c1.selectbox("Model to explain", names, index=names.index(res["best_model"]), key="mr_imp_model")
    repeats = c2.slider("Permutation repeats", 3, 30, 10, key="mr_repeats", help="More repeats give a steadier mean and a better estimate of spread.")
    if c3.button("Compute importance", type="primary", icon=":material/insights:", width="stretch"):
        try:
            with st.spinner(f"Shuffling each feature {repeats}× on the test set…"):
                imp = ex.held_out_permutation_importance(tr.pipelines.get(imp_model), tr.X_test, tr.y_test, res["task"], primary,
                                                         len(res["classes"]) or 2, repeats, res["seed"], imp_model)
            state.set_importance(imp)
            st.rerun()
        except ex.ImportanceUnavailable as exc:
            state.set_importance(None)
            st.error(str(exc), icon=":material/error:")

    imp = state.importance()
    if not imp:
        ui.callout(ex.UNAVAILABLE + " Choose “Compute importance” to calculate it from the trained model and the test set.", "na")
    else:
        feats = imp["features"]
        show = feats[:st.slider("Features to show", 3, max(3, min(40, len(feats))), min(15, len(feats)), key="mr_topn")] if len(feats) > 3 else feats
        t = pd.DataFrame(show).iloc[::-1]
        fig = go.Figure(go.Bar(x=t["mean"], y=t["feature"], orientation="h", error_x=dict(type="data", array=t["std"], color="#39463A", thickness=1.2),
                               marker_color=[eda.SAGE if v >= 0 else "#A65D57" for v in t["mean"]],
                               hovertemplate="%{y}: %{x:.4f} ± %{error_x.array:.4f}<extra></extra>"))
        fig.update_layout(xaxis_title=f"Mean change in test {imp['metric_label']} when shuffled (± 1 sd)")
        fig.update_yaxes(type="category")
        ui.chart(eda.style(fig, f"Permutation importance — {imp['model']}", height=max(300, 26 * len(t) + 110)), "mr_imp")
        st.caption(f"Model: {imp['model']} · Evaluated on: {imp['evaluated_on']} · Scoring: {imp['metric_label']} (`{imp['scoring']}`) · "
                   f"Repeats: {imp['n_repeats']} · Seed: {imp['seed']}")
        ui.callout(f"<strong>How to read this.</strong> {imp['explanation']}", "read")
        for n in imp["notes"]:
            st.warning(n, icon=":material/warning:")
        with st.expander("Importance table"):
            st.dataframe(pd.DataFrame(feats).rename(columns={"feature": "Feature", "mean": "Mean importance", "std": "Std"}), width="stretch", hide_index=True,
                         column_config={c: st.column_config.NumberColumn(format="%.5f") for c in ("Mean importance", "Std")})
    ui.callout("SHAP explanations are not included in this version; permutation importance is the supported explainability method.", "na")
