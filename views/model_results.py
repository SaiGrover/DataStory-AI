"""Step 7 — leaderboard, per-model evaluation and held-out permutation importance."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components import ui
from core import eda, modeling, state
from core import explainability as ex

if not ui.require_data("see model results"):
    st.stop()

ui.page_header("Step 7 · Results", "Model results & explainability",
               "Scores are measured on the held-out test set. Feature importance is computed from the fitted model, never assumed.")
ui.synthetic_notice()

tr = state.training()
if tr is None:
    ui.empty_state("No trained models", "Train models on the Modeling step to see evaluation metrics and feature importance.")
    st.page_link("views/ml.py", label="Go to Modeling", icon=":material/science:")
    st.stop()

res = tr.results
ok = [m for m in res["models"] if "error" not in m]
L = modeling.METRIC_LABELS
primary = res["primary_metric"]

st.subheader("Leaderboard")
board = modeling.leaderboard(res)
st.dataframe(board, width="stretch", hide_index=True,
             column_config={c: st.column_config.NumberColumn(format="%.4f") for c in board.columns if c.startswith("Test ")})
st.caption(f"**Best model: {res['best_model']}** — ranked by {res['ranking_basis']}. "
           f"{res['n_train']:,} training rows, {res['n_test']:,} test rows, seed {res['seed']}.")
for w in res["warnings"]:
    st.warning(w, icon=":material/warning:")

names = [m["name"] for m in ok]
compare_metric = st.selectbox("Compare models on", res["metric_names"], index=res["metric_names"].index(primary), format_func=L.get, key="mr_metric")
vals = [m["metrics"].get(compare_metric) for m in ok]
if all(v is None for v in vals):
    ui.callout(f"{L[compare_metric]} is unavailable for these models.", "na")
else:
    fig = go.Figure(go.Bar(x=names, y=vals, marker_color=[eda.SAGE if n == res["best_model"] else "#C9D6B8" for n in names],
                           text=[("n/a" if v is None else f"{v:.3f}") for v in vals], textposition="outside"))
    fig.update_layout(yaxis_title=f"Test {L[compare_metric]}" + (" (lower is better)" if compare_metric in modeling.LOWER_IS_BETTER else ""))
    ui.chart(eda.style(fig, f"Test {L[compare_metric]} by model", height=340), "mr_compare")

st.subheader("Model detail")
pick = st.selectbox("Model", names, key="mr_model")
m = next(x for x in ok if x["name"] == pick)
cols = st.columns(len(res["metric_names"]))
for col, k in zip(cols, res["metric_names"]):
    v = m["metrics"].get(k)
    col.metric(L[k], "Unavailable" if v is None else f"{v:.4f}")
notes = list(m["notes"])
if m.get("roc_auc_note"):
    notes.append(m["roc_auc_note"])
if m.get("best_params"):
    notes.append("Tuned hyperparameters: " + ", ".join(f"{k} = {v}" for k, v in m["best_params"].items())
                 + f" (best of {m['cv'].get('candidates', '?')} candidates by cross-validated {L[m['cv']['metric']]}).")
if m["cv"]:
    notes.append(f"{m['cv']['folds']}-fold cross-validated {L[m['cv']['metric']]} on training data: {m['cv']['mean']:.4f} ± {m['cv']['std']:.4f}.")
for n in notes:
    st.caption(n)

if res["task"] == "classification":
    classes = res["classes"]
    st.caption(f"Precision, recall and F1 use {m['averaging']} averaging.")
    a, b = st.columns(2, gap="large")
    with a:
        cm = np.array(m["confusion_matrix"])
        fig = go.Figure(go.Heatmap(z=cm, x=classes, y=classes, colorscale=eda.SEQUENTIAL, showscale=False,
                                   text=cm, texttemplate="%{text}" if len(classes) <= 15 else None,
                                   hovertemplate="actual %{y}, predicted %{x}: %{z}<extra></extra>"))
        fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
        fig.update_xaxes(type="category")
        fig.update_yaxes(type="category", autorange="reversed")
        ui.chart(eda.style(fig, "Confusion matrix (test set)", height=380), "mr_cm")
    with b:
        if m.get("roc_curve"):
            rc = m["roc_curve"]
            fig = go.Figure([go.Scatter(x=rc["fpr"], y=rc["tpr"], mode="lines", name=pick, line=dict(color=eda.SAGE, width=2.5)),
                             go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(color="#B9C2B0", dash="dash"))])
            fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate")
            ui.chart(eda.style(fig, f"ROC curve (AUC = {m['metrics']['roc_auc']:.3f})", height=380), "mr_roc")
        else:
            st.markdown("**Class-specific performance**")
        rep = pd.DataFrame(m["class_report"]).T.rename(columns={"precision": "Precision", "recall": "Recall", "f1-score": "F1-score", "support": "Support"})
        rep["Support"] = rep["Support"].astype(int)
        st.dataframe(rep, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.4f") for c in ("Precision", "Recall", "F1-score")})
    worst = min(m["class_report"].items(), key=lambda kv: kv[1]["recall"])
    ui.interpretation(f"On {res['n_test']:,} test rows, **{pick}** classifies {int(np.trace(cm)):,} correctly. The class with the lowest recall is "
                      f"'{worst[0]}' ({worst[1]['recall']:.4f} over {int(worst[1]['support'])} test rows).")
else:
    pr = m["predictions"]
    actual, pred = np.array(pr["actual"]), np.array(pr["predicted"])
    a, b = st.columns(2, gap="large")
    with a:
        lo, hi = float(min(actual.min(), pred.min())), float(max(actual.max(), pred.max()))
        fig = go.Figure([go.Scatter(x=actual, y=pred, mode="markers", marker=dict(color=eda.SAGE, opacity=0.6), name="Test rows"),
                         go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="#B9C2B0", dash="dash"), name="Perfect prediction")])
        fig.update_layout(xaxis_title=f"Actual {res['target']}", yaxis_title="Predicted")
        ui.chart(eda.style(fig, "Predicted vs actual (test set)", height=380), "mr_pva")
    with b:
        fig = go.Figure(go.Histogram(x=pred - actual, nbinsx=40, marker_color=eda.SAGE))
        fig.update_layout(xaxis_title="Residual (predicted − actual)", yaxis_title="Test rows")
        ui.chart(eda.style(fig, "Residual distribution", height=380), "mr_resid")
    r2 = m["metrics"].get("r2")
    ui.interpretation(f"On {res['n_test']:,} test rows, **{pick}** is off by {m['metrics']['mae']:.4g} on average (MAE) with RMSE {m['metrics']['rmse']:.4g}. "
                      + ("R² is unavailable." if r2 is None else f"R² = {r2:.4f}" + (", i.e. no better than predicting the mean." if r2 <= 0 else ".")))

# ------------------------------------------------------- explainability
st.subheader("Feature importance (held-out permutation importance)")
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
