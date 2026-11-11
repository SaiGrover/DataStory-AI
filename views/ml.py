"""Step 6 — target suitability, readiness checks, then training. Nothing trains until the button is pressed."""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from components import ui
from core import ml_readiness, modeling, state
from core import targets as tg
from core.formatting import pct

if not ui.require_data("train models"):
    st.stop()

ui.page_header("Step 6 · Modeling", "Machine learning",
               "DataStory first checks whether a meaningful prediction problem exists, then trains leakage-safe pipelines "
               "only when you ask.")
ui.synthetic_notice()

full = state.df()
frame = state.analysis_df()
p = state.results()["profile"]
kinds = p["kinds"]
if len(frame) < len(full):
    st.info(f"Training uses a seeded random sample of {len(frame):,} of {len(full):,} rows.", icon=":material/info:")

# ------------------------------------------------------ target suitability
key = (id(full), full.shape, tuple(full.columns))
cached = st.session_state.get("_target_assessment")
if not cached or cached[0] != key:
    cached = (key, tg.assess_targets(full, kinds))
    st.session_state["_target_assessment"] = cached
    # default the target only when a column is genuinely recommended, and only once per dataset version
    if state.target() is None and cached[1]["best"]:
        state.set_target(cached[1]["best"])
assessment = cached[1]
by_col = {c["column"]: c for c in assessment["candidates"]}
E = html.escape
VERDICT = {"recommended": ("Recommended", "ok"), "possible": ("Possible", "muted"),
           "discouraged": ("Discouraged", "syn"), "blocked": ("Blocked", "syn")}

if not assessment["supervised_viable"]:
    ui.callout(f"<strong>No suitable prediction target.</strong> {E(assessment['recommendation'])}", "na")
    c1, c2 = st.columns(2)
    c1.page_link("views/eda.py", label="Explore the data (EDA)", icon=":material/show_chart:")
    c2.page_link("views/clustering.py", label="Find segments with clustering", icon=":material/bubble_chart:")
elif not assessment["best"]:
    ui.callout(f"<strong>Target guidance.</strong> {E(assessment['recommendation'])}", "na")
    c1, c2 = st.columns(2)
    c1.page_link("views/eda.py", label="Explore the data (EDA)", icon=":material/show_chart:")
    c2.page_link("views/clustering.py", label="Find segments with clustering", icon=":material/bubble_chart:")
else:
    ui.callout(f"<strong>Target guidance.</strong> {E(assessment['recommendation'])}", "read")

with st.expander("How every column was assessed as a target", icon=":material/rule:"):
    st.dataframe(pd.DataFrame([{
        "Column": c["column"], "Verdict": c["verdict"].title(), "Task": c["task"] or "—",
        "Score": c["score"], "Distinct": c["n_unique"], "Missing": pct(c["missing_pct"]),
        "Why": "; ".join(c["blocking"] or c["reasons"]), "Warnings": "; ".join(c["warnings"]),
    } for c in assessment["candidates"]]), width="stretch", hide_index=True,
        column_config={"Score": st.column_config.NumberColumn(format="%.2f")})
    st.caption(assessment["score_formula"])

left, right = st.columns([1, 1], gap="large")
with left, st.container(border=True):
    st.subheader("Target & features")
    options = [ui.NONE] + list(full.columns)
    cur = state.target()

    def _label(c: str) -> str:
        if c == ui.NONE:
            return c
        v = by_col.get(c)
        return f"{c}  ·  {v['verdict']}" if v else c

    choice = st.selectbox("Target column", options, index=options.index(cur) if cur in options else 0, key="ml_target",
                          format_func=_label, help="The column the models learn to predict. Identifiers and free text are blocked.")
    state.set_target(None if choice == ui.NONE else choice)
    target = state.target()
    if not target:
        ui.callout("Select a target column to continue.", "na")
        st.stop()
    cand = by_col.get(target, {})
    label, cls = VERDICT.get(cand.get("verdict", "possible"), ("", "muted"))
    st.html(f'<span class="ds-pill {cls}">{label}</span> <span style="font-size:.85rem">'
            + E("; ".join(cand.get("blocking") or cand.get("reasons") or [])) + "</span>")

    auto_task, reason = modeling.detect_task(frame[target])
    tchoice = st.radio("Task type", ["Auto", "Classification", "Regression"], horizontal=True, key="ml_task",
                       help="Override when the automatic detection is wrong, e.g. numeric class codes.")
    task = auto_task if tchoice == "Auto" else tchoice.lower()
    st.caption(f"Detected **{auto_task}** because {reason}." + ("" if tchoice == "Auto" else f" Overridden to **{task}**."))

    validity = tg.validate_target(full, target, task, kinds)
    for b in validity["blocking"]:
        st.error(f"'{target}' cannot be used as a {task} target: {b}.", icon=":material/block:")
    for w in validity["warnings"]:
        st.warning(w[0].upper() + w[1:] + ".", icon=":material/warning:")
    override_ok = True
    if validity["overridable"]:
        override_ok = st.checkbox("I understand this target is discouraged and want to train anyway", key=f"ml_override_{target}_{task}")

    candidates = [c for c in frame.columns if c != target]
    pre = ml_readiness.assess_readiness(frame, target, candidates, task, kinds) if validity["ok"] else None
    skip = set(p["id_candidates"]) | set(p["constant_columns"])
    if pre:
        skip |= {e["column"] for e in pre["recommendations"]["exclude"]}
    default = [c for c in candidates if c not in skip]
    features = st.multiselect("Features", candidates, default=default, key=f"ml_feats_{target}",
                              help="Identifier-like, constant, free-text, very sparse and possibly leaking columns are left out by default.")
    if skip & set(candidates):
        st.caption("Left out by default: " + ", ".join(sorted(skip & set(candidates))) + ".")

rec = pre["recommendations"] if pre else {}
with right, st.container(border=True):
    st.subheader("Models & evaluation")
    names = modeling.available_models(task)
    heavy = len(frame) * max(len(features), 1) > 2_000_000
    default_models = [n for n in names if not (heavy and "Gradient Boosting" in n)]
    models = st.multiselect("Models", names, default=default_models, key=f"ml_models_{task}")
    metrics = modeling.CLASSIFICATION_METRICS if task == "classification" else ["r2", "rmse", "mae", "mse"]
    rec_metric = rec.get("metric") if rec.get("metric") in metrics else ("f1" if task == "classification" else "r2")
    c1, c2 = st.columns(2)
    primary = c1.selectbox("Rank models by", metrics, index=metrics.index(rec_metric), format_func=modeling.METRIC_LABELS.get,
                           key=f"ml_metric_{task}", help=rec.get("metric_reason") or None)
    test_size = c2.slider("Test set share", 0.1, 0.4, 0.2, 0.05, key="ml_test")
    with st.expander("Advanced settings", icon=":material/tune:"):
        a1, a2 = st.columns(2)
        cv = a1.selectbox("Cross-validation", [0, 3, 5, 10], index=2, format_func=lambda k: "Off" if not k else f"{k}-fold", key="ml_cv",
                          help="Run on the training data. When on, the best model is chosen by its cross-validated score, not the test set.")
        seed = int(a2.number_input("Random seed", 0, 10_000, 42, key="ml_seed"))
        a1, a2 = st.columns(2)
        tuning = a1.selectbox("Hyperparameter tuning", ["none", "random", "grid"], key="ml_tune",
                              format_func={"none": "Off", "random": "Randomized search", "grid": "Grid search (slower)"}.get)
        imbalance = "none"
        if task == "classification":
            imbalance = a2.selectbox("Class imbalance", ["none", "class_weight", "smote"], key="ml_imb",
                                     format_func={"none": "No handling", "class_weight": "Class weights", "smote": "SMOTE (training folds only)"}.get,
                                     help=rec.get("imbalance_reason") or None)
        scale = st.checkbox("Standardise numeric features", True, key="ml_scale",
                            help="Fitted on the training data inside the pipeline. Helps linear and distance-based models; harmless for trees.")

# ------------------------------------------------------------ readiness
ready_check = ml_readiness.assess_readiness(frame, target, features, task, kinds, test_size, cv)
status = ready_check["status"]
st.subheader("Readiness assessment")
pill = {"ready": ("Ready to train", "ok"), "caution": ("Ready with cautions", "syn"), "blocked": ("Blocked", "syn")}[status]
st.html(f'<span class="ds-pill {pill[1]}">{pill[0]}</span> <span style="font-size:.85rem">'
        "Specific checks, not a score: training is blocked only by a blocker.</span>")
sev_icon = {"blocker": ":material/block:", "warning": ":material/warning:", "info": ":material/info:"}
a, b = st.columns([1.3, 1], gap="large")
with a:
    for f in sorted(ready_check["findings"], key=lambda f: ["blocker", "warning", "info"].index(f["severity"])):
        st.markdown(f"{sev_icon[f['severity']]} **{f['title']}** — {f['detail']}")
    if not ready_check["findings"]:
        st.markdown(":material/check_circle: No issues found.")
with b:
    r_ = ready_check["recommendations"]
    lines = []
    if r_.get("metric"):
        lines.append(f"**Metric:** {modeling.METRIC_LABELS.get(r_['metric'], r_['metric'])} — {r_['metric_reason']}")
    if r_.get("models"):
        lines.append("**Baselines:** " + ", ".join(r_["models"]) + ".")
    if r_.get("cv_reason"):
        lines.append(f"**Validation:** {r_['cv']}-fold CV — {r_['cv_reason']}")
    if task == "classification" and r_.get("imbalance_reason"):
        lines.append(f"**Imbalance:** {r_['imbalance'].replace('_', ' ')} — {r_['imbalance_reason']}")
    if r_.get("exclude"):
        lines.append("**Consider excluding:** " + "; ".join(f"{e['column']} ({e['reason']})" for e in r_["exclude"][:8]) + ".")
    st.markdown("\n\n".join(lines) or "Recommendations appear once a valid target is selected.")

config = {"target": target, "task": task, "features": sorted(features), "models": models, "primary": primary, "test_size": test_size,
          "cv": cv, "seed": seed, "tuning": tuning, "imbalance": imbalance, "scale": scale}

ui.callout("<strong>How leakage is prevented.</strong> The data is split first. Imputation, scaling, encoding and SMOTE are steps of one "
           "pipeline that is fitted on the training portion only and re-fitted inside every cross-validation fold. "
           "The test set is used once, for the final scores.", "read")

ready = bool(features) and bool(models) and status != "blocked" and validity["ok"] and override_ok
if not (features and models):
    st.warning("Select at least one feature and one model.", icon=":material/warning:")
elif status == "blocked" or not validity["ok"]:
    st.error("Training is disabled until the blockers above are resolved.", icon=":material/block:")
elif not override_ok:
    st.warning("Confirm the override above to train on a discouraged target.", icon=":material/warning:")
if tuning == "grid" and len(frame) > 20_000:
    st.caption("Grid search on a dataset this size can take several minutes.")

if st.button("Train models", type="primary", icon=":material/play_arrow:", disabled=not ready):
    try:
        with st.status(f"Training {len(models)} model(s)…", expanded=True) as status_box:
            st.write(f"Splitting {len(frame):,} rows ({1 - test_size:.0%} train / {test_size:.0%} test, seed {seed})…")
            out = modeling.train_models(frame, target, features, task, models, test_size, seed, cv, tuning, imbalance, scale, primary)
            status_box.update(label=f"Trained {len(out.pipelines)} model(s). Best: {out.results['best_model']}", state="complete", expanded=False)
        out.results["readiness"] = ready_check
        state.set_training(out, config)
        st.rerun()
    except modeling.ModelingError as exc:
        st.error(str(exc), icon=":material/error:")
    except MemoryError:
        st.error("Training ran out of memory. Select fewer features or models, or a smaller dataset.", icon=":material/error:")

tr = state.training()
if tr is None:
    ui.empty_state("No models trained yet", "Configure the options above and choose “Train models”.")
else:
    res = tr.results
    if st.session_state["train_config"] != config:
        st.warning("Settings have changed since the last training run. The results below reflect the previous settings until you train again.",
                   icon=":material/history:")
    st.subheader("Latest training run")
    c = st.columns(4)
    c[0].metric("Best model", res["best_model"])
    best = res["models"][0]
    c[1].metric(f"Test {modeling.METRIC_LABELS[res['primary_metric']]}", "Unavailable" if best["metrics"].get(res["primary_metric"]) is None
                else f"{best['metrics'][res['primary_metric']]:.4f}", help="Held-out test set.")
    c[2].metric("Train / test rows", f"{res['n_train']:,} / {res['n_test']:,}")
    c[3].metric("Models trained", sum(1 for m in res["models"] if "error" not in m))
    for w in res["warnings"]:
        st.warning(w, icon=":material/warning:")
    for m in res["models"]:
        if "error" in m:
            st.error(f"{m['name']} failed: {m['error']}", icon=":material/error:")
    st.caption(f"Trained {res['trained_at']}. Ranked by {res['ranking_basis']}.")
    st.page_link("views/model_results.py", label="View results & diagnostics", icon=":material/arrow_forward:")
