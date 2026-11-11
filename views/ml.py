"""Step 6 — configure and train models. Training runs only when the button is pressed."""
from __future__ import annotations

import streamlit as st

from components import ui
from core import modeling, state

if not ui.require_data("train models"):
    st.stop()

ui.page_header("Step 6 · Modeling", "Machine learning",
               "Choose a target, features and models. Preprocessing is fitted on training data only, and nothing trains until you ask.")
ui.synthetic_notice()

frame = state.analysis_df()
p = state.results()["profile"]
if len(frame) < len(state.df()):
    st.info(f"Training uses a seeded random sample of {len(frame):,} of {len(state.df()):,} rows.", icon=":material/info:")

left, right = st.columns([1, 1], gap="large")
with left, st.container(border=True):
    st.subheader("Target & features")
    target = ui.target_picker("Target column", key="ml_target", help="The column the models learn to predict.")
    if not target:
        ui.callout("Select a target column to continue.", "na")
        st.stop()
    auto_task, reason = modeling.detect_task(frame[target])
    choice = st.radio("Task type", ["Auto", "Classification", "Regression"], horizontal=True, key="ml_task",
                      help="Override when the automatic detection is wrong, e.g. numeric class codes.")
    task = auto_task if choice == "Auto" else choice.lower()
    st.caption(f"Detected **{auto_task}** because {reason}." + ("" if choice == "Auto" else f" Overridden to **{task}**."))
    candidates = [c for c in frame.columns if c != target]
    skip = set(p["id_candidates"]) | set(p["constant_columns"])
    default = [c for c in candidates if c not in skip]
    features = st.multiselect("Features", candidates, default=default, key=f"ml_feats_{target}",
                              help="Identifier-like and constant columns are left out by default.")
    if skip & set(candidates):
        st.caption("Left out by default: " + ", ".join(sorted(skip & set(candidates))) + ".")

with right, st.container(border=True):
    st.subheader("Models & evaluation")
    names = modeling.available_models(task)
    heavy = len(frame) * max(len(features), 1) > 2_000_000
    default_models = [n for n in names if not (heavy and "Gradient Boosting" in n)]
    models = st.multiselect("Models", names, default=default_models, key=f"ml_models_{task}")
    metrics = modeling.CLASSIFICATION_METRICS if task == "classification" else ["r2", "rmse", "mae", "mse"]
    c1, c2 = st.columns(2)
    primary = c1.selectbox("Rank models by", metrics, index=metrics.index("f1") if task == "classification" else 0,
                           format_func=modeling.METRIC_LABELS.get, key=f"ml_metric_{task}")
    test_size = c2.slider("Test set share", 0.1, 0.4, 0.2, 0.05, key="ml_test")
    c1, c2 = st.columns(2)
    cv = c1.selectbox("Cross-validation", [0, 3, 5, 10], index=2, format_func=lambda k: "Off" if not k else f"{k}-fold", key="ml_cv",
                      help="Run on the training data. When on, the best model is chosen by its cross-validated score, not the test set.")
    seed = int(c2.number_input("Random seed", 0, 10_000, 42, key="ml_seed"))
    c1, c2 = st.columns(2)
    tuning = c1.selectbox("Hyperparameter tuning", ["none", "random", "grid"], key="ml_tune",
                          format_func={"none": "Off", "random": "Randomized search", "grid": "Grid search (slower)"}.get)
    imbalance = "none"
    if task == "classification":
        imbalance = c2.selectbox("Class imbalance", ["none", "class_weight", "smote"], key="ml_imb",
                                 format_func={"none": "No handling", "class_weight": "Class weights", "smote": "SMOTE (training folds only)"}.get)
    scale = st.checkbox("Standardise numeric features", True, key="ml_scale",
                        help="Fitted on the training data inside the pipeline. Helps linear and distance-based models; harmless for trees.")

config = {"target": target, "task": task, "features": sorted(features), "models": models, "primary": primary, "test_size": test_size,
          "cv": cv, "seed": seed, "tuning": tuning, "imbalance": imbalance, "scale": scale}

ui.callout("<strong>How leakage is prevented.</strong> The data is split first. Imputation, scaling, encoding and SMOTE are steps of one "
           "pipeline that is fitted on the training portion only and re-fitted inside every cross-validation fold. "
           "The test set is used once, for the final scores.", "read")

ready = bool(features) and bool(models)
if not ready:
    st.warning("Select at least one feature and one model.", icon=":material/warning:")
if tuning == "grid" and len(frame) > 20_000:
    st.caption("Grid search on a dataset this size can take several minutes.")

if st.button("Train models", type="primary", icon=":material/play_arrow:", disabled=not ready):
    try:
        with st.status(f"Training {len(models)} model(s)…", expanded=True) as status:
            st.write(f"Splitting {len(frame):,} rows ({1 - test_size:.0%} train / {test_size:.0%} test, seed {seed})…")
            out = modeling.train_models(frame, target, features, task, models, test_size, seed, cv, tuning, imbalance, scale, primary)
            status.update(label=f"Trained {len(out.pipelines)} model(s). Best: {out.results['best_model']}", state="complete", expanded=False)
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
                else f"{best['metrics'][res['primary_metric']]:.4f}")
    c[2].metric("Train / test rows", f"{res['n_train']:,} / {res['n_test']:,}")
    c[3].metric("Models trained", sum(1 for m in res["models"] if "error" not in m))
    for w in res["warnings"]:
        st.warning(w, icon=":material/warning:")
    for m in res["models"]:
        if "error" in m:
            st.error(f"{m['name']} failed: {m['error']}", icon=":material/error:")
    st.caption(f"Trained {res['trained_at']}. Ranked by {res['ranking_basis']}.")
    st.page_link("views/model_results.py", label="View results & explainability", icon=":material/arrow_forward:")
