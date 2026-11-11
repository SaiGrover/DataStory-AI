"""Optional step — clustering & segmentation. No target column is needed."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from components import ui
from core import clustering as cl
from core import eda, state
from core.data_loader import to_csv_bytes
from core.formatting import metric, pct

if not ui.require_data("find segments in it"):
    st.stop()

ui.page_header("Optional · Segments", "Clustering & segmentation",
               "Group similar rows without a target column. Descriptions compare each cluster with the full dataset; "
               "they are observations, not business personas.")
ui.synthetic_notice()

frame = state.df()
p = state.results()["profile"]
cand = cl.cluster_feature_candidates(frame, p["kinds"])
usable = cand["usable"]
numeric_usable = [c for c in usable if p["kinds"].get(c) == "numeric"]

left, right = st.columns([1, 2.1], gap="large")
with left, st.container(border=True):
    st.subheader("Configure")
    if not usable:
        ui.callout("No column is suitable for clustering (identifiers, free text, dates and constant columns are excluded).", "na")
        st.stop()
    features = st.multiselect("Features", usable, default=numeric_usable[:10] or usable[:10], key="cl_feats")
    if cand["excluded"]:
        with st.expander(f"Excluded columns ({len(cand['excluded'])})"):
            for e in cand["excluded"]:
                st.markdown(f"- **{e['column']}** — {e['reason']}")
    for w in cand["warnings"]:
        st.caption(f":material/warning: {w}")
    algo = st.selectbox("Algorithm", list(cl.ALGORITHMS), format_func=cl.ALGORITHMS.get, key="cl_algo")
    params: dict = {}
    if algo in ("kmeans", "agglomerative"):
        params["n_clusters"] = st.slider("Number of clusters (k)", 2, 12, 3, key="cl_k")
        if algo == "agglomerative":
            params["linkage"] = st.selectbox("Linkage", ["ward", "average", "complete", "single"], key="cl_link")
    elif algo == "gmm":
        params["n_components"] = st.slider("Number of components", 2, 12, 3, key="cl_gk")
        params["covariance_type"] = st.selectbox("Covariance", ["full", "tied", "diag", "spherical"], key="cl_cov")
    else:
        params["eps"] = st.number_input("eps (neighbourhood radius, in scaled units)", 0.05, 20.0, 0.5, 0.05, key="cl_eps")
        params["min_samples"] = st.slider("min_samples", 2, 50, 5, key="cl_mins")
    scale = st.selectbox("Scaling", ["standard", "robust", "none"], key="cl_scale",
                         format_func={"standard": "Standard (z-score)", "robust": "Robust (median / IQR)", "none": "None"}.get)
    use_pca = st.checkbox("Reduce with PCA before clustering", key="cl_pca")
    n_comp = st.slider("PCA components", 2, 10, 3, key="cl_ncomp") if use_pca else None
    seed = int(st.number_input("Random seed", 0, 10_000, 42, key="cl_seed"))
    run = st.button("Run clustering", type="primary", icon=":material/hub:", width="stretch", disabled=not features)
    if algo in ("kmeans", "gmm", "agglomerative") and st.button("Elbow & silhouette for k = 2…8", icon=":material/ssid_chart:",
                                                                width="stretch", disabled=not features):
        try:
            with st.spinner("Fitting K-Means for each k…"):
                st.session_state["_cl_k_scan"] = cl.suggest_k(frame, features, scale=scale, seed=seed)
        except cl.ClusteringError as exc:
            st.error(str(exc))

if run:
    try:
        with st.spinner(f"Running {cl.ALGORITHMS[algo]}…"):
            res = cl.run_clustering(frame, features, algo, params, scale, "pca" if use_pca else None, n_comp, seed)
        state.set_clustering(res)
    except cl.ClusteringError as exc:
        st.error(str(exc), icon=":material/error:")

with right:
    scan = st.session_state.get("_cl_k_scan")
    if scan:
        with st.container(border=True):
            t = pd.DataFrame(scan)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t["k"], y=t["inertia"], name="Inertia (within-cluster sum of squares)", mode="lines+markers",
                                     line=dict(color=eda.PALETTE[0])))
            fig.add_trace(go.Scatter(x=t["k"], y=t["silhouette"], name="Silhouette", mode="lines+markers", yaxis="y2",
                                     line=dict(color=eda.PALETTE[1])))
            fig.update_layout(xaxis_title="k", yaxis_title="Inertia", yaxis2=dict(title="Silhouette", overlaying="y", side="right"))
            ui.chart(eda.style(fig, "K-Means scan (seeded sample of up to 5,000 rows)", height=320), "cl_scan")
            best = t.loc[t["silhouette"].idxmax()] if t["silhouette"].notna().any() else None
            if best is not None:
                ui.interpretation(f"The highest silhouette in this scan is {best['silhouette']:.3f} at k = {int(best['k'])}. "
                                  "Silhouette is one heuristic; choose k that is also interpretable.")

    stash = st.session_state.get("clustering")
    if not stash:
        ui.empty_state("No clustering run yet", "Choose features and an algorithm, then select “Run clustering”.")
        st.stop()
    res, names = stash["result"], stash.setdefault("names", {})
    if stash.get("data_key") != state.data_key():
        st.warning("The dataset changed after this clustering run. Results below describe the previous data — run clustering again.",
                   icon=":material/history:")
        st.stop()
    cfg = res["config"]

    def nm(label: int) -> str:
        return names.get(str(label)) or ("Noise" if label == -1 else f"Cluster {label}")

    c = st.columns(4)
    c[0].metric("Clusters", res["n_clusters"], help=f"{cfg['algorithm_label']} on {len(cfg['features'])} feature(s)")
    m = res["metrics"]
    c[1].metric("Silhouette", metric(m["silhouette"], 3) if m["silhouette"] is not None else "Not defined",
                help="−1 to 1; higher means rows sit closer to their own cluster than to others.")
    c[2].metric("Davies–Bouldin", metric(m["davies_bouldin"], 3) if m["davies_bouldin"] is not None else "Not defined",
                help="0 or more; lower means more compact, better separated clusters.")
    c[3].metric("Calinski–Harabasz", f"{m['calinski_harabasz']:,.1f}" if m["calinski_harabasz"] is not None else "Not defined",
                help="Higher means clusters are dense and well separated (relative measure; compare runs on the same data).")
    for n in res["metric_notes"]:
        st.caption(f":material/info: {n}")
    if res["noise_count"]:
        st.caption(f"Noise points (not assigned to any cluster): {res['noise_count']:,} ({pct(100 * res['noise_count'] / cfg['n_rows'])}).")

    a, b = st.columns(2, gap="medium")
    sizes = pd.DataFrame([{"cluster": nm(int(k)), "rows": v} for k, v in res["sizes"].items()])
    with a:
        fig = px.bar(sizes, x="cluster", y="rows", text="rows", color_discrete_sequence=[eda.SAGE])
        fig.update_xaxes(type="category", title=None)
        ui.chart(eda.style(fig, "Cluster sizes", height=340), "cl_sizes")
    with b:
        pr = res["projection"]
        proj = pd.DataFrame({"PC1": pr["x"], "PC2": pr["y"], "cluster": [nm(v) for v in pr["labels"]]})
        fig = px.scatter(proj, x="PC1", y="PC2", color="cluster", opacity=0.7, color_discrete_sequence=eda.PALETTE)
        fig.update_traces(marker=dict(size=5))
        ev = pr["explained_variance"]
        fig.update_layout(xaxis_title=f"PC1 ({pct(100 * ev[0])} of variance)", yaxis_title=f"PC2 ({pct(100 * ev[1])} of variance)")
        ui.chart(eda.style(fig, "PCA projection" + (" (sample of 5,000 rows)" if pr["sampled"] else ""), height=340), "cl_proj")
    st.caption(res["projection"]["note"])

    st.markdown("**What distinguishes each cluster**")
    for d, prof in zip(res["descriptions"], res["profiles"]):
        st.markdown(f"- {d.replace(prof['name'], nm(prof['cluster']), 1)}")
    st.caption("Each description lists the features whose cluster values differ most from the full dataset. "
               "Differences are descriptive; they do not explain why rows group together.")

    nums = cfg["numeric"]
    if nums:
        z = pd.DataFrame({nm(pf["cluster"]): {c_: pf["numeric"][c_]["std_diff"] for c_ in nums} for pf in res["profiles"]}).T
        fig = go.Figure(go.Heatmap(z=z.values, x=z.columns, y=z.index, colorscale=eda.DIVERGING, zmid=0,
                                   text=z.round(2).values, texttemplate="%{text}" if z.size <= 150 else None,
                                   hovertemplate="%{y} · %{x}: %{z:.2f} sd from overall mean<extra></extra>"))
        fig.update_yaxes(type="category", autorange="reversed")
        ui.chart(eda.style(fig, "Cluster profiles — difference from overall mean (in standard deviations)",
                           height=max(260, 34 * len(z) + 120)), "cl_heat")

        col = st.selectbox("Distribution by cluster", nums, key="cl_dist_col")
        lf = cl.labels_frame(frame, res, names)
        fig = px.box(lf.dropna(subset=["cluster"]), x="cluster", y=col, color="cluster", color_discrete_sequence=eda.PALETTE)
        fig.update_layout(showlegend=False)
        fig.update_xaxes(type="category", title=None)
        ui.chart(eda.style(fig, f"{col} by cluster", height=360), "cl_box")

    rows = []
    for pf in res["profiles"]:
        row = {"Cluster": nm(pf["cluster"]), "Rows": pf["size"], "Share": pct(pf["share_pct"])}
        for c_, v in pf["numeric"].items():
            row[f"{c_} (median)"] = v["median"]
        for c_, v in pf["categorical"].items():
            row[f"{c_} (most common)"] = f"{v['top']} · {pct(v['share_pct'])}"
        rows.append(row)
    st.markdown("**Cluster comparison**")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    for w in res["warnings"]:
        st.warning(w, icon=":material/warning:")

    st.markdown("**Rename clusters**")
    edit = pd.DataFrame([{"Label": int(k), "Name": nm(int(k))} for k in res["sizes"]])
    edited = st.data_editor(edit, hide_index=True, disabled=["Label"], key="cl_names_editor", width="stretch")
    default = {str(int(k)): ("Noise" if int(k) == -1 else f"Cluster {k}") for k in res["sizes"]}
    new_names = {str(int(r.Label)): str(r.Name).strip() for r in edited.itertuples()
                 if str(r.Name).strip() and str(r.Name).strip() != default[str(int(r.Label))]}
    if new_names != names:
        stash["names"] = new_names
        st.rerun()

    st.download_button("Download dataset with cluster labels (CSV)", to_csv_bytes(cl.labels_frame(frame, res, stash["names"])),
                       f"{Path(state.meta().name).stem}_clusters.csv", "text/csv", icon=":material/download:", type="primary")
    with st.expander("Reproducibility"):
        st.json(cfg)
