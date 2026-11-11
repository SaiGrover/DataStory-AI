"""Clustering & segmentation (unsupervised; no target needed).

Every output is computed from the data passed in and is JSON-serialisable.
Cluster descriptions state observed differences against the full dataset;
they never assign business personas.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from pandas.api import types as ptypes
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from core.formatting import pct

ALGORITHMS = {"kmeans": "K-Means", "dbscan": "DBSCAN", "agglomerative": "Agglomerative", "gmm": "Gaussian Mixture"}
MAX_CATEGORY_LEVELS = 30
MAX_AGGLOMERATIVE_ROWS = 15_000
PROJECTION_ROWS = 5_000
_ID_NAME = re.compile(r"(^|[_\s])(id|uuid|guid|key|index)($|[_\s])|id$", re.IGNORECASE)


class ClusteringError(ValueError):
    """User-readable reason clustering could not run."""


def _kind(s: pd.Series, kinds: dict[str, str] | None, name: str) -> str:
    if kinds and name in kinds:
        return kinds[name]
    if ptypes.is_bool_dtype(s):
        return "categorical"
    if ptypes.is_datetime64_any_dtype(s):
        return "datetime"
    if ptypes.is_numeric_dtype(s):
        return "numeric"
    nn = s.dropna().astype(str)
    if len(nn) and (nn.str.len().mean() > 40 or (nn.nunique() > 50 and nn.nunique() / len(nn) > 0.5)):
        return "text"
    return "categorical"


def cluster_feature_candidates(df: pd.DataFrame, kinds: dict[str, str] | None = None) -> dict[str, Any]:
    usable, excluded, warnings = [], [], []
    n = len(df)
    for c in df.columns:
        s = df[c]
        k = _kind(s, kinds, c)
        nn = s.dropna()
        uniq = nn.nunique()
        miss = 100 * s.isna().mean() if n else 0
        reason = None
        if len(nn) == 0 or uniq <= 1:
            reason = "constant or empty"
        elif miss > 50:
            reason = f"mostly missing ({pct(miss)})"
        elif k == "datetime":
            reason = "datetime (not a clustering feature)"
        elif n >= 20 and uniq == len(nn) == n and (
                (k == "numeric" and (ptypes.is_integer_dtype(s) or _ID_NAME.search(str(c))))
                or (k != "numeric" and nn.astype(str).str.len().mean() <= 40)):
            reason = "identifier-like (unique per row)"
        elif k == "text":
            reason = "free text"
        elif _ID_NAME.search(str(c)) and uniq > 0.9 * max(len(nn), 1):
            reason = "identifier-like name"
        elif k == "categorical" and uniq > MAX_CATEGORY_LEVELS:
            reason = f"{uniq} categories (limit {MAX_CATEGORY_LEVELS})"
        if reason:
            excluded.append({"column": c, "reason": reason})
        else:
            usable.append(c)
            if k == "categorical" and uniq > 10:
                warnings.append(f"'{c}' has {uniq} categories; one-hot encoding adds many dimensions.")
    return {"usable": usable, "excluded": excluded, "warnings": warnings}


def _split(df: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    if not features:
        raise ClusteringError("Select at least one feature.")
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ClusteringError(f"Column(s) not found: {', '.join(missing)}")
    num, cat = [], []
    for f in features:
        s = df[f]
        if s.notna().sum() == 0:
            raise ClusteringError(f"'{f}' is entirely missing.")
        if ptypes.is_numeric_dtype(s) and not ptypes.is_bool_dtype(s):
            num.append(f)
        elif ptypes.is_datetime64_any_dtype(s):
            raise ClusteringError(f"'{f}' is a datetime column; clustering needs numeric or categorical features.")
        else:
            cat.append(f)
    return num, cat


def _preprocessor(num, cat, scale):
    steps: list = [("impute", SimpleImputer(strategy="median"))]
    if scale == "standard":
        steps.append(("scale", StandardScaler()))
    elif scale == "robust":
        steps.append(("scale", RobustScaler()))
    elif scale != "none":
        raise ClusteringError(f"Unknown scaling '{scale}'.")
    tr = []
    if num:
        tr.append(("num", Pipeline(steps), num))
    if cat:
        tr.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat))
    return ColumnTransformer(tr, remainder="drop")


def _describe_preprocessing(num, cat, scale, reduce, n_components):
    parts = []
    if num:
        parts.append("numeric: median imputation" + ({"standard": " + standard scaling", "robust": " + robust scaling (median/IQR)"}.get(scale, " (no scaling)")))
    if cat:
        parts.append("categorical: most-frequent imputation + one-hot encoding")
    if reduce == "pca":
        parts.append(f"then PCA to {n_components} components")
    return "; ".join(parts) + "."


def _matrix(df, features, scale, reduce, n_components, seed):
    num, cat = _split(df, features)
    data = df[features].copy()
    for c in cat:
        data[c] = data[c].astype(object).where(data[c].notna(), np.nan).map(lambda v: v if pd.isna(v) else str(v))
    X = _preprocessor(num, cat, scale).fit_transform(data)
    X = np.asarray(X, dtype=float)
    if reduce == "pca":
        nc = int(n_components or min(5, X.shape[1]))
        nc = max(1, min(nc, X.shape[1], X.shape[0]))
        X = PCA(n_components=nc, random_state=seed).fit_transform(X)
        n_components = nc
    elif reduce not in (None, "none"):
        raise ClusteringError(f"Unknown reduction '{reduce}'.")
    return X, num, cat, n_components


def _fit(X, algorithm, params, seed):
    p = dict(params or {})
    if algorithm == "kmeans":
        k = int(p.get("n_clusters", 3))
        if k < 2 or k > len(X):
            raise ClusteringError(f"K-Means needs between 2 and {len(X)} clusters.")
        return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
    if algorithm == "gmm":
        k = int(p.get("n_components", p.get("n_clusters", 3)))
        if k < 2 or k > len(X):
            raise ClusteringError(f"Gaussian Mixture needs between 2 and {len(X)} components.")
        return GaussianMixture(n_components=k, covariance_type=p.get("covariance_type", "full"), random_state=seed).fit_predict(X)
    if algorithm == "agglomerative":
        if len(X) > MAX_AGGLOMERATIVE_ROWS:
            raise ClusteringError(f"Agglomerative clustering is limited to {MAX_AGGLOMERATIVE_ROWS:,} rows (memory grows with the square "
                                  f"of the row count); this dataset has {len(X):,}. Use K-Means, or filter/sample the data first.")
        k = int(p.get("n_clusters", 3))
        if k < 2 or k > len(X):
            raise ClusteringError(f"Agglomerative clustering needs between 2 and {len(X)} clusters.")
        return AgglomerativeClustering(n_clusters=k, linkage=p.get("linkage", "ward")).fit_predict(X)
    if algorithm == "dbscan":
        return DBSCAN(eps=float(p.get("eps", 0.5)), min_samples=int(p.get("min_samples", 5))).fit_predict(X)
    raise ClusteringError(f"Unknown algorithm '{algorithm}'.")


def _metrics(X, labels, algorithm, max_rows, seed):
    out = {"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None}
    notes: list[str] = []
    mask = labels != -1
    if algorithm == "dbscan" and (~mask).any():
        notes.append(f"{int((~mask).sum()):,} noise points were excluded from the metrics.")
    Xm, lm = X[mask], labels[mask]
    k = len(np.unique(lm))
    if k < 2:
        notes.append("Metrics are undefined: fewer than two clusters were found.")
        return out, notes
    if k >= len(lm):
        notes.append("Metrics are undefined: every point forms its own cluster.")
        return out, notes
    if len(lm) > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(lm), max_rows, replace=False)
        if len(np.unique(lm[idx])) >= 2:
            out["silhouette"] = float(silhouette_score(Xm[idx], lm[idx]))
            notes.append(f"Silhouette computed on a seeded random sample of {max_rows:,} of {len(lm):,} rows.")
    else:
        out["silhouette"] = float(silhouette_score(Xm, lm))
    out["davies_bouldin"] = float(davies_bouldin_score(Xm, lm))
    out["calinski_harabasz"] = float(calinski_harabasz_score(Xm, lm))
    return out, notes


def _name(label: int) -> str:
    return "Noise" if label == -1 else f"Cluster {label}"


def _profiles(df, num, cat, labels):
    n = len(df)
    profiles, descriptions = [], []
    overall_num = {c: {"mean": float(df[c].mean()), "median": float(df[c].median()), "std": float(df[c].std(ddof=0) or 0)} for c in num}
    overall_cat = {}
    for c in cat:
        vc = df[c].astype(str).where(df[c].notna(), "(missing)").value_counts(normalize=True)
        overall_cat[c] = vc
    for lab in sorted(np.unique(labels)):
        m = labels == lab
        sub = df.loc[m]
        prof: dict[str, Any] = {"cluster": int(lab), "name": _name(int(lab)), "size": int(m.sum()), "share_pct": 100 * m.sum() / n,
                                "numeric": {}, "categorical": {}}
        diffs = []
        for c in num:
            o = overall_num[c]
            mean, med = sub[c].mean(), sub[c].median()
            z = float((mean - o["mean"]) / o["std"]) if o["std"] and pd.notna(mean) else 0.0
            prof["numeric"][c] = {"mean": None if pd.isna(mean) else float(mean), "median": None if pd.isna(med) else float(med),
                                  "overall_mean": o["mean"], "overall_median": o["median"], "std_diff": z}
            diffs.append((abs(z), "num", c))
        for c in cat:
            vc = sub[c].astype(str).where(sub[c].notna(), "(missing)").value_counts(normalize=True)
            if vc.empty:
                continue
            top = vc.index[0]
            s_in, s_all = float(vc.iloc[0] * 100), float(overall_cat[c].get(top, 0) * 100)
            prof["categorical"][c] = {"top": str(top), "share_pct": s_in, "overall_share_pct": s_all}
            diffs.append((abs(s_in - s_all) / 25.0, "cat", c))  # 25 pp difference ~ 1 sd for ranking
        parts = []
        for score, kind, c in sorted(diffs, reverse=True)[:3]:
            if score < 0.2:
                continue
            if kind == "num":
                d = prof["numeric"][c]
                parts.append(f"{c} median {d['median']:.4g} vs {d['overall_median']:.4g} overall")
            else:
                d = prof["categorical"][c]
                parts.append(f"{c}='{d['top']}' {pct(d['share_pct'])} vs {pct(d['overall_share_pct'])} overall")
        lead = f"{prof['name']} ({prof['size']:,} rows, {pct(prof['share_pct'])})"
        descriptions.append(lead + (": " + "; ".join(parts) + "." if parts else ": no feature differs markedly from the full dataset."))
        profiles.append(prof)
    return profiles, descriptions


def _projection(X, labels, seed):
    idx = np.arange(len(X))
    if len(idx) > PROJECTION_ROWS:
        idx = np.sort(np.random.default_rng(seed).choice(idx, PROJECTION_ROWS, replace=False))
    if X.shape[1] >= 2:
        pca = PCA(n_components=2, random_state=seed).fit(X)
        P = pca.transform(X[idx])
        evr = [float(v) for v in pca.explained_variance_ratio_]
    else:
        P = np.column_stack([X[idx, 0], np.zeros(len(idx))])
        evr = [1.0, 0.0]
    return {"x": P[:, 0].tolist(), "y": P[:, 1].tolist(), "labels": [int(v) for v in labels[idx]], "rows": [int(i) for i in idx],
            "explained_variance": evr, "sampled": len(idx) < len(X),
            "note": "Two-dimensional PCA view of the preprocessed features. It is a reduced-dimensional view and does not "
                    "prove that clusters are separated in the original feature space."}


def run_clustering(df: pd.DataFrame, features: list[str], algorithm: str = "kmeans", params: dict[str, Any] | None = None,
                   scale: str = "standard", reduce: str | None = None, n_components: int | None = None, seed: int = 42,
                   max_rows_for_metrics: int = 10_000) -> dict[str, Any]:
    if algorithm not in ALGORITHMS:
        raise ClusteringError(f"Unknown algorithm '{algorithm}'.")
    if len(df) < 4:
        raise ClusteringError("At least 4 rows are needed for clustering.")
    X, num, cat, n_components = _matrix(df, features, scale, reduce, n_components, seed)
    try:
        labels = np.asarray(_fit(X, algorithm, params, seed), dtype=int)
    except ClusteringError:
        raise
    except (ValueError, MemoryError) as exc:
        raise ClusteringError(f"{ALGORITHMS[algorithm]} failed: {exc}") from exc
    metrics, notes = _metrics(X, labels, algorithm, max_rows_for_metrics, seed)
    sizes = {str(int(k)): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
    clusters = [k for k in sizes if k != "-1"]
    noise = sizes.get("-1", 0)
    warnings = []
    if scale == "none" and len(num) > 1:
        warnings.append("Features are unscaled: columns with large numeric ranges dominate the distance measure.")
    if cat:
        width = X.shape[1] if reduce != "pca" else None
        if width and width > 3 * max(len(features), 1):
            warnings.append(f"One-hot encoding expanded {len(features)} features to {width} dimensions; distances may be dominated by categories.")
    small = [k for k in clusters if sizes[k] < max(5, 0.01 * len(df))]
    if small:
        warnings.append(f"{len(small)} cluster(s) hold fewer than 1% of rows (or under 5 rows); they may be unstable.")
    if algorithm == "dbscan" and noise > 0.5 * len(df):
        warnings.append(f"DBSCAN labelled {pct(100 * noise / len(df))} of rows as noise; try a larger eps or smaller min_samples.")
    warnings.append("Cluster assignments depend on the chosen features, scaling and algorithm settings; re-run with other settings to check robustness.")
    profiles, descriptions = _profiles(df, num, cat, labels)
    return {
        "labels": labels.tolist(), "index": [x.item() if hasattr(x, "item") else x for x in df.index],
        "n_clusters": len(clusters), "noise_count": noise, "sizes": sizes,
        "metrics": metrics, "metric_notes": notes, "profiles": profiles, "descriptions": descriptions,
        "projection": _projection(X, labels, seed), "warnings": warnings,
        "config": {"features": list(features), "numeric": num, "categorical": cat, "algorithm": algorithm,
                   "algorithm_label": ALGORITHMS[algorithm], "params": dict(params or {}), "scale": scale,
                   "reduce": reduce, "n_components": n_components, "seed": seed, "n_rows": len(df),
                   "preprocessing": _describe_preprocessing(num, cat, scale, reduce, n_components),
                   "sklearn_version": sklearn.__version__},
    }


def suggest_k(df: pd.DataFrame, features: list[str], k_min: int = 2, k_max: int = 8, scale: str = "standard",
              seed: int = 42, sample: int = 5000) -> list[dict[str, Any]]:
    data = df if len(df) <= sample else df.sample(sample, random_state=seed)
    X, *_ = _matrix(data, features, scale, None, None, seed)
    out = []
    for k in range(max(2, k_min), min(k_max, len(X) - 1) + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        sil = float(silhouette_score(X, km.labels_)) if len(np.unique(km.labels_)) > 1 else None
        out.append({"k": k, "inertia": float(km.inertia_), "silhouette": sil})
    return out


def stability(df: pd.DataFrame, features: list[str], algorithm: str = "kmeans", params: dict[str, Any] | None = None,
              scale: str = "standard", n_runs: int = 5, seed: int = 42) -> dict[str, Any]:
    """Adjusted Rand index between runs with different seeds (1 = identical partitions)."""
    if algorithm not in ("kmeans", "gmm"):
        raise ClusteringError("Seed stability applies to K-Means and Gaussian Mixture only (the others are deterministic).")
    X, *_ = _matrix(df, features, scale, None, None, seed)
    runs = [np.asarray(_fit(X, algorithm, params, seed + i)) for i in range(n_runs)]
    scores = [float(adjusted_rand_score(runs[0], r)) for r in runs[1:]]
    return {"ari_mean": float(np.mean(scores)) if scores else None, "ari_min": float(np.min(scores)) if scores else None,
            "n_runs": n_runs, "scores": scores}


def labels_frame(df: pd.DataFrame, result: dict[str, Any], names: dict[str, str] | None = None) -> pd.DataFrame:
    names = {str(k): v for k, v in (names or {}).items() if str(v).strip()}
    lab = pd.Series(result["labels"], index=result["index"])
    out = df.copy()
    aligned = lab.reindex(out.index)
    out["cluster"] = [None if pd.isna(v) else names.get(str(int(v)), _name(int(v))) for v in aligned]
    return out
