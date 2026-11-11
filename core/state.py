"""Session state: one place that owns the dataset, the cleaning history, model
artefacts and the cached results structure — and that invalidates dependent
results when something upstream changes.

Dependency chain:  dataset → cleaning → (target, rules) → models → importance
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from core import cleaning
from core.data_loader import DatasetMeta, sample_for_display
from core.results import build_results

_DEFAULTS: dict[str, Any] = {
    "raw_df": None, "df": None, "meta": None, "history": [], "rules": [], "completeness_exclude": [],
    "target": None, "training": None, "train_config": None, "importance": None,
    "chat": [], "narrative": None, "_version": 0, "_results": None, "_results_key": None,
    # analytics upgrade
    "charts": [], "clustering": None, "domain_override": None, "report_config": None, "_report": None,
    "eda_generated": 0, "_overview": None, "_overview_key": None, "chat_spec": None,
}

# Workflow stages and what "complete" means. A stage is complete only when its
# operation actually ran on the current data, never because it was visited.
STAGE_RULES = {
    "dataset": "A dataset is loaded.",
    "quality": "Quality metrics are computed for the active dataset version.",
    "cleaning": "At least one cleaning step has been applied.",
    "compare": "The cleaned data differs from the original.",
    "eda": "At least one chart has been generated or saved.",
    "clustering": "A clustering run exists for the current data version.",
    "insights": "A data story has been written from the current findings.",
    "ml": "Models have been trained on the current data.",
    "results": "Held-out feature importance has been computed.",
    "assistant": "The copilot has answered at least one question.",
    "reports": "A report has been generated for the current analysis.",
}


def init() -> None:
    for k, v in _DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v.copy() if isinstance(v, list) else v


def _bump() -> None:
    st.session_state["_version"] += 1
    st.session_state["narrative"] = None


def has_data() -> bool:
    return st.session_state.get("df") is not None


def df() -> pd.DataFrame:
    return st.session_state["df"]


def raw_df() -> pd.DataFrame:
    return st.session_state["raw_df"]


def meta() -> DatasetMeta:
    return st.session_state["meta"]


def analysis_df() -> pd.DataFrame:
    """Working frame, down-sampled with a fixed seed when very large (charts / training)."""
    return sample_for_display(df())


# ------------------------------------------------------------------ dataset
def set_dataset(frame: pd.DataFrame, m: DatasetMeta) -> None:
    for k, v in _DEFAULTS.items():
        st.session_state[k] = v.copy() if isinstance(v, list) else v
    st.session_state["raw_df"] = frame
    st.session_state["df"] = frame
    st.session_state["meta"] = m
    st.session_state["target"] = m.suggested_target if m.suggested_target in frame.columns else None
    _bump()


def _data_changed() -> None:
    """The working data changed: models and everything derived from them are stale."""
    ss = st.session_state
    ss["training"], ss["train_config"], ss["importance"] = None, None, None
    ss["_report"] = None
    cols = set(ss["df"].columns)
    if ss["target"] not in cols:
        ss["target"] = None
    ss["completeness_exclude"] = [c for c in ss["completeness_exclude"] if c in cols]
    _bump()


def data_version() -> str:
    """Active dataset version: v0 is the original; each applied cleaning step adds one."""
    return f"v{len(st.session_state['history'])}"


def data_key() -> str:
    m = st.session_state.get("meta")
    return f"{m.dataset_id if m else '-'}:{data_version()}"


def apply_cleaning(new_df: pd.DataFrame, record: dict[str, Any]) -> None:
    st.session_state["df"] = new_df
    st.session_state["history"] = st.session_state["history"] + [record]
    _data_changed()


def undo_last() -> None:
    hist = st.session_state["history"][:-1]
    st.session_state["df"] = cleaning.replay(raw_df(), hist) if hist else raw_df()
    st.session_state["history"] = hist
    _data_changed()


def reset_cleaning() -> None:
    st.session_state["df"] = raw_df()
    st.session_state["history"] = []
    _data_changed()


# ---------------------------------------------------- target / rules / models
def target() -> str | None:
    return st.session_state["target"]


def set_target(t: str | None) -> None:
    if t != st.session_state["target"]:
        st.session_state["target"] = t
        _bump()


def set_rules(rules: list[dict[str, Any]]) -> None:
    st.session_state["rules"] = rules
    _bump()


def set_completeness_exclude(cols: list[str]) -> None:
    if cols != st.session_state["completeness_exclude"]:
        st.session_state["completeness_exclude"] = cols
        _bump()


def set_training(output: Any, config: dict[str, Any]) -> None:
    st.session_state["training"] = output
    st.session_state["train_config"] = config
    st.session_state["importance"] = None
    _bump()


def training() -> Any:
    return st.session_state["training"]


def set_importance(imp: dict[str, Any] | None) -> None:
    st.session_state["importance"] = imp
    _bump()


def importance() -> dict[str, Any] | None:
    return st.session_state["importance"]


# ---------------------------------------------------------- session I/O
SNAPSHOT_KEYS = ["raw_df", "df", "meta", "history", "rules", "completeness_exclude", "target", "train_config", "training",
                 "importance", "clustering", "charts", "chat", "narrative", "report_config", "domain_override"]


def snapshot() -> dict[str, Any]:
    """Plain dict of everything a session package stores (see core.session_io)."""
    return {k: st.session_state.get(k) for k in SNAPSHOT_KEYS}


def clear() -> None:
    """Start a new analysis: drop every piece of analysis state in this tab."""
    for k, v in _DEFAULTS.items():
        st.session_state[k] = v.copy() if isinstance(v, list) else v
    for k in [k for k in st.session_state if str(k).startswith(("_cl_", "_sess", "_confirm", "_loaded_upload", "_imp", "_nl"))]:
        del st.session_state[k]
    _bump()


def restore(restored: dict[str, Any]) -> None:
    """Load the state returned by core.session_io.import_session into this tab."""
    clear()
    ss = st.session_state
    for k in SNAPSHOT_KEYS:
        if k in restored and k not in ("clustering",):
            ss[k] = restored[k]
    ss["training"], ss["importance"] = None, None
    ss["previous_training_results"] = restored.get("previous_training_results")
    cl = restored.get("clustering")
    if cl and cl.get("result"):
        ss["clustering"] = {"result": cl["result"], "names": cl.get("names") or {}, "version": ss["_version"], "data_key": data_key()}
    ss["eda_generated"] = len(ss.get("charts") or [])
    _bump()


# ------------------------------------------------------- charts / clustering
def save_chart(spec: dict[str, Any], title: str, source: str = "EDA") -> dict[str, Any]:
    """Store a chart *specification* so it can be re-rendered, reported, exported and restored."""
    import hashlib
    import json
    from datetime import datetime, timezone

    cid = hashlib.sha1(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:10]
    existing = next((c for c in st.session_state["charts"] if c["id"] == cid), None)
    if existing:
        return existing
    rec = {"id": cid, "title": title, "spec": spec, "source": source, "data_version": data_version(),
           "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "in_report": True}
    st.session_state["charts"] = st.session_state["charts"] + [rec]
    mark_eda()
    _bump()
    return rec


def remove_chart(cid: str) -> None:
    st.session_state["charts"] = [c for c in st.session_state["charts"] if c["id"] != cid]
    _bump()


def charts() -> list[dict[str, Any]]:
    return st.session_state["charts"]


def mark_eda() -> None:
    st.session_state["eda_generated"] = st.session_state.get("eda_generated", 0) + 1


def set_clustering(result: dict[str, Any] | None, names: dict[str, str] | None = None) -> None:
    if result is None:
        st.session_state["clustering"] = None
    else:
        st.session_state["clustering"] = {"result": result, "names": names or {}, "version": st.session_state["_version"],
                                          "data_key": data_key()}
    _bump()


def clustering() -> dict[str, Any] | None:
    """The clustering run, or None when there is none for the current data version."""
    c = st.session_state.get("clustering")
    return c if c and c.get("data_key") == data_key() else None


def set_domain(domain: str | None) -> None:
    st.session_state["domain_override"] = domain
    st.session_state["_overview"] = None


def overview() -> dict[str, Any]:
    """Dataset-aware overview, cached per results version."""
    from core.overview import build_overview

    ss = st.session_state
    key = (ss["_version"], ss["domain_override"])
    if ss.get("_overview_key") != key or ss.get("_overview") is None:
        r = results()
        ss["_overview"] = build_overview(ss["df"], r["profile"], r["quality"], ss["target"], ss["domain_override"])
        ss["_overview_key"] = key
    return ss["_overview"]


def stage_status() -> dict[str, dict[str, Any]]:
    """Data-driven status per workflow stage: done / available (+ the rule or the prerequisite)."""
    ss = st.session_state
    has = has_data()
    tr = ss.get("training")
    rep = ss.get("_report")
    need = "Load a dataset first."
    out: dict[str, dict[str, Any]] = {}

    def put(k, done, available=None, why=None):
        available = has if available is None else available
        out[k] = {"done": bool(has and done), "available": bool(available),
                  "why": why or (STAGE_RULES[k] if available else need)}

    put("dataset", has, True)
    put("quality", True)
    put("cleaning", bool(has and ss["history"]))
    put("compare", bool(has and ss["history"]), has and bool(ss["history"]),
        None if (has and ss["history"]) else ("Apply a cleaning step first." if has else need))
    put("eda", ss.get("eda_generated", 0) > 0 or bool(ss.get("charts")))
    put("clustering", has and clustering() is not None)
    put("insights", ss.get("narrative") is not None)
    put("ml", tr is not None)
    put("results", ss.get("importance") is not None, tr is not None,
        None if tr is not None else ("Train a model first." if has else need))
    put("assistant", any(m.get("role") == "assistant" for m in ss.get("chat", [])))
    put("reports", bool(rep) and rep.get("sig", (None,))[0] == ss["_version"])
    return out


# ------------------------------------------------------------------ results
def results() -> dict[str, Any]:
    """The shared results structure, rebuilt only when an input changed."""
    ss = st.session_state
    key = ss["_version"]
    if ss["_results_key"] != key or ss["_results"] is None:
        tr = ss["training"]
        cl = clustering()
        ss["_results"] = build_results(
            ss["df"], ss["meta"], original=ss["raw_df"], history=ss["history"], rules=ss["rules"],
            exclude_from_completeness=ss["completeness_exclude"], target=ss["target"],
            modeling=tr.results if tr is not None else None, importance=ss["importance"],
            clustering=cluster_summary(cl) if cl else None, charts=ss.get("charts", []),
        )
        ss["_results_key"] = key
    return ss["_results"]


def cluster_summary(cl: dict[str, Any]) -> dict[str, Any]:
    r = cl["result"]
    return {"algorithm": r["config"].get("algorithm"), "n_clusters": r["n_clusters"], "noise_count": r.get("noise_count", 0),
            "sizes": r["sizes"], "metrics": r["metrics"], "metric_notes": r.get("metric_notes", {}),
            "descriptions": r.get("descriptions", []), "names": cl.get("names") or {}, "features": r["config"].get("features"),
            "config": r["config"], "warnings": r.get("warnings", [])}
