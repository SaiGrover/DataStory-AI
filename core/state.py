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
    cols = set(ss["df"].columns)
    if ss["target"] not in cols:
        ss["target"] = None
    ss["completeness_exclude"] = [c for c in ss["completeness_exclude"] if c in cols]
    _bump()


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


# ------------------------------------------------------------------ results
def results() -> dict[str, Any]:
    """The shared results structure, rebuilt only when an input changed."""
    ss = st.session_state
    key = ss["_version"]
    if ss["_results_key"] != key or ss["_results"] is None:
        tr = ss["training"]
        ss["_results"] = build_results(
            ss["df"], ss["meta"], original=ss["raw_df"], history=ss["history"], rules=ss["rules"],
            exclude_from_completeness=ss["completeness_exclude"], target=ss["target"],
            modeling=tr.results if tr is not None else None, importance=ss["importance"],
        )
        ss["_results_key"] = key
    return ss["_results"]
