"""Export and import analysis sessions as validated, versioned ZIP packages.

A package holds plain JSON and (optionally) the original dataset as CSV:

* ``manifest.json`` — schema version, creation time, dataset metadata, frame
  hashes and a SHA-256 checksum for every other member.
* ``state.json``    — cleaning history, rules, target, configurations, saved
  charts, clustering output, chat text and report settings.
* ``data/original.csv`` — only when the user chooses to include the data.

Nothing executable is ever stored or loaded: no pickle, no eval, no YAML.
Fitted models are not portable, so only their configuration and recorded
results are saved; they must be retrained after import.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from datetime import date, datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from core import cleaning
from core.data_loader import DataLoadError, DatasetMeta, frame_hash, load_upload, to_csv_bytes

SCHEMA_VERSION = 1
SUPPORTED_SCHEMAS = {1}
MAX_PACKAGE_MB = 50
MAX_UNCOMPRESSED_MB = 200
MAX_RATIO = 200  # compressed → uncompressed expansion allowed per member
APP_NAME = "DataStory AI"

MANIFEST, STATE, DATA = "manifest.json", "state.json", "data/original.csv"
ALLOWED_MEMBERS = {MANIFEST, STATE, DATA}

RETRAIN_NOTE = "Models must be retrained; fitted models are not stored in session packages."
IMPORTANCE_NOTE = "Feature importance was not restored: it depends on a fitted model. Recompute it after retraining."


class SessionError(ValueError):
    """The package is invalid, tampered with, or does not match the supplied dataset."""


# ------------------------------------------------------------------ JSON
def _default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if not math.isfinite(v) else v
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (pd.Timestamp, datetime, date)):
        return o.isoformat()
    if isinstance(o, (set, tuple)):
        return list(o)
    if o is pd.NA or o is pd.NaT:
        return None
    raise TypeError(f"not serialisable: {type(o).__name__}")


def _clean(x: Any) -> Any:
    """Recursively convert to JSON-safe values; NaN/inf → None; drop non-serialisable leaves."""
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            try:
                out[str(k)] = _clean(v)
            except TypeError:
                continue
        return out
    if isinstance(x, (list, tuple, set)):
        return [_clean(v) for v in x]
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if x is None or isinstance(x, (str, int, bool)):
        return x
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    return _clean(_default(x))


def _dumps(obj: Any) -> bytes:
    return json.dumps(_clean(obj), ensure_ascii=False, indent=1, allow_nan=False).encode("utf-8")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _roundtrip(raw_df: pd.DataFrame) -> pd.DataFrame:
    return load_upload("original.csv", to_csv_bytes(raw_df))[0]


def _chat_for_export(chat: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    keep = ("role", "content", "answer", "note", "sources", "spec", "mode", "tool", "followups", "assumptions", "created_at")
    out = []
    for m in chat or []:
        if not isinstance(m, dict):
            continue
        msg = {k: m[k] for k in keep if k in m}
        try:
            out.append(_clean(msg))
        except TypeError:
            continue
    return out


# ----------------------------------------------------------------- export
def export_session(snap: dict[str, Any], include_data: bool = True, app_version: str = "2.0") -> bytes:
    raw_df: pd.DataFrame = snap["raw_df"]
    meta: DatasetMeta = snap["meta"]
    history = list(snap.get("history") or [])
    rt = _roundtrip(raw_df)
    rt_hash = frame_hash(rt)
    try:
        current_rt_hash = frame_hash(cleaning.replay(rt, history)) if history else rt_hash
    except Exception:  # noqa: BLE001 - recorded as unknown; import will warn
        current_rt_hash = None
    current = snap.get("df") if snap.get("df") is not None else raw_df
    tr = snap.get("training")
    clustering = snap.get("clustering")
    state = {
        "history": history,
        "rules": list(snap.get("rules") or []),
        "completeness_exclude": list(snap.get("completeness_exclude") or []),
        "target": snap.get("target"),
        "train_config": snap.get("train_config"),
        "training_results": getattr(tr, "results", None) if tr is not None else None,
        "fitted_models_included": False,
        "importance_computed": bool(snap.get("importance")),
        "clustering": clustering,
        "charts": list(snap.get("charts") or []),
        "chat": _chat_for_export(snap.get("chat")),
        "narrative": snap.get("narrative"),
        "report_config": snap.get("report_config"),
        "domain_override": snap.get("domain_override"),
    }
    files: dict[str, bytes] = {STATE: _dumps(state)}
    if include_data:
        files[DATA] = to_csv_bytes(raw_df)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "app": {"name": APP_NAME, "version": app_version},
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "include_data": include_data,
        "files": {name: _sha(b) for name, b in files.items()},
        "dataset": {
            "meta": meta.to_dict(),
            "original_frame_hash": frame_hash(raw_df),
            "original_roundtrip_hash": rt_hash,
            "current_frame_hash": frame_hash(current),
            "current_roundtrip_hash": current_rt_hash,
            "original_shape": list(raw_df.shape),
            "current_shape": list(current.shape),
            "columns": [str(c) for c in raw_df.columns],
        },
        "contents": {"cleaning_steps": len(history), "charts": len(state["charts"]), "chat_messages": len(state["chat"]),
                     "has_model_results": state["training_results"] is not None, "has_clustering": clustering is not None},
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(MANIFEST, _dumps(manifest))
        for name, b in files.items():
            z.writestr(name, b)
    return buf.getvalue()


# --------------------------------------------------------------- reading
def _open(data: bytes) -> dict[str, bytes]:
    if not isinstance(data, (bytes, bytearray)) or not data:
        raise SessionError("The session file is empty.")
    if len(data) > MAX_PACKAGE_MB * 1024 * 1024:
        raise SessionError(f"The session file is larger than {MAX_PACKAGE_MB} MB.")
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise SessionError("This is not a DataStory session package (not a ZIP file).") from exc
    with z:
        infos = z.infolist()
        total = 0
        for info in infos:
            name = info.filename
            if name.startswith("/") or "\\" in name or ".." in name.split("/") or ":" in name:
                raise SessionError(f"Rejected unsafe path in package: {name!r}.")
            if info.is_dir():
                if name.rstrip("/") + "/" != "data/":
                    raise SessionError(f"Unexpected folder in package: {name!r}.")
                continue
            if name not in ALLOWED_MEMBERS:
                raise SessionError(f"Unexpected file in package: {name!r}. Only {', '.join(sorted(ALLOWED_MEMBERS))} are allowed.")
            if info.flag_bits & 0x1:
                raise SessionError("Encrypted packages are not supported.")
            total += info.file_size
            if info.compress_size and info.file_size / max(info.compress_size, 1) > MAX_RATIO and info.file_size > 1024 * 1024:
                raise SessionError("The package expands suspiciously when decompressed and was rejected.")
        if total > MAX_UNCOMPRESSED_MB * 1024 * 1024:
            raise SessionError(f"The package contents exceed {MAX_UNCOMPRESSED_MB} MB when decompressed.")
        out: dict[str, bytes] = {}
        read = 0
        for info in infos:
            if info.is_dir():
                continue
            with z.open(info) as fh:
                b = fh.read(MAX_UNCOMPRESSED_MB * 1024 * 1024 + 1)
            read += len(b)
            if read > MAX_UNCOMPRESSED_MB * 1024 * 1024:
                raise SessionError(f"The package contents exceed {MAX_UNCOMPRESSED_MB} MB when decompressed.")
            out[info.filename] = b
    if MANIFEST not in out or STATE not in out:
        raise SessionError("The package is missing manifest.json or state.json.")
    return out


def _json(b: bytes, name: str) -> Any:
    try:
        return json.loads(b.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SessionError(f"{name} is not valid JSON.") from exc


def _check(obj: Any, schema: dict[str, Any], where: str) -> None:
    """Tiny type validator: schema maps key → type or tuple of types (None allowed via type(None))."""
    if not isinstance(obj, dict):
        raise SessionError(f"{where} must be an object.")
    for key, typ in schema.items():
        if key not in obj:
            raise SessionError(f"{where} is missing '{key}'.")
        if not isinstance(obj[key], typ):
            raise SessionError(f"{where}: '{key}' has the wrong type.")


_NONE = type(None)
_MANIFEST_SCHEMA = {"schema_version": int, "created_at": str, "include_data": bool, "files": dict, "dataset": dict}
_DATASET_SCHEMA = {"meta": dict, "original_roundtrip_hash": str, "original_shape": list, "columns": list}
_STATE_SCHEMA = {"history": list, "rules": list, "completeness_exclude": list, "target": (str, _NONE),
                 "train_config": (dict, _NONE), "training_results": (dict, _NONE), "clustering": (dict, _NONE),
                 "charts": list, "chat": list, "narrative": (dict, _NONE), "report_config": (dict, _NONE),
                 "domain_override": (str, _NONE)}


def _manifest(files: dict[str, bytes]) -> dict[str, Any]:
    m = _json(files[MANIFEST], MANIFEST)
    _check(m, _MANIFEST_SCHEMA, "manifest.json")
    if m["schema_version"] not in SUPPORTED_SCHEMAS:
        raise SessionError(f"Unsupported session schema version {m['schema_version']} (this app reads {sorted(SUPPORTED_SCHEMAS)}).")
    _check(m["dataset"], _DATASET_SCHEMA, "manifest.json dataset")
    listed = set(m["files"])
    present = set(files) - {MANIFEST}
    if listed != present:
        raise SessionError("The package contents do not match its manifest.")
    for name in present:
        if not isinstance(m["files"][name], str) or _sha(files[name]) != m["files"][name]:
            raise SessionError(f"Checksum mismatch for {name}: the package was modified or corrupted.")
    if m["include_data"] != (DATA in files):
        raise SessionError("The manifest's include_data flag does not match the package contents.")
    return m


def inspect_package(data: bytes) -> dict[str, Any]:
    """Validated manifest summary for previewing a package before restoring it."""
    files = _open(data)
    m = _manifest(files)
    ds = m["dataset"]
    meta = ds["meta"]
    return {
        "created_at": m["created_at"], "schema_version": m["schema_version"], "app": m.get("app"),
        "include_data": m["include_data"], "dataset_name": meta.get("name"),
        "provenance": meta.get("provenance_label") or meta.get("provenance_category"),
        "original_shape": ds["original_shape"], "current_shape": ds.get("current_shape"),
        "columns": ds["columns"], "contents": m.get("contents", {}),
    }


# ---------------------------------------------------------------- import
def _valid_history(history: list[Any]) -> list[dict[str, Any]]:
    for i, h in enumerate(history, 1):
        if not isinstance(h, dict) or not isinstance(h.get("operation"), str) or not isinstance(h.get("params"), dict):
            raise SessionError(f"Cleaning step {i} in state.json is malformed.")
        if h["operation"] not in cleaning.OPERATIONS:
            raise SessionError(f"Cleaning step {i} uses an unknown operation '{h['operation']}'.")
    return history


def import_session(data: bytes, dataset_bytes: bytes | None = None, dataset_name: str | None = None) -> dict[str, Any]:
    files = _open(data)
    m = _manifest(files)
    st = _json(files[STATE], STATE)
    _check(st, _STATE_SCHEMA, "state.json")
    ds = m["dataset"]
    report: dict[str, list[str]] = {"restored": [], "not_restored": [], "warnings": []}

    # dataset
    if m["include_data"]:
        raw_bytes, fname = files[DATA], "original.csv"
    else:
        if not dataset_bytes:
            raise SessionError("This package does not contain the dataset. Upload the original data file to restore it.")
        raw_bytes, fname = dataset_bytes, dataset_name or "dataset.csv"
    try:
        raw_df = load_upload(fname, raw_bytes)[0]
    except DataLoadError as exc:
        raise SessionError(f"The dataset could not be read: {exc}") from exc
    rt_hash = frame_hash(raw_df)
    if rt_hash != ds["original_roundtrip_hash"]:
        if m["include_data"]:
            raise SessionError("The packaged dataset does not match the checksum recorded when the session was saved.")
        raise SessionError("The uploaded dataset is not the one this session was saved with (content checksum differs). "
                           "Upload the exact original file.")
    report["restored"].append(f"Dataset '{ds['meta'].get('name')}' ({raw_df.shape[0]:,} rows × {raw_df.shape[1]} columns), checksum verified.")

    try:
        meta = DatasetMeta.from_dict(ds["meta"])
    except TypeError as exc:
        raise SessionError("Dataset metadata in the manifest is malformed.") from exc
    meta.notes = list(meta.notes or []) + [f"Restored from session package created {m['created_at']}."]

    # cleaning history
    history = _valid_history(st["history"])
    df = raw_df
    if history:
        try:
            df = cleaning.replay(raw_df, history)
            if ds.get("current_roundtrip_hash") and frame_hash(df) != ds["current_roundtrip_hash"]:
                report["warnings"].append("Replaying the cleaning history produced data that differs from the saved session "
                                          "(for example after a library upgrade). Review the cleaned dataset.")
            report["restored"].append(f"{len(history)} cleaning step(s) replayed on the original data.")
        except Exception as exc:  # noqa: BLE001
            df = raw_df
            report["warnings"].append(f"The cleaning history could not be replayed ({exc}); the working data was reset to the original. "
                                      "The recorded steps are kept for reference.")

    cols = set(df.columns)
    target = st["target"] if st["target"] in cols else None
    if st["target"] and target is None:
        report["warnings"].append(f"Target '{st['target']}' is not in the restored data and was cleared.")
    elif target:
        report["restored"].append(f"Target: {target}.")

    rules = [r for r in st["rules"] if isinstance(r, dict)]
    if rules:
        report["restored"].append(f"{len(rules)} quality rule(s).")

    if st["train_config"] or st["training_results"]:
        report["not_restored"].append(RETRAIN_NOTE)
        if st["train_config"]:
            report["restored"].append("Model training configuration (retrain to reproduce results).")
    if st.get("importance_computed"):
        report["not_restored"].append(IMPORTANCE_NOTE)

    # clustering
    clustering = st["clustering"]
    if clustering is not None:
        res = clustering.get("result") if isinstance(clustering, dict) else None
        idx = res.get("index") if isinstance(res, dict) else None
        if not isinstance(idx, list) or not set(idx) <= set(df.index.tolist()):
            report["warnings"].append("The saved clustering result no longer matches the restored rows and was dropped.")
            clustering = None
        else:
            clustering = {"result": res, "names": clustering.get("names") or {}, "version": None}
            report["restored"].append(f"Clustering result ({res.get('n_clusters')} clusters).")

    # charts
    from core.charts import ChartSpec, ChartSpecError, validate

    charts, dropped = [], []
    for c in st["charts"]:
        try:
            spec = ChartSpec.from_dict(c["spec"])
            problems = validate(df, spec)
        except (ChartSpecError, KeyError, TypeError) as exc:
            problems = [str(exc)]
        if problems:
            dropped.append(f"{c.get('title', 'chart') if isinstance(c, dict) else 'chart'}: {problems[0]}")
        else:
            charts.append(c)
    if charts:
        report["restored"].append(f"{len(charts)} saved chart(s).")
    for d in dropped:
        report["not_restored"].append(f"Chart dropped — {d}")

    chat = [x for x in st["chat"] if isinstance(x, dict)]
    if chat:
        report["restored"].append(f"{len(chat)} chat message(s) (text only).")
    if st["report_config"]:
        report["restored"].append("Report configuration.")

    state = {
        "raw_df": raw_df, "df": df, "meta": meta, "history": history, "rules": rules,
        "completeness_exclude": [c for c in st["completeness_exclude"] if c in cols],
        "target": target, "train_config": st["train_config"], "training": None,
        "previous_training_results": st["training_results"], "importance": None,
        "clustering": clustering, "charts": charts, "chat": chat, "narrative": st["narrative"],
        "report_config": st["report_config"], "domain_override": st["domain_override"],
    }
    return {"state": state, "report": report, "manifest": m}


# --------------------------------------------------------------- compare
def compare_configs(saved_state: dict[str, Any], current_snap: dict[str, Any]) -> list[dict[str, Any]]:
    """Differences between a saved (imported) state and the current session, for display."""
    diffs: list[dict[str, Any]] = []

    def add(item, saved, current):
        if saved != current:
            diffs.append({"item": item, "saved": saved, "current": current})

    def ops(h):
        return [x.get("operation") for x in (h or [])]

    add("Target", saved_state.get("target"), current_snap.get("target"))
    add("Cleaning steps", len(saved_state.get("history") or []), len(current_snap.get("history") or []))
    if ops(saved_state.get("history")) != ops(current_snap.get("history")):
        add("Cleaning operations", ops(saved_state.get("history")), ops(current_snap.get("history")))
    add("Quality rules", len(saved_state.get("rules") or []), len(current_snap.get("rules") or []))
    sc, cc = saved_state.get("train_config") or {}, current_snap.get("train_config") or {}
    for k in sorted(set(sc) | set(cc)):
        add(f"Training: {k}", sc.get(k), cc.get(k))

    def ccfg(x):
        r = (x or {}).get("result") if isinstance(x, dict) else None
        return (r or {}).get("config")

    add("Clustering configuration", ccfg(saved_state.get("clustering")), ccfg(current_snap.get("clustering")))
    add("Saved charts", len(saved_state.get("charts") or []), len(current_snap.get("charts") or []))
    add("Report configuration", saved_state.get("report_config"), current_snap.get("report_config"))
    return diffs
