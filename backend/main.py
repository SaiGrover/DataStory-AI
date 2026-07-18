"""
DataStory AI — FastAPI Backend
Provides REST endpoints for dataset processing, ML training, and AI generation.
"""
import os
import uuid
import logging
from threading import Lock
from typing import Optional, List

# Keep numerical libraries inside the memory/CPU envelope of small hosted instances.
# These must be set before NumPy and scikit-learn are imported.
_numeric_threads = os.getenv("DATASTORY_NUMERIC_THREADS", "1")
for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable] = _numeric_threads

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
import pandas as pd
import numpy as np
import io

from backend.services.profiler import profile_dataset
from backend.services.cleaner import apply_cleaning, preview_cleaning
from backend.services.task_detector import detect_task_type, detect_imbalance
from backend.services.trainer import train_models
from backend.services.trainer import get_recommended_models
from backend.services.report_generator import generate_report, report_to_markdown
from backend.services.chat import chat_with_dataset
from backend.schemas.requests import (
    CleanRequest, TrainRequest, ChatRequest, ReportRequest
)
from backend.utils.db import init_db, save_dataset_meta, get_dataset_meta, save_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("datastory")

app = FastAPI(
    title="DataStory AI API",
    description="AI-powered data analysis and ML pipeline backend",
    version="1.0.0",
)

def _cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS") or os.getenv("FRONTEND_ORIGINS") or "*"
    if raw.strip() == "*":
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


_allowed_origins = _cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=_allowed_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory dataset store (keyed by dataset_id)
_dataset_store: dict = {}
_results_store: dict = {}
_training_lock = Lock()


def _store_active_dataset(dataset_id: str, df: pd.DataFrame) -> None:
    """Keep only the active dataset on the single-instance demo backend."""
    _dataset_store.clear()
    _results_store.clear()
    _dataset_store[dataset_id] = df

SAMPLE_DATASETS = {
    "titanic": {"name": "Titanic", "path": "data/sample_datasets/titanic.csv", "description": "Passenger survival data"},
    "student": {"name": "Student Performance", "path": "data/sample_datasets/student_performance.csv", "description": "Student exam performance"},
    "churn": {"name": "Customer Churn", "path": "data/sample_datasets/customer_churn.csv", "description": "Telecom customer data"},
    "bike": {"name": "Bike Sharing", "path": "data/sample_datasets/bike_sharing.csv", "description": "Bike sharing demand data"},
    "iris": {"name": "Iris", "path": "data/sample_datasets/iris.csv", "description": "Iris flower dataset"},
    "wine": {"name": "Wine", "path": "data/sample_datasets/wine.csv", "description": "Classic wine classification"},
    "breast_cancer": {"name": "Breast Cancer", "path": "data/sample_datasets/breast_cancer.csv", "description": "Diagnostic classification"},
    "diabetes": {"name": "Diabetes", "path": "data/sample_datasets/diabetes.csv", "description": "Classic regression dataset"},
    "digits": {"name": "Digits", "path": "data/sample_datasets/digits.csv", "description": "Handwritten digit classification"},
    "linnerud": {"name": "Linnerud", "path": "data/sample_datasets/linnerud.csv", "description": "Classic fitness regression"},
}


def _density_curve(series: pd.Series) -> dict:
    """Small dependency-free Gaussian KDE approximation for frontend plots."""
    values = series.dropna().astype(float)
    if len(values) < 3:
        return {"x": [], "y": []}
    std = float(values.std(ddof=1) or 0)
    if std == 0:
        return {"x": [round(float(values.iloc[0]), 4)], "y": [1.0]}
    x_min, x_max = float(values.min()), float(values.max())
    xs = np.linspace(x_min, x_max, 80)
    bandwidth = max(1.06 * std * (len(values) ** (-1 / 5)), std * 0.05)
    diffs = (xs[:, None] - values.to_numpy()[None, :]) / bandwidth
    ys = np.exp(-0.5 * diffs**2).sum(axis=1) / (len(values) * bandwidth * np.sqrt(2 * np.pi))
    return {
        "x": [round(float(x), 4) for x in xs.tolist()],
        "y": [round(float(y), 6) for y in ys.tolist()],
    }

@app.on_event("startup")
def startup():
    init_db()
    logger.info("DataStory AI API started.")


@app.get("/")
def root():
    return {"message": "DataStory AI API is running.", "version": "1.0.0"}


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/samples")
def list_samples():
    samples = []
    for key, meta in SAMPLE_DATASETS.items():
        path = meta["path"]
        if os.path.exists(path):
            df = pd.read_csv(path)
            rows, columns = df.shape
        else:
            rows, columns = 0, 0
        samples.append({
            "id": key,
            "name": meta["name"],
            "description": meta["description"],
            "rows": rows,
            "columns": columns,
        })
    return {"samples": samples}


@app.post("/samples/{sample_id}")
def load_sample(sample_id: str):
    meta = SAMPLE_DATASETS.get(sample_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Sample dataset not found.")
    if not os.path.exists(meta["path"]):
        raise HTTPException(status_code=404, detail=f"Sample file missing: {meta['path']}")
    df = pd.read_csv(meta["path"])
    dataset_id = str(uuid.uuid4())[:8]
    _store_active_dataset(dataset_id, df)
    profile = profile_dataset(df)
    save_dataset_meta(dataset_id, f"{sample_id}.csv", df.shape[0], df.shape[1])
    return {
        "dataset_id": dataset_id,
        "filename": f"{sample_id}.csv",
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist(),
        "profile": profile,
    }


@app.get("/models")
def list_models(task_type: str = "classification"):
    if task_type not in {"classification", "regression"}:
        raise HTTPException(status_code=400, detail="task_type must be classification or regression.")
    return {"task_type": task_type, "models": get_recommended_models(task_type)}


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV file, validate, and profile it."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {e}")

    dataset_id = str(uuid.uuid4())[:8]
    _store_active_dataset(dataset_id, df)

    profile = profile_dataset(df)
    save_dataset_meta(dataset_id, file.filename, df.shape[0], df.shape[1])

    logger.info(f"Uploaded: {file.filename} [{dataset_id}] — {df.shape}")
    return {
        "dataset_id": dataset_id,
        "filename": file.filename,
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist(),
        "profile": profile,
    }


@app.get("/profile/{dataset_id}")
def get_profile(dataset_id: str):
    df = _get_df(dataset_id)
    return profile_dataset(df)


@app.post("/clean/{dataset_id}")
def clean_dataset(dataset_id: str, body: CleanRequest):
    df = _get_df(dataset_id)
    if body.preview_only:
        return preview_cleaning(df, body.config)
    cleaned = apply_cleaning(df, body.config)
    # Replace the original frame instead of retaining two full in-memory copies.
    _dataset_store[dataset_id] = cleaned
    _dataset_store.pop(f"{dataset_id}_cleaned", None)
    meta = get_dataset_meta(dataset_id) or {}
    save_dataset_meta(dataset_id, meta.get("filename", f"{dataset_id}.csv"), cleaned.shape[0], cleaned.shape[1])
    profile = profile_dataset(cleaned)
    return {
        "dataset_id": dataset_id,
        "filename": meta.get("filename", f"{dataset_id}.csv"),
        "rows": cleaned.shape[0],
        "columns": cleaned.shape[1],
        "column_names": cleaned.columns.tolist(),
        "profile": profile,
        "missing": int(cleaned.isnull().sum().sum()),
        "duplicates": int(cleaned.duplicated().sum()),
    }


@app.get("/eda/{dataset_id}")
def get_eda(dataset_id: str):
    df = _get_df(dataset_id, prefer_cleaned=True)
    profile = profile_dataset(df)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    id_cols = profile.get("id_cols", [])
    analysis_num_cols = [col for col in num_cols if col not in id_cols]
    analysis_cat_cols = [col for col in cat_cols if col not in id_cols]
    numeric_profiles = []
    for col in analysis_num_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        counts, bins = np.histogram(series, bins=min(12, max(4, int(np.sqrt(len(series))))))
        q1, median, q3 = series.quantile([0.25, 0.5, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        numeric_profiles.append({
            "column": col,
            "histogram": {"bins": [round(float(x), 4) for x in bins.tolist()], "counts": counts.astype(int).tolist()},
            "values": [round(float(x), 4) for x in series.sample(min(len(series), 500), random_state=42).tolist()],
            "kde": _density_curve(series),
            "box": {
                "min": round(float(series.min()), 4),
                "q1": round(float(q1), 4),
                "median": round(float(median), 4),
                "q3": round(float(q3), 4),
                "max": round(float(series.max()), 4),
                "outliers": int(((series < lower) | (series > upper)).sum()),
            },
            "stats": {
                "mean": round(float(series.mean()), 4),
                "std": round(float(series.std()), 4) if len(series) > 1 else 0.0,
                "skew": round(float(series.skew()), 4) if len(series) > 2 else 0.0,
                "missing": int(df[col].isna().sum()),
            },
        })
    categorical_profiles = []
    for col in analysis_cat_cols:
        counts = df[col].fillna("(missing)").astype(str).value_counts().head(8)
        categorical_profiles.append({
            "column": col,
            "labels": counts.index.tolist(),
            "counts": counts.astype(int).tolist(),
        })
    if analysis_num_cols:
        corr_df = (
            df[analysis_num_cols]
            .corr(numeric_only=True)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )
        correlation = {
            "columns": corr_df.columns.tolist(),
            "values": [[round(float(value), 4) for value in row] for row in corr_df.values.tolist()],
        }
    else:
        correlation = {"columns": [], "values": []}
    return {
        "shape": list(df.shape),
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "id_columns": id_cols,
        "missing_per_column": df.isnull().sum().to_dict(),
        "unique_per_column": df.nunique().to_dict(),
        "summary": {},
        "numeric_profiles": numeric_profiles,
        "categorical_profiles": categorical_profiles,
        "correlation": correlation,
    }


@app.post("/select-target/{dataset_id}")
def select_target(dataset_id: str, body: dict):
    target = body.get("target")
    df = _get_df(dataset_id, prefer_cleaned=True)
    if target not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{target}' not found.")
    task = detect_task_type(df, target)
    imbalance = detect_imbalance(df, target) if task == "classification" else None
    return {"target": target, "task_type": task, "imbalance_info": imbalance}


@app.post("/train/{dataset_id}")
def train(dataset_id: str, body: TrainRequest):
    df = _get_df(dataset_id, prefer_cleaned=True)
    if not _training_lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Another model training job is already running. Please wait for it to finish.")
    try:
        results = train_models(
            df=df,
            target=body.target,
            task_type=body.task_type,
            model_names=body.model_names,
            imbalance_strategy=body.imbalance_strategy,
            cv_folds=body.cv_folds,
            test_size=body.test_size,
        )
    finally:
        _training_lock.release()
    _results_store[dataset_id] = results
    save_results(dataset_id, results)
    return {"results": results}


@app.get("/results/{dataset_id}")
def get_results(dataset_id: str):
    results = _results_store.get(dataset_id)
    if not results:
        raise HTTPException(status_code=404, detail="No results found. Train models first.")
    return {"results": results}


@app.get("/best-model/{dataset_id}")
def get_best_model(dataset_id: str):
    results = _results_store.get(dataset_id)
    if not results:
        raise HTTPException(status_code=404, detail="No results found.")
    valid = [r for r in results if "error" not in r]
    if not valid:
        raise HTTPException(status_code=400, detail="All models failed training.")
    best = max(valid, key=lambda r: r.get("primary_score", -999))
    return best


@app.post("/chat/{dataset_id}")
def chat(dataset_id: str, body: ChatRequest):
    df = _get_df(dataset_id, prefer_cleaned=True)
    results = _results_store.get(dataset_id)
    best = max(results, key=lambda r: r.get("primary_score", -999)) if results else None
    response = chat_with_dataset(
        question=body.question,
        df=df,
        profile=profile_dataset(df),
        results=results,
        best_model=best.get("model_name") if best else None,
        task=body.task_type,
        target=body.target,
        cleaning_actions=body.cleaning_actions or [],
        history=body.history or [],
    )
    return response


@app.post("/report/{dataset_id}")
def generate_report_endpoint(dataset_id: str, body: ReportRequest):
    df = _get_df(dataset_id, prefer_cleaned=True)
    results = _results_store.get(dataset_id)
    best = max(results, key=lambda r: r.get("primary_score", -999)) if results else None
    report = generate_report(
        df=df,
        profile=profile_dataset(df),
        cleaning_actions=body.cleaning_actions or [],
        results=results,
        best_model=best.get("model_name") if best else None,
        task=body.task_type,
        target=body.target,
        style=body.style,
    )
    return {"report": report}


@app.get("/download-report/{dataset_id}")
def download_report(dataset_id: str, style: str = "Beginner Friendly"):
    df = _get_df(dataset_id, prefer_cleaned=True)
    results = _results_store.get(dataset_id)
    best = max(results, key=lambda r: r.get("primary_score", -999)) if results else None
    report = generate_report(
        df=df,
        profile=profile_dataset(df),
        cleaning_actions=[],
        results=results,
        best_model=best.get("model_name") if best else None,
        task=None,
        target=None,
        style=style,
    )
    md = report_to_markdown(report)
    return StreamingResponse(
        io.StringIO(md),
        media_type="text/markdown",
        headers={"Content-Disposition": "attachment; filename=datastory_report.md"},
    )


def _get_df(dataset_id: str, prefer_cleaned: bool = False) -> pd.DataFrame:
    if prefer_cleaned and f"{dataset_id}_cleaned" in _dataset_store:
        return _dataset_store[f"{dataset_id}_cleaned"]
    df = _dataset_store.get(dataset_id)
    if df is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    return df
