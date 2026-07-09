"""
DataStory AI - Upload Router
Handles CSV file uploads and sample dataset loading.
"""

import os
import uuid
import logging
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import insert
from datetime import datetime

from backend.database import get_db, DatasetRecord
from backend.schemas.dataset import DatasetMeta

logger = logging.getLogger(__name__)
router = APIRouter()

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./data/uploads")
SAMPLES_DIR = "./data/samples"

SAMPLE_DATASETS = {
    "titanic": "titanic.csv",
    "churn": "customer_churn.csv",
    "student": "student_performance.csv",
    "bike": "bike_sharing.csv",
    "iris": "iris.csv",
}


@router.post("", response_model=DatasetMeta)
async def upload_csv(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Upload a CSV file and register it in the database."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    contents = await file.read()
    file_size_kb = round(len(contents) / 1024, 2)

    # Validate CSV
    try:
        import io
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=400, detail="The uploaded CSV file is empty.")

    dataset_id = str(uuid.uuid4())
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    save_path = os.path.join(UPLOAD_DIR, f"{dataset_id}.csv")

    with open(save_path, "wb") as f:
        f.write(contents)

    # Save to DB
    record = DatasetRecord(
        id=dataset_id,
        file_name=file.filename,
        upload_timestamp=datetime.utcnow(),
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
        column_names=df.columns.tolist(),
        file_size_kb=file_size_kb,
        status="uploaded",
    )
    db.add(record)
    await db.commit()

    logger.info(f"Uploaded dataset {dataset_id}: {file.filename} ({df.shape[0]}×{df.shape[1]})")

    return DatasetMeta(
        dataset_id=dataset_id,
        file_name=file.filename,
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
        column_names=df.columns.tolist(),
        file_size_kb=file_size_kb,
        upload_timestamp=datetime.utcnow(),
    )


@router.post("/sample/{sample_name}", response_model=DatasetMeta)
async def load_sample_dataset(
    sample_name: str,
    db: AsyncSession = Depends(get_db),
):
    """Load a bundled sample dataset."""
    if sample_name not in SAMPLE_DATASETS:
        raise HTTPException(
            status_code=404,
            detail=f"Sample dataset '{sample_name}' not found. Available: {list(SAMPLE_DATASETS.keys())}",
        )

    file_name = SAMPLE_DATASETS[sample_name]
    file_path = os.path.join(SAMPLES_DIR, file_name)

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail=f"Sample file '{file_name}' not found on server. Please add it to data/samples/.",
        )

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read sample file: {str(e)}")

    dataset_id = str(uuid.uuid4())
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    save_path = os.path.join(UPLOAD_DIR, f"{dataset_id}.csv")
    df.to_csv(save_path, index=False)

    file_size_kb = round(os.path.getsize(save_path) / 1024, 2)

    record = DatasetRecord(
        id=dataset_id,
        file_name=file_name,
        upload_timestamp=datetime.utcnow(),
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
        column_names=df.columns.tolist(),
        file_size_kb=file_size_kb,
        status="uploaded",
    )
    db.add(record)
    await db.commit()

    return DatasetMeta(
        dataset_id=dataset_id,
        file_name=file_name,
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
        column_names=df.columns.tolist(),
        file_size_kb=file_size_kb,
        upload_timestamp=datetime.utcnow(),
    )


def load_dataset_df(dataset_id: str) -> pd.DataFrame:
    """Load a dataset DataFrame by ID. Raises 404 if not found."""
    path = os.path.join(UPLOAD_DIR, f"{dataset_id}.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    return pd.read_csv(path)


def save_dataset_df(dataset_id: str, df: pd.DataFrame):
    """Overwrite a stored dataset with a cleaned version."""
    path = os.path.join(UPLOAD_DIR, f"{dataset_id}.csv")
    df.to_csv(path, index=False)
