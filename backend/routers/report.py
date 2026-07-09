"""
DataStory AI - Report Router
Generates, retrieves, and downloads AI-powered reports.
"""

import uuid
import logging
import os
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from backend.database import get_db, DatasetRecord, ModelResult, ReportRecord
from backend.routers.upload import load_dataset_df
from backend.services.profiler import profile_dataset
from backend.services.ai_explainer import generate_full_report
from backend.schemas.report import ReportRequest, ReportResponse

logger = logging.getLogger(__name__)
router = APIRouter()

REPORTS_DIR = os.getenv("REPORTS_DIR", "./reports")


@router.post("/{dataset_id}", response_model=ReportResponse)
async def generate_report(
    dataset_id: str,
    request: ReportRequest,
    db: AsyncSession = Depends(get_db),
):
    """Generate a full AI-powered report for the dataset."""
    df = load_dataset_df(dataset_id)
    profile = profile_dataset(df)

    ds_result = await db.execute(
        select(DatasetRecord).where(DatasetRecord.id == dataset_id)
    )
    ds = ds_result.scalar_one_or_none()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    model_results = await db.execute(
        select(ModelResult).where(ModelResult.dataset_id == dataset_id)
    )
    models = model_results.scalars().all()
    model_list = [
        {
            "model_name": m.model_name,
            "test_f1": m.test_f1,
            "test_accuracy": m.test_accuracy,
            "test_rmse": m.test_rmse,
            "test_r2": m.test_r2,
            "is_best": bool(m.is_best),
            "best_params": m.best_params,
        }
        for m in models
    ]

    best_model = next((m for m in model_list if m["is_best"]), {})

    cleaning_summary = {
        "actions_applied": ds.cleaning_actions or [],
    }

    content_md = await generate_full_report(
        profile=profile,
        cleaning_summary=cleaning_summary,
        task_type=ds.task_type or "unknown",
        target_col=ds.selected_target or "unknown",
        best_model=best_model.get("model_name", "N/A"),
        best_params=best_model.get("best_params", {}),
        results=model_list,
        imbalance_strategy=ds.imbalance_strategy or "none",
        report_style=request.report_style,
    )

    report_id = str(uuid.uuid4())
    os.makedirs(REPORTS_DIR, exist_ok=True)
    md_path = os.path.join(REPORTS_DIR, f"{report_id}.md")
    with open(md_path, "w") as f:
        f.write(content_md)

    record = ReportRecord(
        id=report_id,
        dataset_id=dataset_id,
        report_style=request.report_style,
        content_md=content_md,
        created_at=datetime.utcnow(),
    )
    db.add(record)
    await db.commit()

    return ReportResponse(
        report_id=report_id,
        dataset_id=dataset_id,
        report_style=request.report_style,
        content_md=content_md,
        created_at=datetime.utcnow(),
    )


@router.get("/{dataset_id}/latest")
async def get_latest_report(dataset_id: str, db: AsyncSession = Depends(get_db)):
    """Retrieve the most recently generated report for a dataset."""
    result = await db.execute(
        select(ReportRecord)
        .where(ReportRecord.dataset_id == dataset_id)
        .order_by(ReportRecord.created_at.desc())
        .limit(1)
    )
    record = result.scalar_one_or_none()
    if not record:
        return {"message": "No report found. Please generate one first."}
    return {
        "report_id": record.id,
        "report_style": record.report_style,
        "content_md": record.content_md,
        "created_at": record.created_at,
    }


@router.get("/download/{report_id}")
async def download_report(report_id: str):
    """Download a report as a Markdown file."""
    path = os.path.join(REPORTS_DIR, f"{report_id}.md")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Report file not found.")
    return FileResponse(
        path,
        media_type="text/markdown",
        filename=f"datastory_report_{report_id[:8]}.md",
    )
