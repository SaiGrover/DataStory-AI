"""
DataStory AI - Results Router
"""

import logging
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.database import get_db, ModelResult

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/{dataset_id}")
async def get_results(dataset_id: str, db: AsyncSession = Depends(get_db)):
    """Return all model results for a dataset."""
    result = await db.execute(
        select(ModelResult)
        .where(ModelResult.dataset_id == dataset_id)
        .order_by(ModelResult.is_best.desc(), ModelResult.test_f1.desc())
    )
    records = result.scalars().all()

    return {
        "dataset_id": dataset_id,
        "results": [
            {
                "model_name": r.model_name,
                "task_type": r.task_type,
                "best_cv_score": r.best_cv_score,
                "best_params": r.best_params,
                "test_accuracy": r.test_accuracy,
                "test_precision": r.test_precision,
                "test_recall": r.test_recall,
                "test_f1": r.test_f1,
                "test_roc_auc": r.test_roc_auc,
                "test_mae": r.test_mae,
                "test_rmse": r.test_rmse,
                "test_r2": r.test_r2,
                "smote_applied": bool(r.smote_applied),
                "class_weight_applied": bool(r.class_weight_applied),
                "is_best": bool(r.is_best),
            }
            for r in records
        ],
    }


@router.get("/{dataset_id}/best")
async def get_best_model(dataset_id: str, db: AsyncSession = Depends(get_db)):
    """Return the best model result for a dataset."""
    result = await db.execute(
        select(ModelResult)
        .where(ModelResult.dataset_id == dataset_id)
        .where(ModelResult.is_best == 1)
    )
    record = result.scalar_one_or_none()
    if not record:
        return {"message": "No best model found. Please run training first."}
    return {
        "model_name": record.model_name,
        "best_params": record.best_params,
        "test_f1": record.test_f1,
        "test_accuracy": record.test_accuracy,
        "test_rmse": record.test_rmse,
        "test_r2": record.test_r2,
        "is_best": True,
    }
