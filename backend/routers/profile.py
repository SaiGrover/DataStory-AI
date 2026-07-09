"""
DataStory AI - Profile Router
Generates dataset health check and profiling results.
"""

import logging
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import update

from backend.database import get_db, DatasetRecord
from backend.routers.upload import load_dataset_df
from backend.services.profiler import profile_dataset
from backend.schemas.dataset import DataProfileResult

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/{dataset_id}", response_model=DataProfileResult)
async def get_profile(
    dataset_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Run dataset profiling and return health check results."""
    df = load_dataset_df(dataset_id)
    profile = profile_dataset(df)

    # Update health score in DB
    await db.execute(
        update(DatasetRecord)
        .where(DatasetRecord.id == dataset_id)
        .values(health_score=profile["health_score"], status="profiled")
    )
    await db.commit()

    return DataProfileResult(
        dataset_id=dataset_id,
        **profile,
    )


@router.get("/{dataset_id}/preview")
async def get_preview(dataset_id: str, rows: int = 5):
    """Return the first N rows of a dataset as JSON."""
    df = load_dataset_df(dataset_id)
    return {
        "columns": df.columns.tolist(),
        "rows": df.head(rows).fillna("").to_dict(orient="records"),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }
