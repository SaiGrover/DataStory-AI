"""
DataStory AI - EDA Router
"""

import logging
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_db
from backend.routers.upload import load_dataset_df
from backend.services.eda_service import generate_eda

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/{dataset_id}")
async def get_eda(dataset_id: str, target_col: str = None):
    """Generate full EDA results for a dataset."""
    df = load_dataset_df(dataset_id)
    results = generate_eda(df, target_col=target_col)
    return {"dataset_id": dataset_id, "eda": results}
