"""
DataStory AI - Clean Router
Handles data cleaning preview and application.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import update

from backend.database import get_db, DatasetRecord
from backend.routers.upload import load_dataset_df, save_dataset_df
from backend.services.cleaner import clean_dataset, preview_cleaning, AVAILABLE_ACTIONS
from backend.schemas.dataset import CleaningRequest, CleaningResult

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/actions")
async def get_available_actions():
    """Return the list of available cleaning actions."""
    return {"actions": AVAILABLE_ACTIONS}


@router.post("/preview/{dataset_id}")
async def preview_clean(dataset_id: str, request: CleaningRequest):
    """Preview what cleaning would do, without applying changes."""
    df = load_dataset_df(dataset_id)
    result = preview_cleaning(
        df,
        request.actions,
        request.custom_fill_value,
        request.drop_missing_threshold,
    )
    return result


@router.post("/{dataset_id}", response_model=CleaningResult)
async def apply_cleaning(
    dataset_id: str,
    request: CleaningRequest,
    db: AsyncSession = Depends(get_db),
):
    """Apply selected cleaning actions and save the cleaned dataset."""
    df = load_dataset_df(dataset_id)
    cleaned_df, applied, summary = clean_dataset(
        df,
        request.actions,
        request.custom_fill_value,
        request.drop_missing_threshold,
    )

    save_dataset_df(dataset_id, cleaned_df)

    await db.execute(
        update(DatasetRecord)
        .where(DatasetRecord.id == dataset_id)
        .values(
            cleaning_actions=applied,
            rows=int(cleaned_df.shape[0]),
            columns=int(cleaned_df.shape[1]),
            status="cleaned",
        )
    )
    await db.commit()

    return CleaningResult(
        dataset_id=dataset_id,
        actions_applied=applied,
        before=summary["before"],
        after=summary["after"],
        summary=f"Applied {len(applied)} cleaning action(s). Dataset is now ready for EDA.",
    )


@router.post("/reset/{dataset_id}")
async def reset_cleaning(dataset_id: str):
    """
    Note: True reset would require storing the original file separately.
    This endpoint signals the frontend that reset was requested.
    """
    return {
        "message": (
            "To reset cleaning, please re-upload your original CSV file. "
            "DataStory AI does not store the original file after cleaning is applied."
        )
    }
