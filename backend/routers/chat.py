"""
DataStory AI - Chat Router
RAG-powered dataset Q&A using OpenRouter.
"""

import uuid
import logging
from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.database import get_db, DatasetRecord, ModelResult, ChatMessage
from backend.routers.upload import load_dataset_df
from backend.services.profiler import profile_dataset
from backend.services.chat import chat_with_dataset as answer_with_dataset
from backend.schemas.report import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/{dataset_id}", response_model=ChatResponse)
async def chat_with_dataset(
    dataset_id: str,
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """Answer user questions about the dataset using RAG + OpenRouter."""
    # Load dataset context
    df = load_dataset_df(dataset_id)
    profile = profile_dataset(df)

    ds_result = await db.execute(
        select(DatasetRecord).where(DatasetRecord.id == dataset_id)
    )
    ds = ds_result.scalar_one_or_none()

    model_results = await db.execute(
        select(ModelResult).where(ModelResult.dataset_id == dataset_id)
    )
    models = model_results.scalars().all()
    best_model = next((m for m in models if m.is_best), None)

    result_dicts = []
    for model in models:
        result_dicts.append({
            "model_name": model.model_name,
            "accuracy": model.test_accuracy,
            "precision": model.test_precision,
            "recall": model.test_recall,
            "f1": model.test_f1,
            "roc_auc": model.test_roc_auc,
            "mae": model.test_mae,
            "rmse": model.test_rmse,
            "r2": model.test_r2,
            "best_params": model.best_params,
        })

    answer = answer_with_dataset(
        question=request.question,
        df=df,
        profile=profile,
        results=result_dicts,
        best_model=best_model.model_name if best_model else None,
        task=ds.task_type if ds else None,
        target=ds.selected_target if ds else None,
        cleaning_actions=ds.cleaning_actions if ds else [],
    )

    # Save messages to DB
    user_msg = ChatMessage(
        id=str(uuid.uuid4()),
        dataset_id=dataset_id,
        role="user",
        content=request.question,
        created_at=datetime.utcnow(),
    )
    assistant_msg = ChatMessage(
        id=str(uuid.uuid4()),
        dataset_id=dataset_id,
        role="assistant",
        content=answer,
        created_at=datetime.utcnow(),
    )
    db.add(user_msg)
    db.add(assistant_msg)
    await db.commit()

    return ChatResponse(
        dataset_id=dataset_id,
        question=request.question,
        answer=answer,
        sources=[],
    )


@router.get("/{dataset_id}/history")
async def get_chat_history(dataset_id: str, db: AsyncSession = Depends(get_db)):
    """Return full chat history for a dataset."""
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.dataset_id == dataset_id)
        .order_by(ChatMessage.created_at.asc())
    )
    messages = result.scalars().all()
    return {
        "dataset_id": dataset_id,
        "messages": [
            {"role": m.role, "content": m.content, "created_at": str(m.created_at)}
            for m in messages
        ],
    }
