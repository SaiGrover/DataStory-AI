"""
DataStory AI - Train Router
Handles model training, task detection, and imbalance checking.
"""

import uuid
import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import update, insert

from backend.database import get_db, DatasetRecord, ModelResult
from backend.routers.upload import load_dataset_df
from backend.schemas.training import (
    TrainingRequest, TrainingResponse, ModelSelectionRequest,
    ImbalanceHandlingRequest,
)
from backend.schemas.dataset import TargetSelectionRequest, TaskDetectionResult
from backend.services.trainer import (
    train_all_models, detect_task_type, get_recommended_models
)
from backend.services.imbalance_handler import detect_imbalance
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/detect-task/{dataset_id}", response_model=TaskDetectionResult)
async def detect_task(
    dataset_id: str,
    request: TargetSelectionRequest,
    db: AsyncSession = Depends(get_db),
):
    """Detect whether the task is classification or regression."""
    df = load_dataset_df(dataset_id)

    if request.target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{request.target_column}' not found.")

    target = df[request.target_column]

    if target.isnull().sum() > 0:
        raise HTTPException(
            status_code=400,
            detail="Target column has missing values. Please clean the dataset first.",
        )

    task_type, warnings = detect_task_type(target)
    n_classes = int(target.nunique()) if task_type == "classification" else None

    if task_type == "classification":
        counts = target.value_counts()
        class_distribution = {str(k): int(v) for k, v in counts.items()}
        target_summary = None
    else:
        class_distribution = None
        target_summary = {
            "mean": round(float(target.mean()), 4),
            "std": round(float(target.std()), 4),
            "min": round(float(target.min()), 4),
            "max": round(float(target.max()), 4),
        }

    await db.execute(
        update(DatasetRecord)
        .where(DatasetRecord.id == dataset_id)
        .values(
            selected_target=request.target_column,
            task_type=task_type,
            status="target_selected",
        )
    )
    await db.commit()

    return TaskDetectionResult(
        dataset_id=dataset_id,
        target_column=request.target_column,
        task_type=task_type,
        num_classes=n_classes,
        class_distribution=class_distribution,
        target_summary=target_summary,
        warnings=warnings,
    )


@router.post("/recommend-models/{dataset_id}")
async def recommend_models(dataset_id: str, task_type: str):
    """Return recommended models for the detected task type."""
    models = get_recommended_models(task_type)
    descriptions = {
        "Logistic Regression": "A simple, fast baseline for classification. Works well for linearly separable data.",
        "Decision Tree Classifier": "Intuitive tree-based model. Easy to interpret and visualize.",
        "Random Forest Classifier": "Ensemble of many trees. Handles non-linear patterns well.",
        "Support Vector Machine": "Effective in high-dimensional spaces. Good for smaller datasets.",
        "K-Nearest Neighbors": "Classifies based on similar nearby examples. Simple but effective.",
        "Naive Bayes": "Fast probabilistic model. Works well for text and simple problems.",
        "Linear Regression": "Simple baseline for regression. Assumes a linear relationship.",
        "Ridge Regression": "Linear regression with regularization. Helps prevent overfitting.",
        "Decision Tree Regressor": "Tree-based model for regression. Easy to understand.",
        "Random Forest Regressor": "Ensemble model for regression. Reduces overfitting vs single trees.",
        "Gradient Boosting Regressor": "Powerful boosting model. Often gives best results on tabular data.",
    }
    return {
        "task_type": task_type,
        "recommended_models": [
            {"name": m, "description": descriptions.get(m, "")} for m in models
        ],
    }


@router.post("/check-imbalance/{dataset_id}")
async def check_imbalance(dataset_id: str, target_col: str):
    """Check class imbalance for a classification target."""
    df = load_dataset_df(dataset_id)
    if target_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{target_col}' not found.")
    imbalance = detect_imbalance(df[target_col])
    return {"dataset_id": dataset_id, "target_column": target_col, **imbalance}


@router.post("/{dataset_id}", response_model=TrainingResponse)
async def train_models(
    dataset_id: str,
    request: TrainingRequest,
    db: AsyncSession = Depends(get_db),
):
    """Run GridSearchCV training for all selected models."""
    df = load_dataset_df(dataset_id)

    if request.target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{request.target_column}' not found.")

    if not request.selected_models:
        raise HTTPException(status_code=400, detail="Please select at least one model.")

    logger.info(
        f"Training {len(request.selected_models)} models on dataset {dataset_id}. "
        f"Task: {request.target_column}, Strategy: {request.imbalance_strategy}"
    )

    training_result = train_all_models(
        df=df,
        target_col=request.target_column,
        selected_models=request.selected_models,
        task_type=detect_task_type(df[request.target_column])[0],
        imbalance_strategy=request.imbalance_strategy,
        cv_folds=request.cv_folds,
        test_size=request.test_size,
        random_state=request.random_state,
        dataset_id=dataset_id,
    )

    # Save all model results to DB
    for res in training_result["results"]:
        result_id = str(uuid.uuid4())
        model_record = ModelResult(
            id=result_id,
            dataset_id=dataset_id,
            model_name=res["model_name"],
            task_type=res["task_type"],
            best_cv_score=res.get("best_cv_score", 0),
            best_params=res.get("best_params", {}),
            test_accuracy=res.get("test_accuracy"),
            test_precision=res.get("test_precision"),
            test_recall=res.get("test_recall"),
            test_f1=res.get("test_f1"),
            test_roc_auc=res.get("test_roc_auc"),
            test_mae=res.get("test_mae"),
            test_rmse=res.get("test_rmse"),
            test_r2=res.get("test_r2"),
            smote_applied=int(res.get("smote_applied", False)),
            class_weight_applied=int(res.get("class_weight_applied", False)),
            is_best=int(res.get("is_best", False)),
            trained_at=datetime.utcnow(),
        )
        db.add(model_record)

    await db.execute(
        update(DatasetRecord)
        .where(DatasetRecord.id == dataset_id)
        .values(
            imbalance_strategy=request.imbalance_strategy,
            status="trained",
        )
    )
    await db.commit()

    return TrainingResponse(
        dataset_id=dataset_id,
        task_type=training_result["results"][0]["task_type"] if training_result["results"] else "unknown",
        models_trained=len(training_result["results"]),
        results=training_result["results"],
        best_model=training_result["best_model"],
        best_model_params=training_result["best_model_params"],
        best_score=training_result["best_score"],
        selection_reason=training_result["selection_reason"],
        imbalance_strategy=training_result["imbalance_strategy"],
    )
