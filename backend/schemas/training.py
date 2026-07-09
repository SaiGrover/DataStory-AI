"""
DataStory AI - Training Schemas
"""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class ModelSelectionRequest(BaseModel):
    dataset_id: str
    use_recommended: bool = True
    selected_models: Optional[List[str]] = None


class ImbalanceHandlingRequest(BaseModel):
    dataset_id: str
    strategy: str  # "smote" | "class_weight" | "none"


class TrainingRequest(BaseModel):
    dataset_id: str
    target_column: str
    selected_models: List[str]
    imbalance_strategy: str = "none"
    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42


class ModelResultSchema(BaseModel):
    model_name: str
    task_type: str
    best_cv_score: float
    best_params: Dict[str, Any]
    test_accuracy: Optional[float] = None
    test_precision: Optional[float] = None
    test_recall: Optional[float] = None
    test_f1: Optional[float] = None
    test_roc_auc: Optional[float] = None
    test_mae: Optional[float] = None
    test_rmse: Optional[float] = None
    test_r2: Optional[float] = None
    smote_applied: bool = False
    class_weight_applied: bool = False
    is_best: bool = False
    confusion_matrix: Optional[List[List[int]]] = None
    feature_importance: Optional[Dict[str, float]] = None


class TrainingResponse(BaseModel):
    dataset_id: str
    task_type: str
    models_trained: int
    results: List[ModelResultSchema]
    best_model: str
    best_model_params: Dict[str, Any]
    best_score: float
    selection_reason: str
    imbalance_strategy: str
