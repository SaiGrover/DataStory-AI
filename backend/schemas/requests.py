from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class CleanRequest(BaseModel):
    preview_only: bool = False
    config: Dict[str, Any] = Field(default_factory=dict)


class TrainRequest(BaseModel):
    target: str
    task_type: str
    model_names: List[str] = Field(min_length=1)
    imbalance_strategy: Optional[str] = None
    cv_folds: int = Field(default=5, ge=2, le=10)
    test_size: float = Field(default=0.2, gt=0, lt=0.5)


class ChatRequest(BaseModel):
    question: str
    target: Optional[str] = None
    task_type: Optional[str] = None
    cleaning_actions: Optional[List[str]] = None
    history: Optional[List[Dict[str, str]]] = None


class ReportRequest(BaseModel):
    target: Optional[str] = None
    task_type: Optional[str] = None
    cleaning_actions: Optional[List[str]] = None
    style: str = "Beginner Friendly"
