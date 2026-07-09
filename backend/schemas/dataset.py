"""
DataStory AI - Dataset Schemas
"""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime


class DatasetMeta(BaseModel):
    dataset_id: str
    file_name: str
    rows: int
    columns: int
    column_names: List[str]
    file_size_kb: float
    upload_timestamp: datetime


class DataProfileResult(BaseModel):
    dataset_id: str
    total_rows: int
    total_columns: int
    numeric_columns: List[str]
    categorical_columns: List[str]
    date_columns: List[str]
    missing_values: Dict[str, int]
    total_missing: int
    duplicate_rows: int
    constant_columns: List[str]
    high_cardinality_columns: List[str]
    unique_values: Dict[str, int]
    dtypes: Dict[str, str]
    summary_stats: Dict[str, Any]
    possible_targets: List[str]
    warnings: List[str]
    health_score: float


class CleaningRequest(BaseModel):
    dataset_id: str
    actions: List[str]
    custom_fill_value: Optional[str] = None
    drop_missing_threshold: float = 0.5


class CleaningResult(BaseModel):
    dataset_id: str
    actions_applied: List[str]
    before: Dict[str, Any]
    after: Dict[str, Any]
    summary: str


class TargetSelectionRequest(BaseModel):
    dataset_id: str
    target_column: str


class TaskDetectionResult(BaseModel):
    dataset_id: str
    target_column: str
    task_type: str
    num_classes: Optional[int]
    class_distribution: Optional[Dict[str, Any]]
    target_summary: Optional[Dict[str, Any]]
    warnings: List[str]
