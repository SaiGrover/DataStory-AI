"""
DataStory AI - Report Schemas
"""

from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class ReportRequest(BaseModel):
    dataset_id: str
    report_style: str = "beginner"  # "beginner" | "technical" | "business"


class ReportResponse(BaseModel):
    report_id: str
    dataset_id: str
    report_style: str
    content_md: str
    created_at: datetime


class ChatRequest(BaseModel):
    dataset_id: str
    question: str


class ChatResponse(BaseModel):
    dataset_id: str
    question: str
    answer: str
    sources: Optional[list] = None
