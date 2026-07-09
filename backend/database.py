"""
DataStory AI - Database Setup (SQLite via SQLAlchemy)
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy import Column, String, Integer, Float, Text, DateTime, JSON
from datetime import datetime
import os


DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./datastory.db")

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class DatasetRecord(Base):
    __tablename__ = "datasets"

    id = Column(String, primary_key=True)
    file_name = Column(String, nullable=False)
    upload_timestamp = Column(DateTime, default=datetime.utcnow)
    rows = Column(Integer)
    columns = Column(Integer)
    column_names = Column(JSON)
    file_size_kb = Column(Float)
    health_score = Column(Float, nullable=True)
    selected_target = Column(String, nullable=True)
    task_type = Column(String, nullable=True)
    cleaning_actions = Column(JSON, nullable=True)
    imbalance_detected = Column(Integer, default=0)
    imbalance_strategy = Column(String, nullable=True)
    status = Column(String, default="uploaded")


class ModelResult(Base):
    __tablename__ = "model_results"

    id = Column(String, primary_key=True)
    dataset_id = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    task_type = Column(String)
    best_cv_score = Column(Float)
    best_params = Column(JSON)
    test_accuracy = Column(Float, nullable=True)
    test_precision = Column(Float, nullable=True)
    test_recall = Column(Float, nullable=True)
    test_f1 = Column(Float, nullable=True)
    test_roc_auc = Column(Float, nullable=True)
    test_mae = Column(Float, nullable=True)
    test_rmse = Column(Float, nullable=True)
    test_r2 = Column(Float, nullable=True)
    smote_applied = Column(Integer, default=0)
    class_weight_applied = Column(Integer, default=0)
    is_best = Column(Integer, default=0)
    trained_at = Column(DateTime, default=datetime.utcnow)


class ReportRecord(Base):
    __tablename__ = "reports"

    id = Column(String, primary_key=True)
    dataset_id = Column(String, nullable=False)
    report_style = Column(String, default="beginner")
    content_md = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(String, primary_key=True)
    dataset_id = Column(String, nullable=False)
    role = Column(String)
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
