"""
DataStory AI — SQLite database utilities.
Stores dataset metadata, results, and chat history.
"""
import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

DB_PATH = os.environ.get("DATASTORY_DB", "datastory.db")


def _connect():
    return sqlite3.connect(DB_PATH)


def init_db():
    """Create tables if they don't exist."""
    conn = _connect()
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS datasets (
            id TEXT PRIMARY KEY,
            filename TEXT,
            rows INTEGER,
            columns INTEGER,
            uploaded_at TEXT,
            target_column TEXT,
            task_type TEXT,
            best_model TEXT
        );

        CREATE TABLE IF NOT EXISTS training_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id TEXT,
            model_name TEXT,
            metrics TEXT,
            best_params TEXT,
            created_at TEXT,
            FOREIGN KEY(dataset_id) REFERENCES datasets(id)
        );

        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id TEXT,
            style TEXT,
            content TEXT,
            created_at TEXT,
            FOREIGN KEY(dataset_id) REFERENCES datasets(id)
        );

        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id TEXT,
            role TEXT,
            content TEXT,
            created_at TEXT,
            FOREIGN KEY(dataset_id) REFERENCES datasets(id)
        );
    """)
    conn.commit()
    conn.close()


def save_dataset_meta(dataset_id: str, filename: str, rows: int, cols: int):
    conn = _connect()
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO datasets (id, filename, rows, columns, uploaded_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (dataset_id, filename, rows, cols, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_dataset_meta(dataset_id: str) -> Optional[Dict]:
    conn = _connect()
    c = conn.cursor()
    row = c.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
    conn.close()
    if row:
        cols = ["id", "filename", "rows", "columns", "uploaded_at", "target_column", "task_type", "best_model"]
        return dict(zip(cols, row))
    return None


def save_results(dataset_id: str, results: List[Dict[str, Any]]):
    conn = _connect()
    c = conn.cursor()
    for r in results:
        metrics = {k: v for k, v in r.items() if k not in ["best_params", "confusion_matrix", "ai_explanation"]}
        c.execute(
            "INSERT INTO training_results (dataset_id, model_name, metrics, best_params, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                dataset_id,
                r.get("model_name", ""),
                json.dumps(metrics),
                json.dumps(r.get("best_params", {})),
                datetime.utcnow().isoformat(),
            ),
        )
    conn.commit()
    conn.close()


def save_chat_message(dataset_id: str, role: str, content: str):
    conn = _connect()
    c = conn.cursor()
    c.execute(
        "INSERT INTO chat_history (dataset_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (dataset_id, role, content, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_chat_history(dataset_id: str) -> List[Dict]:
    conn = _connect()
    c = conn.cursor()
    rows = c.execute(
        "SELECT role, content, created_at FROM chat_history WHERE dataset_id = ? ORDER BY id",
        (dataset_id,),
    ).fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1], "at": r[2]} for r in rows]


def save_report(dataset_id: str, style: str, content: str):
    conn = _connect()
    c = conn.cursor()
    c.execute(
        "INSERT INTO reports (dataset_id, style, content, created_at) VALUES (?, ?, ?, ?)",
        (dataset_id, style, json.dumps(content), datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()
