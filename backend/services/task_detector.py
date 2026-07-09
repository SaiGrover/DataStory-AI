import pandas as pd
import numpy as np
from typing import Dict, Any


def detect_task_type(df: pd.DataFrame, target: str) -> str:
    """Detect whether the target column indicates classification or regression."""
    col = df[target]
    nuniq = col.nunique()
    dtype = col.dtype

    if dtype == "object" or dtype.name == "category" or dtype == "bool":
        return "classification"

    if nuniq <= 20:
        return "classification"

    return "regression"


def detect_imbalance(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    """Detect class imbalance for classification tasks."""
    col = df[target].dropna()
    value_counts = col.value_counts(normalize=True) * 100

    distribution = {str(k): round(float(v), 1) for k, v in value_counts.items()}

    # Imbalanced if any class < 20% and len > 1
    min_pct = value_counts.min()
    imbalanced = (min_pct < 20.0) and (len(value_counts) > 1)

    return {
        "imbalanced": bool(imbalanced),
        "distribution": distribution,
        "min_class_pct": round(float(min_pct), 1),
        "n_classes": len(value_counts),
    }
