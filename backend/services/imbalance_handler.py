"""
DataStory AI - Class Imbalance Handler
Detects and handles class imbalance using SMOTE or class_weight.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

SMOTE_SUPPORTED_MODELS = [
    "Logistic Regression",
    "Decision Tree Classifier",
    "Random Forest Classifier",
    "Support Vector Machine",
    "K-Nearest Neighbors",
]

CLASS_WEIGHT_SUPPORTED_MODELS = [
    "Logistic Regression",
    "Decision Tree Classifier",
    "Random Forest Classifier",
    "Support Vector Machine",
]


def detect_imbalance(y: pd.Series, threshold: float = 0.2) -> Dict[str, Any]:
    """
    Detect class imbalance in the target series.

    Args:
        y: Target column
        threshold: If minority class ratio < threshold, flag as imbalanced

    Returns:
        Dict with imbalance info
    """
    counts = y.value_counts()
    total = len(y)
    ratios = (counts / total).to_dict()
    min_ratio = min(ratios.values())
    is_imbalanced = min_ratio < threshold

    return {
        "is_imbalanced": is_imbalanced,
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "class_ratios": {str(k): round(float(v) * 100, 1) for k, v in ratios.items()},
        "minority_class": str(counts.idxmin()),
        "majority_class": str(counts.idxmax()),
        "minority_ratio": round(float(min_ratio) * 100, 1),
        "majority_ratio": round(float(max(ratios.values())) * 100, 1),
    }


def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, bool, str]:
    """
    Apply SMOTE to training data only.

    Returns:
        X_resampled, y_resampled, success, message
    """
    try:
        from imblearn.over_sampling import SMOTE

        # Check minimum samples
        counts = pd.Series(y_train).value_counts()
        min_count = counts.min()

        if min_count < 6:
            return (
                X_train,
                y_train,
                False,
                f"SMOTE cannot be applied — the minority class has only {min_count} samples. "
                f"At least 6 are needed. Using class_weight='balanced' instead.",
            )

        smote = SMOTE(random_state=random_state)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        msg = (
            f"SMOTE applied on training data. Samples increased from {len(y_train)} to {len(y_res)}."
        )
        return X_res, y_res, True, msg

    except ImportError:
        return (
            X_train,
            y_train,
            False,
            "imbalanced-learn is not installed. Run: pip install imbalanced-learn",
        )
    except Exception as e:
        return X_train, y_train, False, f"SMOTE failed: {str(e)}"


def get_class_weight_param(model_name: str, is_imbalanced: bool) -> Optional[str]:
    """
    Return class_weight='balanced' if the model supports it and imbalance exists.
    """
    if is_imbalanced and model_name in CLASS_WEIGHT_SUPPORTED_MODELS:
        return "balanced"
    return None
