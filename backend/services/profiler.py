import pandas as pd
import numpy as np
from typing import Dict, Any, List


ID_NAME_MARKERS = {"id", "index", "serial", "uuid", "identifier", "passengerid"}
TARGET_NAME_MARKERS = {
    "target",
    "label",
    "outcome",
    "class",
    "category",
    "survived",
    "churn",
    "default",
    "diagnosis",
    "quality",
    "species",
    "y",
}


def _normalized_name(name: str) -> str:
    return str(name).lower().replace("_", "").replace("-", "").replace(" ", "")


def _is_probable_id_column(name: str, series: pd.Series, row_count: int) -> bool:
    """Detect identifiers so they do not become targets or analysis charts."""
    if row_count <= 0:
        return False
    normalized = _normalized_name(name)
    non_null = series.dropna()
    unique_count = int(non_null.nunique())
    unique_ratio = unique_count / max(len(non_null), 1)
    name_looks_like_id = (
        normalized in ID_NAME_MARKERS
        or normalized.endswith("id")
        or normalized.endswith("index")
        or normalized.startswith("id")
    )
    if name_looks_like_id and unique_ratio >= 0.8:
        return True
    if unique_ratio >= 0.98 and pd.api.types.is_integer_dtype(series) and unique_count > 20:
        return True
    return False


def _rank_possible_targets(df: pd.DataFrame, id_cols: List[str]) -> List[str]:
    row_count = max(len(df), 1)
    scored_targets = []
    for col in df.columns:
        if col in id_cols:
            continue
        series = df[col]
        unique_count = int(series.nunique(dropna=True))
        if unique_count <= 1:
            continue

        normalized = _normalized_name(col)
        unique_ratio = unique_count / row_count
        score = 0

        if normalized in TARGET_NAME_MARKERS:
            score += 100
        if any(marker in normalized for marker in ("target", "label", "outcome", "surviv", "churn", "class")):
            score += 45
        if unique_count == 2:
            score += 50
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
            if unique_count <= 20:
                score += 30
        elif pd.api.types.is_numeric_dtype(series):
            if 2 < unique_count <= max(25, int(row_count * 0.35)):
                score += 18

        if unique_ratio > 0.9:
            score -= 75
        if normalized in {"name", "ticket", "cabin"}:
            score -= 25

        if score > 0:
            scored_targets.append((score, col))

    scored_targets.sort(key=lambda item: (-item[0], str(item[1]).lower()))
    return [col for _, col in scored_targets[:10]]


def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate a comprehensive data profile and health score."""
    profile: Dict[str, Any] = {}
    warnings: List[Dict[str, str]] = []

    profile["rows"] = len(df)
    profile["columns"] = len(df.columns)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    bool_cols = df.select_dtypes(include="bool").columns.tolist()

    # Try to detect date columns
    date_cols = []
    for col in cat_cols:
        try:
            parsed = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
            if parsed.notna().mean() > 0.7:
                date_cols.append(col)
        except Exception:
            pass

    profile["numeric_cols"] = len(numeric_cols)
    profile["categorical_cols"] = len(cat_cols)
    profile["date_cols"] = len(date_cols)
    profile["bool_cols"] = len(bool_cols)

    id_cols = [col for col in df.columns if _is_probable_id_column(col, df[col], len(df))]
    profile["id_cols"] = id_cols

    # Missing values
    total_missing = df.isnull().sum().sum()
    profile["total_missing"] = int(total_missing)

    missing_by_col = df.isnull().sum()
    cols_with_missing = missing_by_col[missing_by_col > 0]
    if len(cols_with_missing) > 0:
        warnings.append({
            "title": f"{len(cols_with_missing)} column(s) contain missing values",
            "message": f"Columns: {', '.join(cols_with_missing.index[:5].tolist())}. "
                       f"Visit Clean Room to handle them.",
        })

    # High missing columns
    high_missing = [c for c in df.columns if df[c].isnull().mean() > 0.5]
    if high_missing:
        warnings.append({
            "title": f"{len(high_missing)} column(s) have >50% missing values",
            "message": f"Consider dropping: {', '.join(high_missing[:5])}",
        })

    # Duplicates
    n_dups = int(df.duplicated().sum())
    profile["duplicates"] = n_dups
    if n_dups > 0:
        warnings.append({
            "title": f"{n_dups:,} duplicate rows found",
            "message": "Duplicate rows can bias model training. Remove them in Clean Room.",
        })

    # Constant columns
    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    profile["constant_cols"] = constant_cols
    if constant_cols:
        warnings.append({
            "title": f"{len(constant_cols)} constant column(s) found",
            "message": f"These add no information: {', '.join(constant_cols)}",
        })

    # High cardinality
    high_card = [c for c in cat_cols if df[c].nunique() > 50]
    profile["high_cardinality_cols"] = high_card
    if high_card:
        warnings.append({
            "title": f"{len(high_card)} high-cardinality column(s)",
            "message": f"These may need special encoding: {', '.join(high_card[:5])}",
        })

    # Column details
    col_details = []
    for col in df.columns:
        col_details.append({
            "Column": col,
            "Type": str(df[col].dtype),
            "Non-Null": int(df[col].notna().sum()),
            "Null %": round(df[col].isnull().mean() * 100, 1),
            "Unique": int(df[col].nunique()),
            "Top Value": str(df[col].mode().iloc[0]) if not df[col].empty else "—",
        })
    profile["column_details"] = col_details

    # Possible target columns
    profile["possible_targets"] = _rank_possible_targets(df, id_cols)

    profile["warnings"] = warnings

    # Health score
    score = 100
    missing_pct = total_missing / max(df.size, 1)
    score -= min(30, int(missing_pct * 100))
    if n_dups > 0:
        score -= min(15, int((n_dups / len(df)) * 30))
    score -= min(10, len(high_missing) * 5)
    score -= min(10, len(constant_cols) * 3)
    score -= min(5, len(high_card) * 2)
    profile["health_score"] = max(0, score)

    return profile
