"""
DataStory AI - EDA Service
Generates exploratory data analysis results and chart data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
import logging

logger = logging.getLogger(__name__)

# Soft Studio color palette
SOFT_PALETTE = [
    "#9AAFE8",  # Soft Periwinkle
    "#E9A7B3",  # Dusty Rose
    "#B8DCC8",  # Soft Mint
    "#F6E7B8",  # Butter Cream
    "#8FB996",  # Sage Green
    "#E5B567",  # Soft Amber
]


def generate_eda(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Run full EDA on a cleaned dataframe.
    Returns chart JSON configs and insights.
    """
    result = {}

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Overview
    result["overview"] = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numeric_count": len(numeric_cols),
        "categorical_count": len(categorical_cols),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    # Column type distribution chart
    result["column_type_chart"] = _column_type_pie(df, numeric_cols, categorical_cols)

    # Missing values chart
    result["missing_chart"] = _missing_values_chart(df)
    result["missing_insight"] = _missing_insight(df)

    # Numeric summaries + histograms + boxplots
    result["numeric_summaries"] = []
    for col in numeric_cols[:10]:  # limit to 10
        result["numeric_summaries"].append({
            "column": col,
            "histogram": _histogram(df, col),
            "boxplot": _boxplot(df, col),
            "stats": _col_stats(df, col),
            "insight": _numeric_insight(df, col),
        })

    # Categorical summaries + bar charts
    result["categorical_summaries"] = []
    for col in categorical_cols[:8]:  # limit to 8
        result["categorical_summaries"].append({
            "column": col,
            "bar_chart": _bar_chart(df, col),
            "value_counts": df[col].value_counts().head(10).to_dict(),
            "insight": _categorical_insight(df, col),
        })

    # Correlation matrix
    if len(numeric_cols) >= 2:
        result["correlation_matrix"] = _correlation_heatmap(df, numeric_cols)
        result["top_correlations"] = _top_correlations(df, numeric_cols)
    else:
        result["correlation_matrix"] = None
        result["top_correlations"] = []

    # Target distribution
    if target_col and target_col in df.columns:
        result["target_distribution"] = _target_distribution(df, target_col)
        result["target_insight"] = _target_insight(df, target_col)
    else:
        result["target_distribution"] = None
        result["target_insight"] = None

    return result


def _column_type_pie(df, numeric_cols, categorical_cols) -> Dict:
    other = df.shape[1] - len(numeric_cols) - len(categorical_cols)
    labels = ["Numeric", "Categorical"]
    values = [len(numeric_cols), len(categorical_cols)]
    if other > 0:
        labels.append("Other")
        values.append(other)
    fig = px.pie(
        names=labels,
        values=values,
        color_discrete_sequence=SOFT_PALETTE,
        hole=0.4,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif", color="#2F3437"),
        margin=dict(t=10, b=10, l=10, r=10),
        showlegend=True,
    )
    return fig.to_json()


def _missing_values_chart(df: pd.DataFrame) -> Dict:
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=True)
    if missing.empty:
        return None
    fig = px.bar(
        x=missing.values,
        y=missing.index,
        orientation="h",
        color_discrete_sequence=["#E9A7B3"],
        labels={"x": "Missing Count", "y": "Column"},
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif", color="#2F3437"),
        margin=dict(t=10, b=30, l=10, r=10),
        xaxis=dict(gridcolor="#E8E6E1"),
        yaxis=dict(gridcolor="#E8E6E1"),
    )
    return fig.to_json()


def _missing_insight(df: pd.DataFrame) -> str:
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        return "No missing values found. Your dataset is complete."
    worst = missing.idxmax()
    return (
        f"The column '{worst}' has the most missing values ({missing[worst]} rows). "
        f"You can fill or drop these in Clean Room before training."
    )


def _histogram(df: pd.DataFrame, col: str) -> str:
    fig = px.histogram(
        df,
        x=col,
        nbins=30,
        color_discrete_sequence=["#9AAFE8"],
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif", color="#2F3437"),
        margin=dict(t=10, b=30, l=10, r=10),
        xaxis=dict(gridcolor="#E8E6E1", title=col),
        yaxis=dict(gridcolor="#E8E6E1", title="Count"),
        bargap=0.05,
    )
    return fig.to_json()


def _boxplot(df: pd.DataFrame, col: str) -> str:
    fig = px.box(
        df,
        y=col,
        color_discrete_sequence=["#B8DCC8"],
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif", color="#2F3437"),
        margin=dict(t=10, b=30, l=10, r=10),
        yaxis=dict(gridcolor="#E8E6E1"),
    )
    return fig.to_json()


def _col_stats(df: pd.DataFrame, col: str) -> Dict:
    s = df[col].describe()
    return {k: round(float(v), 4) for k, v in s.items()}


def _numeric_insight(df: pd.DataFrame, col: str) -> str:
    skew = df[col].skew()
    mean = df[col].mean()
    std = df[col].std()
    if abs(skew) > 1:
        direction = "right" if skew > 0 else "left"
        return (
            f"'{col}' is skewed to the {direction} (skewness={round(skew,2)}). "
            f"Mean is {round(mean,2)} with std deviation {round(std,2)}."
        )
    return (
        f"'{col}' looks roughly symmetric. "
        f"Average value is {round(mean,2)} with std deviation {round(std,2)}."
    )


def _bar_chart(df: pd.DataFrame, col: str) -> str:
    counts = df[col].value_counts().head(10)
    fig = px.bar(
        x=counts.index.astype(str),
        y=counts.values,
        color_discrete_sequence=["#9AAFE8"],
        labels={"x": col, "y": "Count"},
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif", color="#2F3437"),
        margin=dict(t=10, b=60, l=10, r=10),
        xaxis=dict(gridcolor="#E8E6E1", title=col),
        yaxis=dict(gridcolor="#E8E6E1", title="Count"),
    )
    return fig.to_json()


def _categorical_insight(df: pd.DataFrame, col: str) -> str:
    top = df[col].value_counts().idxmax()
    top_pct = round(df[col].value_counts(normalize=True).max() * 100, 1)
    n_unique = df[col].nunique()
    return (
        f"'{col}' has {n_unique} unique values. "
        f"The most common value is '{top}', appearing in {top_pct}% of rows."
    )


def _correlation_heatmap(df: pd.DataFrame, numeric_cols: List[str]) -> str:
    corr = df[numeric_cols].corr()
    fig = px.imshow(
        corr,
        color_continuous_scale=[
            [0, "#E9A7B3"],
            [0.5, "#FAFAF7"],
            [1, "#9AAFE8"],
        ],
        zmin=-1,
        zmax=1,
        text_auto=".2f",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif", color="#2F3437", size=11),
        margin=dict(t=10, b=10, l=10, r=10),
        coloraxis_showscale=True,
    )
    return fig.to_json()


def _top_correlations(df: pd.DataFrame, numeric_cols: List[str]) -> List[Dict]:
    corr = df[numeric_cols].corr()
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if not np.isnan(val):
                pairs.append({
                    "col_a": cols[i],
                    "col_b": cols[j],
                    "correlation": round(float(val), 4),
                })
    pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return pairs[:10]


def _target_distribution(df: pd.DataFrame, target_col: str) -> str:
    if df[target_col].dtype == "object" or df[target_col].nunique() <= 20:
        counts = df[target_col].value_counts()
        fig = px.bar(
            x=counts.index.astype(str),
            y=counts.values,
            color_discrete_sequence=["#8FB996"],
            labels={"x": target_col, "y": "Count"},
        )
    else:
        fig = px.histogram(
            df,
            x=target_col,
            nbins=30,
            color_discrete_sequence=["#8FB996"],
        )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif", color="#2F3437"),
        margin=dict(t=10, b=40, l=10, r=10),
        xaxis=dict(gridcolor="#E8E6E1"),
        yaxis=dict(gridcolor="#E8E6E1"),
    )
    return fig.to_json()


def _target_insight(df: pd.DataFrame, target_col: str) -> str:
    if df[target_col].dtype == "object" or df[target_col].nunique() <= 20:
        counts = df[target_col].value_counts(normalize=True)
        dominant = counts.idxmax()
        pct = round(counts.max() * 100, 1)
        if pct > 70:
            return (
                f"The target '{target_col}' is imbalanced — '{dominant}' makes up {pct}% of rows. "
                f"Consider handling class imbalance during training."
            )
        return f"The target '{target_col}' appears reasonably balanced across classes."
    else:
        mean = round(df[target_col].mean(), 2)
        std = round(df[target_col].std(), 2)
        return f"Target '{target_col}' is continuous. Mean={mean}, Std={std}. This is a regression task."
