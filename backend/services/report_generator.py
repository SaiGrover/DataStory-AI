import pandas as pd
from typing import Dict, Any, List, Optional
from backend.services.openrouter import generate_report_section


def generate_report(
    df: pd.DataFrame,
    profile: Optional[Dict],
    cleaning_actions: List[str],
    results: Optional[List[Dict]],
    best_model: Optional[str],
    task: Optional[str],
    target: Optional[str],
    style: str = "Beginner Friendly",
) -> Dict[str, str]:
    report = {}

    # 1. Dataset Overview
    rows = df.shape[0] if df is not None else "?"
    cols = df.shape[1] if df is not None else "?"
    col_names = ", ".join(df.columns[:8].tolist()) + ("..." if df.shape[1] > 8 else "") if df is not None else "?"
    prompt = (
        f"Write a dataset overview section. The dataset has {rows} rows and {cols} columns. "
        f"Columns include: {col_names}. Target column: {target}. Task type: {task}."
    )
    report["dataset_overview"] = generate_report_section(prompt, style)

    # 2. Data Health Summary
    if profile:
        score = profile.get("health_score", "?")
        missing = profile.get("total_missing", 0)
        dups = profile.get("duplicates", 0)
        prompt = (
            f"Summarise the data health. Health score: {score}/100. "
            f"Total missing values: {missing}. Duplicate rows: {dups}. "
            f"Warnings: {[w['title'] for w in profile.get('warnings', [])]}."
        )
        report["health_summary"] = generate_report_section(prompt, style)

    # 3. Cleaning Summary
    if cleaning_actions:
        prompt = (
            f"Summarise the data cleaning steps applied: {cleaning_actions}. "
            f"Explain why each step matters for model accuracy."
        )
        report["cleaning_summary"] = generate_report_section(prompt, style)
    else:
        report["cleaning_summary"] = "No cleaning actions were applied to this dataset."

    # 4. EDA Insights
    if df is not None:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        prompt = (
            f"Write EDA insights. The dataset has {len(num_cols)} numeric columns "
            f"({', '.join(num_cols[:5])}) and {len(cat_cols)} categorical columns "
            f"({', '.join(cat_cols[:5])}). Target: {target}. Task: {task}."
        )
        report["eda_insights"] = generate_report_section(prompt, style)

    # 5. ML Task Framing
    prompt = (
        f"Explain how this ML problem is framed. Target column: {target}. "
        f"Task type: {task}. Explain what the model is trying to predict."
    )
    report["task_framing"] = generate_report_section(prompt, style)

    # 6. Model Selection
    if results:
        model_names = [r.get("model_name", "") for r in results if "error" not in r]
        prompt = (
            f"Explain why these models were selected for training: {model_names}. "
            f"Task type: {task}."
        )
        report["model_selection"] = generate_report_section(prompt, style)

    # 7. Imbalance Handling
    if task == "classification":
        prompt = (
            "Explain why class imbalance should be checked for classification workflows. "
            "Mention that resampling or class weighting should be applied only on training data."
        )
        report["imbalance_handling"] = generate_report_section(prompt, style)

    # 8. GridSearchCV Results
    if results:
        valid = [r for r in results if "error" not in r]
        summaries = [
            f"{r['model_name']}: CV={r.get('cv_score', '?')}, "
            f"best params={r.get('best_params', {})}"
            for r in valid
        ]
        prompt = (
            f"Summarise GridSearchCV results: {summaries}. "
            f"Explain what GridSearchCV does and why best parameters matter."
        )
        report["gridsearch_results"] = generate_report_section(prompt, style)

    # 9. Best Model
    if best_model and results:
        best = next((r for r in results if r.get("model_name") == best_model), None)
        if best:
            metrics = {k: v for k, v in best.items()
                       if k in ["accuracy", "f1", "recall", "precision", "roc_auc",
                                 "rmse", "mae", "r2", "best_params"]}
            prompt = (
                f"Write a conclusion about the best model: {best_model}. "
                f"Metrics: {metrics}. Reason: {best.get('reason', '')}. "
                f"What does this mean for the user?"
            )
            report["best_model_section"] = generate_report_section(prompt, style)

    # 10. Recommendations
    prompt = (
        f"Write final recommendations for the user after completing this data science workflow. "
        f"Best model: {best_model}. Task: {task}. Target: {target}. "
        f"What should they do next? How can they improve results?"
    )
    report["recommendations"] = generate_report_section(prompt, style)

    return report


def report_to_markdown(report: Dict[str, str]) -> str:
    sections = {
        "dataset_overview": "## 📁 Dataset Overview",
        "health_summary": "## 📊 Data Health Summary",
        "cleaning_summary": "## 🧹 Cleaning Summary",
        "eda_insights": "## 🔍 EDA Insights",
        "task_framing": "## 🎯 ML Task Framing",
        "model_selection": "## 🤖 Model Selection",
        "imbalance_handling": "## ⚖️ Class Imbalance Handling",
        "gridsearch_results": "## ⚙️ GridSearchCV Results",
        "best_model_section": "## 🏆 Best Model & Parameters",
        "recommendations": "## 📈 Final Recommendations",
    }
    lines = ["# DataStory AI — Analysis Report", ""]
    for key, heading in sections.items():
        if report.get(key):
            lines.append(heading)
            lines.append("")
            lines.append(report[key])
            lines.append("")
    return "\n".join(lines)
