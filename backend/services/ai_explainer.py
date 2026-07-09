"""
DataStory AI - AI Explainer Service
Uses OpenRouter API to generate natural language explanations.
"""

import os
import httpx
import logging
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct")


def _get_api_key() -> Optional[str]:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        logger.warning("OPENROUTER_API_KEY not set.")
    return key


async def generate_explanation(
    prompt: str,
    system_prompt: str = "You are DataStory AI, a friendly and beginner-friendly data science assistant. Explain concepts in simple, clear language.",
    max_tokens: int = 800,
) -> str:
    """
    Call OpenRouter API to generate an AI explanation.
    Returns the explanation text or a fallback error message.
    """
    api_key = _get_api_key()
    if not api_key:
        return (
            "OpenRouter API key not found. "
            "Please add OPENROUTER_API_KEY to your .env file to enable AI explanations."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://datastory.ai",
        "X-Title": "DataStory AI",
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.4,
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(OPENROUTER_BASE_URL, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
    except httpx.HTTPStatusError as e:
        logger.error(f"OpenRouter HTTP error: {e.response.status_code} - {e.response.text}")
        return f"AI explanation unavailable (HTTP {e.response.status_code}). Please try again."
    except Exception as e:
        logger.error(f"OpenRouter error: {e}")
        return "AI explanation temporarily unavailable. Please try again later."


async def explain_dataset_summary(profile: Dict) -> str:
    prompt = f"""
A user uploaded a dataset with the following profile:
- Rows: {profile.get('total_rows')}
- Columns: {profile.get('total_columns')}
- Numeric columns: {len(profile.get('numeric_columns', []))}
- Categorical columns: {len(profile.get('categorical_columns', []))}
- Total missing values: {profile.get('total_missing')}
- Duplicate rows: {profile.get('duplicate_rows')}
- Data Health Score: {profile.get('health_score')}/100
- Warnings: {', '.join(profile.get('warnings', [])) or 'None'}

Please explain what this profile means in simple, beginner-friendly language. 
What should the user pay attention to before proceeding?
"""
    return await generate_explanation(prompt)


async def explain_cleaning(actions_applied: List[str], before: Dict, after: Dict) -> str:
    prompt = f"""
The following data cleaning actions were applied to a dataset:
{chr(10).join(f'- {a}' for a in actions_applied)}

Before cleaning:
- Rows: {before.get('rows')}
- Missing values: {before.get('missing_values')}
- Duplicates: {before.get('duplicate_rows')}

After cleaning:
- Rows: {after.get('rows')}
- Missing values: {after.get('missing_values')}
- Duplicates: {after.get('duplicate_rows')}

Explain what was done and why it helps, in plain language that a beginner can understand.
"""
    return await generate_explanation(prompt)


async def explain_model_results(results: List[Dict], task_type: str, is_imbalanced: bool, best_model: str) -> str:
    model_summaries = []
    for r in results:
        if task_type == "classification":
            model_summaries.append(
                f"- {r['model_name']}: F1={r.get('test_f1', 'N/A')}, Accuracy={r.get('test_accuracy', 'N/A')}"
            )
        else:
            model_summaries.append(
                f"- {r['model_name']}: RMSE={r.get('test_rmse', 'N/A')}, R2={r.get('test_r2', 'N/A')}"
            )

    prompt = f"""
A machine learning experiment was run with the following results:
Task type: {task_type}
Class imbalance: {'Yes' if is_imbalanced else 'No'}
Best model: {best_model}

Model comparison:
{chr(10).join(model_summaries)}

Please explain:
1. Why the best model was selected
2. What the metrics mean
3. What the user should take away from these results

Keep the explanation beginner-friendly.
"""
    return await generate_explanation(prompt)


async def explain_best_parameters(model_name: str, params: Dict, task_type: str) -> str:
    param_str = "\n".join(f"- {k}: {v}" for k, v in params.items())
    prompt = f"""
The best hyperparameters found by GridSearchCV for {model_name} ({task_type}) are:
{param_str}

Please explain what each hyperparameter means and why these particular values might have worked well.
Explain in simple, beginner-friendly language.
"""
    return await generate_explanation(prompt)


async def generate_full_report(
    profile: Dict,
    cleaning_summary: Dict,
    task_type: str,
    target_col: str,
    best_model: str,
    best_params: Dict,
    results: List[Dict],
    imbalance_strategy: str,
    report_style: str = "beginner",
) -> str:
    style_instruction = {
        "beginner": "Write in very simple, friendly language. Avoid jargon. Explain terms when used.",
        "technical": "Write in a technical style with precise terminology suitable for a data science audience.",
        "business": "Write in a concise business style focused on insights and outcomes, not technical details.",
    }.get(report_style, "beginner")

    model_results_str = "\n".join(
        f"- {r['model_name']}: F1={r.get('test_f1','N/A')} Acc={r.get('test_accuracy','N/A')}"
        if task_type == "classification"
        else f"- {r['model_name']}: RMSE={r.get('test_rmse','N/A')} R2={r.get('test_r2','N/A')}"
        for r in results
    )

    prompt = f"""
Generate a complete data analysis report for DataStory AI in Markdown format.

Style: {style_instruction}

Dataset Profile:
- Rows: {profile.get('total_rows')}, Columns: {profile.get('total_columns')}
- Health Score: {profile.get('health_score')}/100
- Missing values: {profile.get('total_missing')}, Duplicates: {profile.get('duplicate_rows')}

Cleaning Applied: {', '.join(cleaning_summary.get('actions_applied', ['None']))}

ML Task: {task_type}
Target Column: {target_col}
Imbalance Strategy: {imbalance_strategy}
Best Model: {best_model}
Best Parameters: {best_params}

Model Comparison:
{model_results_str}

The report should include these sections:
1. Dataset Overview
2. Data Health Summary
3. Cleaning Summary
4. Key EDA Insights
5. ML Task Framing
6. Model Selection Process
7. Class Imbalance Handling
8. GridSearchCV Results
9. Best Model and Parameters
10. Final Recommendations

Make it educational and actionable.
"""
    return await generate_explanation(prompt, max_tokens=2000)
