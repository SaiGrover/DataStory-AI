import pandas as pd
from typing import Optional, List, Dict, Any
from backend.services.openrouter import answer_question
from backend.rag.retriever import retrieve_context


def chat_with_dataset(
    question: str,
    df: Optional[pd.DataFrame],
    profile: Optional[Dict],
    results: Optional[List[Dict]],
    best_model: Optional[str],
    task: Optional[str],
    target: Optional[str],
    cleaning_actions: Optional[List[str]],
) -> str:
    """Answer a user question about their dataset and analysis."""

    # Build context string
    context_parts = []

    if df is not None:
        context_parts.append(
            f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns. "
            f"Columns: {', '.join(df.columns[:15].tolist())}."
        )

    if profile:
        context_parts.append(
            f"Health score: {profile.get('health_score', '?')}/100. "
            f"Missing values: {profile.get('total_missing', 0)}. "
            f"Duplicates: {profile.get('duplicates', 0)}."
        )

    if task and target:
        context_parts.append(f"Task type: {task}. Target column: {target}.")

    if cleaning_actions:
        context_parts.append(f"Cleaning applied: {'; '.join(cleaning_actions)}.")

    if results:
        valid = [r for r in results if "error" not in r]
        summaries = [
            f"{r.get('model_name')}: "
            f"{'F1=' + str(r.get('f1')) if task == 'classification' else 'RMSE=' + str(r.get('rmse'))}"
            for r in valid
        ]
        context_parts.append(f"Models trained: {', '.join(summaries)}.")

    if best_model:
        best = next((r for r in (results or []) if r.get("model_name") == best_model), None)
        if best:
            context_parts.append(
                f"Best model: {best_model}. "
                f"Reason: {best.get('reason', 'Highest performance metric.')}. "
                f"Best params: {best.get('best_params', {})}."
            )

    # Retrieve RAG context
    rag_context = retrieve_context(question)
    if rag_context:
        context_parts.append(f"\nRelevant knowledge:\n{rag_context}")

    full_context = "\n".join(context_parts)
    return answer_question(question, full_context)
