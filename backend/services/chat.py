"""Grounded dataset Q&A for the DataStory agent."""

import re
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from backend.rag.retriever import retrieve_documents
from backend.services.openrouter import answer_question


ERROR_PREFIXES = (
    "OpenRouter API key not found", "API error", "Unexpected error", "Request timed out"
)


def _normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _mentioned_columns(question: str, df: pd.DataFrame) -> List[str]:
    q_normal = _normalise(question)
    q_tokens = {_normalise(token) for token in re.findall(r"[A-Za-z0-9_]+", question)}
    matches = []
    for column in df.columns:
        normal = _normalise(column)
        if normal and (normal in q_tokens or (len(normal) >= 4 and normal in q_normal)):
            matches.append(column)
    return matches[:6]


def _format_number(value: Any) -> str:
    try:
        number = float(value)
        if pd.isna(number):
            return "n/a"
        return f"{number:,.3f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(value)


def _column_evidence(df: pd.DataFrame, columns: Sequence[str]) -> List[str]:
    evidence: List[str] = []
    for column in columns:
        series = df[column]
        missing = int(series.isna().sum())
        missing_pct = float(series.isna().mean())
        unique = int(series.nunique(dropna=True))
        prefix = f"Column '{column}' ({series.dtype}): {missing:,} missing ({missing_pct:.1%}), {unique:,} unique."
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if numeric.empty:
                evidence.append(prefix)
            else:
                evidence.append(
                    prefix + " " +
                    f"Min {_format_number(numeric.min())}; Q1 {_format_number(numeric.quantile(.25))}; "
                    f"median {_format_number(numeric.median())}; mean {_format_number(numeric.mean())}; "
                    f"Q3 {_format_number(numeric.quantile(.75))}; max {_format_number(numeric.max())}."
                )
        else:
            counts = series.fillna("(missing)").astype(str).value_counts().head(8)
            total = max(len(series), 1)
            distribution = ", ".join(
                f"{label}={int(count):,} ({count / total:.1%})" for label, count in counts.items()
            )
            evidence.append(prefix + f" Top values: {distribution}.")
    return evidence


def _candidate_class_columns(df: pd.DataFrame) -> List[str]:
    candidates: List[str] = []
    for column in df.columns:
        non_null = df[column].dropna()
        unique = int(non_null.nunique())
        if not non_null.empty and 2 <= unique <= 20 and unique / len(non_null) <= 0.2:
            candidates.append(column)
    return candidates[:6]


def _class_balance(df: pd.DataFrame, column: str) -> str:
    counts = df[column].fillna("(missing)").astype(str).value_counts()
    if len(counts) < 2:
        return f"'{column}' has fewer than two observed classes, so class balance cannot be assessed."
    total = int(counts.sum())
    minority = int(counts.min())
    status = "imbalanced" if minority / total < 0.2 else "not strongly imbalanced"
    distribution = ", ".join(f"{label}={int(count):,} ({count / total:.1%})" for label, count in counts.head(10).items())
    return f"Class balance for '{column}': {status}; {distribution}."


def _model_evidence(results: Optional[List[Dict]], best_model: Optional[str], task: Optional[str]) -> List[str]:
    evidence: List[str] = []
    for result in results or []:
        if result.get("error"):
            continue
        metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"] if task == "classification" else ["rmse", "mae", "r2"]
        metrics = ", ".join(
            f"{name.upper()}={_format_number(result.get(name))}"
            for name in metric_names if result.get(name) is not None
        )
        evidence.append(f"Model {result.get('model_name', 'Unknown')}: {metrics or 'no metrics recorded'}." + (" Selected as best." if result.get("model_name") == best_model else ""))
    return evidence


def _dataset_evidence(
    question: str, df: Optional[pd.DataFrame], profile: Optional[Dict], results: Optional[List[Dict]],
    best_model: Optional[str], task: Optional[str], target: Optional[str], cleaning_actions: Optional[List[str]],
) -> tuple[List[str], List[str], List[str]]:
    evidence: List[str] = []
    sources: List[str] = []
    followups: List[str] = []
    q = question.lower()
    if df is None:
        return ["No dataset is currently loaded."], ["Dataset state"], ["How do I load a dataset?"]

    evidence.append(f"Dataset shape: {len(df):,} rows x {len(df.columns):,} columns.")
    visible_columns = list(map(str, df.columns[:100]))
    evidence.append("Columns: " + ", ".join(visible_columns) + (f" ... and {len(df.columns) - 100} more." if len(df.columns) > 100 else "."))
    sources.extend(["Dataset schema", "Live dataset profile"])

    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    row_count = max(len(df), 1)
    missing_text = ", ".join(f"{column}={int(count):,} ({count / row_count:.1%})" for column, count in missing.head(10).items())
    duplicates = int(df.duplicated().sum())
    health = profile.get("health_score") if profile else None
    evidence.append(f"Quality: health score {health if health is not None else 'n/a'}/100; {int(df.isna().sum().sum()):,} missing cells; {duplicates:,} duplicate rows. Missing by column: {missing_text or 'none' }.")
    if profile:
        risks = []
        if profile.get("constant_cols"):
            risks.append("constant columns=" + ", ".join(profile["constant_cols"][:10]))
        if profile.get("high_cardinality_cols"):
            risks.append("high-cardinality columns=" + ", ".join(profile["high_cardinality_cols"][:10]))
        if profile.get("id_cols"):
            risks.append("probable identifiers=" + ", ".join(profile["id_cols"][:10]))
        if risks:
            evidence.append("Pre-modeling risks: " + "; ".join(risks) + ".")

    mentioned = _mentioned_columns(question, df)
    if mentioned:
        evidence.extend(_column_evidence(df, mentioned))
        sources.extend(f"Column: {column}" for column in mentioned)
        followups.extend([f"What patterns stand out in {mentioned[0]}?", f"How should I prepare {mentioned[0]} for modeling?"])

    if target and target in df.columns:
        evidence.append(f"Modeling setup: task={task or 'not detected'}; target='{target}'.")
        evidence.extend(_column_evidence(df, [target]))
        if task == "classification":
            evidence.append(_class_balance(df, target))
            sources.append(f"Target distribution: {target}")
        followups.append(f"Which features are most related to {target}?")
    elif "imbalance" in q or "balance" in q:
        candidates = _candidate_class_columns(df)
        if candidates:
            evidence.append("No target is selected. Likely class columns checked: " + " ".join(_class_balance(df, column) for column in candidates))
            sources.append("Candidate class distributions")

    if task == "classification":
        evidence.append("Metric guidance: use weighted F1 as the default comparison metric; also inspect per-class recall and precision, especially when classes are imbalanced.")
    elif task == "regression":
        evidence.append("Metric guidance: compare RMSE and MAE (lower is better) alongside R2; RMSE penalizes large errors more strongly.")

    if any(term in q for term in ("correlation", "related", "relationship", "feature")):
        numeric = df.select_dtypes(include="number")
        if numeric.shape[1] >= 2:
            corr = numeric.corr().abs()
            pairs: List[tuple[float, str, str]] = []
            for left_index, left in enumerate(corr.columns):
                for right in corr.columns[left_index + 1:]:
                    value = corr.loc[left, right]
                    if pd.notna(value):
                        pairs.append((float(value), left, right))
            pairs.sort(reverse=True)
            evidence.append("Strongest absolute numeric correlations: " + ", ".join(f"{left} vs {right}={value:.3f}" for value, left, right in pairs[:8]) + ".")
            sources.append("Numeric correlation matrix")

    evidence.extend(_model_evidence(results, best_model, task))
    if results:
        sources.append("Model leaderboard")
        followups.append("Why was the best model selected?")
    if cleaning_actions:
        evidence.append("Cleaning already applied: " + "; ".join(cleaning_actions) + ".")
        sources.append("Cleaning history")

    if not followups:
        followups = ["Which columns need cleaning first?", "What are the strongest EDA signals?", "What should I do next?"]
    return evidence, list(dict.fromkeys(sources)), list(dict.fromkeys(followups))[:3]


def _fallback_answer(question: str, evidence: List[str], method_docs: List[Dict], task: Optional[str], target: Optional[str]) -> str:
    q = question.lower()
    selected = [evidence[0]]
    if any(term in q for term in ("missing", "clean", "duplicate", "quality", "risky")):
        selected = [line for line in evidence if line.startswith(("Quality:", "Pre-modeling risks:", "Column '", "Cleaning"))] or evidence[:3]
    elif any(term in q for term in ("model", "metric", "score", "best", "leaderboard")):
        selected = [line for line in evidence if line.startswith(("Model ", "Modeling setup", "Metric guidance:"))]
        if not selected:
            selected = [f"No trained model metrics are available yet. The current task is {task or 'not selected'}." ]
    elif "imbalance" in q or "balance" in q:
        selected = [line for line in evidence if "balance" in line.lower() or "Likely class" in line]
        if not selected:
            selected = ["Select a classification target so class balance can be calculated exactly."]
    elif any(term in q for term in ("correlation", "related", "relationship", "feature")):
        selected = [line for line in evidence if "correlation" in line.lower() or line.startswith("Column '")] or evidence[:3]
    elif any(_normalise(column) in _normalise(question) for line in evidence for column in re.findall(r"Column '([^']+)'", line)):
        selected = [line for line in evidence if line.startswith("Column '")]
    else:
        selected = [line for line in evidence if line.startswith(("Dataset shape", "Quality:", "Modeling setup"))]

    heading = "Here is what the current dataset shows:"
    answer = heading + "\n\n" + "\n".join(f"- {line}" for line in selected[:8])
    if method_docs:
        answer += f"\n\nMethod note: {method_docs[0]['content']}"
    if target is None and ("target" in q or "imbalance" in q):
        answer += "\n\nSelect the actual target on the Modeling page for a target-specific answer."
    return answer


def chat_with_dataset(
    question: str, df: Optional[pd.DataFrame], profile: Optional[Dict], results: Optional[List[Dict]],
    best_model: Optional[str], task: Optional[str], target: Optional[str], cleaning_actions: Optional[List[str]],
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Return a grounded answer plus evidence metadata for the UI."""
    evidence, sources, followups = _dataset_evidence(
        question, df, profile, results, best_model, task, target, cleaning_actions
    )
    method_docs = retrieve_documents(question, top_k=2)
    sources.extend(f"Knowledge: {document['title']}" for document in method_docs)
    method_context = "\n".join(f"[{doc['title']}] {doc['content']}" for doc in method_docs)
    history_context = "\n".join(
        f"{item.get('role', 'user')}: {item.get('content', '')[:500]}" for item in (history or [])[-6:]
    )
    full_context = (
        "VERIFIED DATASET EVIDENCE:\n- " + "\n- ".join(evidence) +
        (f"\n\nMETHOD KNOWLEDGE:\n{method_context}" if method_context else "") +
        (f"\n\nRECENT CONVERSATION:\n{history_context}" if history_context else "")
    )
    answer = answer_question(question, full_context)
    used_fallback = answer.startswith(ERROR_PREFIXES)
    if used_fallback:
        answer = _fallback_answer(question, evidence, method_docs, task, target)

    confidence = "high" if df is not None and any(source.startswith(("Column:", "Target distribution", "Model leaderboard", "Numeric correlation")) for source in sources) else "medium"
    return {
        "answer": answer,
        "sources": list(dict.fromkeys(sources))[:8],
        "confidence": confidence,
        "follow_up_questions": followups,
        "mode": "local-grounded" if used_fallback else "ai-grounded",
    }
