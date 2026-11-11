"""Dataset assistant: grounded answers with or without an external LLM."""
from __future__ import annotations

import re
from typing import Any

from ai import llm, prompts
from ai.retrieval import Retriever, build_documents
from core import insights as insight_engine

CANNOT_DETERMINE = (
    "I can't determine that from the analysis computed so far. I can answer questions about the dataset's "
    "shape, column statistics, missing values, duplicates, outliers, correlations, class balance, cleaning "
    "steps, trained models, evaluation metrics and feature importance."
)
ROW_LEVEL = (
    "I don't have access to individual rows — only to the computed summaries (profile, quality metrics, "
    "correlations, model results). Use the preview on the Dataset page to inspect specific records."
)

# keyword → analysis document that must be included when the keyword appears
_ROUTES = [
    (r"missing|null|nan|empty|incomplete", "missing"),
    (r"duplicat|repeat", "duplicates"),
    (r"quality|issue|problem|health|valid|consisten|complete|unique", "quality"),
    (r"imbalanc|balanc|class distribution|skewed target|minority|majority", "balance"),
    (r"correlat|related|relationship|associat", "correlations"),
    (r"outlier|extreme|anomal", "outliers"),
    (r"best model|which model|perform|leaderboard|accura|score|evaluat|metric|f1|recall|precision|auc|rmse|mae|r2|r²", "leaderboard"),
    (r"import|influenc|driver|matter|explain|contribut", "importance"),
    (r"clean|fix|prepar|impute|what should i do|next step", "cleaning_suggestions"),
    (r"applied|history|changed|transform", "cleaning_history"),
    (r"how many rows|how many columns|shape|size|overview|describe|summary|synthetic|real|source|provenance", "overview"),
    (r"split|leak|methodolog|preprocess|how .* trained|pipeline|smote", "model_setup"),
    (r"identifier|constant|\bid\b", "structure"),
]
_ROW_QUESTION = re.compile(r"\brow\s*#?\d+|\b(first|last|specific|individual)\s+(row|record)|show me (the )?(rows|records)|which (row|record|passenger|customer)s?\b", re.I)


_DEFINITIONAL = re.compile(r"what (does|do|is|are) .*(mean|measure)|\bmeaning\b|\bdefine|definition|how (do i|to) (read|interpret)|what is (a |an |the )?(smote|cross|roc|auc|f1|precision|recall|accuracy|rmse|mae|mse|r2|r²|permutation|leakage|confusion)", re.I)


def _select(question: str, retriever: Retriever, task: str | None = None) -> list[dict[str, Any]]:
    """Pick supporting documents: keyword-routed analysis facts first, then the closest retrieved ones."""
    q = question.lower()
    by_id = {d["id"]: d for d in retriever.docs}
    hits = retriever.search(question, k=6)
    definitional = bool(_DEFINITIONAL.search(q))
    routed = [by_id[i] for pat, i in _ROUTES if re.search(pat, q) and i in by_id]
    routed = list({d["id"]: d for d in routed}.values())[:2]
    analysis_hits = [(d, s) for d, s in hits if d["kind"] == "analysis" and d not in routed]
    concept_hits = [(d, s) for d, s in hits if d["kind"] == "concept"]
    analysis = list(routed)
    if analysis_hits and len(analysis) < 2:
        top_score = analysis_hits[0][1]
        floor = 0.12 if routed else 0.05
        analysis += [d for d, s in analysis_hits[: 2 - len(analysis)] if s >= max(floor, 0.6 * top_score)]
    concepts = [d for d, s in concept_hits[:1] if s >= (0.05 if definitional else 0.12)]
    if definitional and re.search(r"metric|score", q):
        ids = {"classification": ["concept_precision_recall", "concept_accuracy", "concept_roc_auc"],
               "regression": ["concept_regression"]}.get(task or "", ["concept_precision_recall", "concept_regression"])
        return [by_id[i] for i in ids] + analysis[:1]
    if definitional and concepts:
        return concepts + analysis[:1]
    return analysis + concepts


def answer(question: str, results: dict[str, Any], use_llm: bool = True) -> dict[str, Any]:
    """Return {"answer", "sources", "mode", "note"}; never raises for ordinary failures."""
    question = (question or "").strip()
    if not question:
        return {"answer": "Ask a question about the active dataset or its analysis.", "sources": [], "mode": "deterministic", "note": None}
    if _ROW_QUESTION.search(question):
        return {"answer": ROW_LEVEL, "sources": [], "mode": "deterministic", "note": None}
    retriever = Retriever(build_documents(results))
    docs = _select(question, retriever, (results.get("modeling") or {}).get("task"))
    if not docs:
        return {"answer": CANNOT_DETERMINE, "sources": [], "mode": "deterministic", "note": None}
    sources = [f"{d['title']} — {d['source']}" for d in docs]
    facts = "\n".join(f"[{i}] {d['title']}: {d['text']}" for i, d in enumerate(docs, 1))
    deterministic = _compose(docs)
    note = None
    cfg = llm.get_config() if use_llm else None
    if cfg is not None:
        try:
            text = llm.chat([
                {"role": "system", "content": prompts.ASSISTANT_SYSTEM},
                {"role": "user", "content": f"{prompts.facts_block(facts)}\n\nQUESTION: {question}"},
            ], config=cfg)
            bad = llm.unverified_numbers(text, facts + " " + question)
            if not bad:
                return {"answer": text, "sources": sources, "mode": "llm", "note": None}
            note = ("The language model's draft contained figures that could not be traced to the computed "
                    f"analysis ({', '.join(bad[:5])}), so the verified facts are shown instead.")
        except llm.LLMError as exc:
            note = f"{exc} Showing the verified facts instead."
    return {"answer": deterministic, "sources": sources, "mode": "deterministic", "note": note}


def _compose(docs: list[dict[str, Any]]) -> str:
    return "\n\n".join(f"**{d['title']}.** {d['text']}" for d in docs)


def polished_narrative(results: dict[str, Any]) -> dict[str, Any]:
    """Narrative for the insights page: LLM-polished when configured and verifiable, deterministic otherwise."""
    ins = results["insights"]
    base = insight_engine.narrative(ins)
    cfg = llm.get_config()
    if cfg is None:
        return {"text": base, "mode": "deterministic", "note": None}
    facts = insight_engine.facts_text(ins)
    try:
        text = llm.chat([
            {"role": "system", "content": prompts.NARRATIVE_SYSTEM},
            {"role": "user", "content": prompts.facts_block(facts)},
        ], max_tokens=600, config=cfg)
    except llm.LLMError as exc:
        return {"text": base, "mode": "deterministic", "note": f"{exc} Showing the rule-based narrative."}
    bad = llm.unverified_numbers(text, facts)
    if bad:
        return {"text": base, "mode": "deterministic",
                "note": f"The LLM draft was discarded because it contained unverifiable figures ({', '.join(bad[:5])})."}
    return {"text": text, "mode": "llm", "note": None}
