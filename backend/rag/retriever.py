"""Small, dependency-free retriever for DataStory's method knowledge base."""

import math
import re
from collections import Counter
from typing import Dict, List

from backend.rag.knowledge_base import DOCUMENTS


STOP_WORDS = {
    "a", "about", "and", "are", "do", "does", "for", "from", "have", "how",
    "column", "data", "dataset", "describe", "explain", "i", "in", "is", "it", "me",
    "my", "of", "on", "or", "should", "tell", "the", "this", "to", "what", "when",
    "which", "with", "would", "you",
}


def _tokens(text: str) -> List[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in STOP_WORDS]


def _document_frequencies() -> Counter:
    frequencies: Counter = Counter()
    for document in DOCUMENTS:
        frequencies.update(set(_tokens(f"{document['title']} {document['content']}")))
    return frequencies


DOCUMENT_FREQUENCIES = _document_frequencies()


def _keyword_score(query: str, document: Dict) -> float:
    """Weighted lexical score with title and exact-phrase boosts."""
    query_tokens = _tokens(query)
    if not query_tokens:
        return 0.0

    title = document["title"].lower()
    content = document["content"].lower()
    doc_counts = Counter(_tokens(f"{title} {content}"))
    score = 0.0
    for token in set(query_tokens):
        if token not in doc_counts:
            continue
        rarity = math.log((len(DOCUMENTS) + 1) / (DOCUMENT_FREQUENCIES[token] + 1)) + 1
        score += rarity * (2.3 if token in _tokens(title) else 1.0)

    query_phrase = " ".join(query_tokens)
    if len(query_tokens) > 1 and query_phrase in f"{title} {content}":
        score += 3.0
    return score / math.sqrt(max(len(set(query_tokens)), 1))


def retrieve_documents(query: str, top_k: int = 3, min_score: float = 0.65) -> List[Dict]:
    """Return ranked knowledge documents with transparent relevance scores."""
    if not query.strip():
        return []
    ranked = [
        {**document, "score": round(_keyword_score(query, document), 3)}
        for document in DOCUMENTS
    ]
    ranked.sort(key=lambda document: document["score"], reverse=True)
    return [document for document in ranked[:top_k] if document["score"] >= min_score]


def retrieve_context(query: str, top_k: int = 3) -> str:
    documents = retrieve_documents(query, top_k=top_k)
    return "\n\n".join(f"[{doc['title']}]\n{doc['content']}" for doc in documents)


def build_faiss_index():
    """Retained for CLI compatibility; live retrieval intentionally stays local and fast."""
    print("DataStory uses its deterministic weighted knowledge retriever; no index build is required.")
