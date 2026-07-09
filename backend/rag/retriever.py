"""
DataStory AI — Simple keyword-based RAG retriever.
Falls back to FAISS vector search when available.
"""
import re
from typing import List, Optional
from backend.rag.knowledge_base import DOCUMENTS


def _keyword_score(query: str, doc: dict) -> float:
    """Score a document by keyword overlap with the query."""
    query_tokens = set(re.findall(r"\w+", query.lower()))
    content_tokens = set(re.findall(r"\w+", (doc["title"] + " " + doc["content"]).lower()))
    overlap = query_tokens & content_tokens
    return len(overlap) / max(len(query_tokens), 1)


def retrieve_context(query: str, top_k: int = 3) -> str:
    """
    Retrieve top-k relevant documents from the knowledge base.
    Returns them as a single formatted string for use as RAG context.
    """
    if not query.strip():
        return ""

    # Try FAISS first (optional, degrades gracefully)
    try:
        return _faiss_retrieve(query, top_k)
    except Exception:
        pass

    # Fallback: keyword matching
    scored = [(doc, _keyword_score(query, doc)) for doc in DOCUMENTS]
    scored.sort(key=lambda x: x[1], reverse=True)
    top_docs = [d for d, score in scored[:top_k] if score > 0]

    if not top_docs:
        return ""

    parts = []
    for doc in top_docs:
        parts.append(f"[{doc['title']}]\n{doc['content']}")
    return "\n\n".join(parts)


def _faiss_retrieve(query: str, top_k: int) -> str:
    """FAISS-based retrieval (requires faiss-cpu and sentence-transformers)."""
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    corpus = [d["title"] + ". " + d["content"] for d in DOCUMENTS]
    corpus_emb = model.encode(corpus, convert_to_numpy=True)

    dim = corpus_emb.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(corpus_emb.astype("float32"))

    query_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_emb, top_k)

    parts = []
    for idx in indices[0]:
        if idx < len(DOCUMENTS):
            doc = DOCUMENTS[idx]
            parts.append(f"[{doc['title']}]\n{doc['content']}")
    return "\n\n".join(parts)


def build_faiss_index():
    """Pre-build and cache FAISS index (optional setup step)."""
    try:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        corpus = [d["title"] + ". " + d["content"] for d in DOCUMENTS]
        embeddings = model.encode(corpus, convert_to_numpy=True).astype("float32")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        faiss.write_index(index, "datastory_rag.index")
        print(f"FAISS index built with {len(DOCUMENTS)} documents.")
    except ImportError:
        print("FAISS or sentence-transformers not installed. Keyword fallback will be used.")
