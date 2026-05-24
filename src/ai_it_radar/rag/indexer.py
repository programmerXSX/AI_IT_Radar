"""Index candidates into the KB (SQLite + ChromaDB)."""
from __future__ import annotations

from ..memory.kb import KnowledgeBase
from ..schemas import Candidate
from .embedder import Embedder


def index_candidate(c: Candidate, kb: KnowledgeBase, embedder: Embedder) -> list[float]:
    """Compute the candidate's embedding, persist both metadata and vector.

    Returns the embedding for downstream reuse (e.g. dedup queries in the same step).
    """
    text = embedding_text(c)
    vec = embedder.embed_documents([text])[0]
    kb.upsert_candidate(c, embedding=vec)
    return vec


def embedding_text(c: Candidate) -> str:
    """Canonical text used to embed a Candidate. Stable across call sites."""
    parts = [c.title, c.summary]
    if c.content:
        parts.append(c.content[:1500])
    return "\n".join(p for p in parts if p)
