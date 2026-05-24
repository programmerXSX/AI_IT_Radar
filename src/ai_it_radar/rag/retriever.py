"""Retriever — wraps KB vector search with reranker hooks.

Three RAG call sites use this:
1. TriageAgent — dedup (high-similarity hits become duplicate candidates).
2. AnalystAgent — analogy retrieval ("compare candidate to nearest known repo").
3. EvaluatorAgent — anchor scores ("here's how we previously scored similar items").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..memory.kb import KnowledgeBase
from ..schemas import Candidate
from .embedder import Embedder
from .reranker import LLMReranker


@dataclass
class Neighbor:
    uid: str
    title: str
    kind: str
    distance: float
    metadata: dict[str, Any]


class Retriever:
    def __init__(
        self,
        kb: KnowledgeBase,
        embedder: Embedder,
        *,
        reranker: LLMReranker | None = None,
    ) -> None:
        self._kb = kb
        self._embedder = embedder
        self._reranker = reranker

    def neighbors(
        self,
        candidate: Candidate,
        *,
        k: int = 5,
        exclude_self: bool = True,
        embedding: list[float] | None = None,
    ) -> list[Neighbor]:
        if embedding is None:
            embedding = self._embedder.embed_query(_query_text(candidate))
        exclude = [candidate.uid] if exclude_self else []
        raw = self._kb.query_similar(embedding, k=k, exclude_uids=exclude)
        return [
            Neighbor(uid=u, title=meta.get("title", ""), kind=meta.get("kind", ""),
                     distance=d, metadata=meta)
            for (u, d, meta) in raw
        ]

    def rerank_for_dedup(
        self,
        candidate: Candidate,
        neighbors: list[Neighbor],
    ) -> tuple[Neighbor, str] | None:
        """Run an LLM second-pass: same paper / same project / different.

        Returns (matched_neighbor, reason) if any neighbor is judged to be a duplicate.
        """
        if not self._reranker or not neighbors:
            return None
        return self._reranker.judge_duplicates(candidate, neighbors, kb=self._kb)


def _query_text(c: Candidate) -> str:
    return f"{c.title}\n{c.summary}\n{c.content[:800]}"
