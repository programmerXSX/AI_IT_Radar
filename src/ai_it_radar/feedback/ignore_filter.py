"""IgnoreFilter — derive a 'do-not-show-again' signal from past Ignore feedback.

When the decision-maker clicks **Ignore** on a candidate in the report UI, we record
that signal in SQLite. This module aggregates all such signals into a single negative
centroid in embedding space; the Triage agent then drops any new candidate whose
similarity to that centroid exceeds a threshold — BEFORE wasting LLM calls on it.

Design choices:
- Whole-cluster centroid (one cosine compare per candidate) instead of per-item NN
  search — same complexity as positive profile match, very cheap.
- Embeddings are reused from Chroma when possible (a candidate that's been
  fed back necessarily lives in KB), falling back to re-embedding if a vector is
  somehow missing.
- A candidate that was first Ignored and later Adopted/Watched is excluded from the
  negative pool — the latest positive signal overrides.
- Cold-start safe: if no Ignore feedback exists, score() always returns 0.
"""
from __future__ import annotations

import logging
from datetime import datetime

import numpy as np

from ..memory.kb import KnowledgeBase
from ..rag.embedder import Embedder
from ..rag.indexer import embedding_text
from ..schemas import FeedbackTag

log = logging.getLogger(__name__)


class IgnoreFilter:
    """Build a negative-interest centroid from past Ignore feedback."""

    def __init__(self, kb: KnowledgeBase | None = None) -> None:
        self._kb = kb or KnowledgeBase()
        self._centroid: list[float] | None = None
        self._uid_set: set[str] = set()
        self._loaded = False

    # ---- Loading & centroid construction ---------------------------------------------

    def _load(self, embedder: Embedder) -> None:
        if self._loaded:
            return

        # All-time feedback (we want long-term ignore signals, not just last 30d).
        all_fb = self._kb.feedback_since(datetime(1970, 1, 1))
        ignore_uids = {f.candidate_uid for f in all_fb if f.tag == FeedbackTag.IGNORE}
        positive_uids = {
            f.candidate_uid
            for f in all_fb
            if f.tag in (FeedbackTag.ADOPT, FeedbackTag.WATCH)
        }
        # An item that's been later promoted (adopt/watch) overrides the earlier ignore.
        self._uid_set = ignore_uids - positive_uids

        if not self._uid_set:
            self._loaded = True
            log.info("IgnoreFilter: no ignored items, filter inactive")
            return

        vecs = self._fetch_embeddings(list(self._uid_set), embedder)
        if not vecs:
            log.warning("IgnoreFilter: ignore_uids present but no embeddings recoverable")
            self._loaded = True
            return

        arr = np.array(vecs)
        centroid = arr.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        self._centroid = centroid.tolist()
        self._loaded = True
        log.info(
            "IgnoreFilter loaded: %d ignored items, centroid dim=%d",
            len(self._uid_set), len(self._centroid),
        )

    def _fetch_embeddings(self, uids: list[str], embedder: Embedder) -> list[list[float]]:
        """Try Chroma first (no embed cost); fall back to re-embedding from KB text."""
        vecs: list[list[float]] = []
        try:
            res = self._kb._collection.get(ids=uids, include=["embeddings"])
            chroma_ids = res.get("ids") or []
            chroma_embs = res.get("embeddings") or []
            id_to_vec = {uid: emb for uid, emb in zip(chroma_ids, chroma_embs) if emb is not None}
        except Exception as e:
            log.warning("IgnoreFilter: chroma fetch failed (%s); will re-embed", e)
            id_to_vec = {}

        missing: list[str] = []
        for uid in uids:
            v = id_to_vec.get(uid)
            if v is not None:
                vecs.append(list(v))
            else:
                missing.append(uid)

        if missing:
            texts: list[str] = []
            for uid in missing:
                cand = self._kb.get_candidate(uid)
                if cand:
                    texts.append(embedding_text(cand))
            if texts:
                vecs.extend(embedder.embed_documents(texts))

        return vecs

    # ---- Public API ------------------------------------------------------------------

    def score(self, candidate_vec: list[float], embedder: Embedder) -> float:
        """Cosine similarity between candidate and ignore centroid (0 if filter inactive)."""
        self._load(embedder)
        if self._centroid is None:
            return 0.0
        a = np.array(candidate_vec)
        b = np.array(self._centroid)
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
        return float(np.dot(a, b) / denom)

    def is_active(self, embedder: Embedder | None = None) -> bool:
        if embedder is not None:
            self._load(embedder)
        return self._centroid is not None

    @property
    def ignore_count(self) -> int:
        return len(self._uid_set)

    @property
    def ignored_uids(self) -> set[str]:
        return frozenset(self._uid_set)
