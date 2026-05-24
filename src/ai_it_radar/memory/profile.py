"""Lab interest profile — value/preference memory.

The profile is two things:
1. A weighted set of topics (with keywords) loaded from `config/lab_profile.yaml`.
2. A computed embedding centroid that represents the Lab's current interest in vector space.

The auto-learned section is updated by `feedback.profile_updater`; anchors are immutable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..settings import load_lab_profile, save_lab_profile


@dataclass
class TopicEntry:
    topic: str
    weight: float
    keywords: list[str]
    is_anchor: bool


class LabProfile:
    """In-memory view of the Lab profile, plus an interest centroid."""

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self._data = data or load_lab_profile()
        self._centroid: list[float] | None = None
        self._embed_provider = None  # lazy

    # ---- Topics -----------------------------------------------------------------------

    def anchors(self) -> list[TopicEntry]:
        return [
            TopicEntry(a["topic"], float(a.get("weight", 1.0)), list(a.get("keywords", [])), True)
            for a in self._data.get("anchors", [])
        ]

    def auto_learned(self) -> list[TopicEntry]:
        return [
            TopicEntry(a["topic"], float(a.get("weight", 0.5)), list(a.get("keywords", [])), False)
            for a in self._data.get("auto_learned", [])
        ]

    def all_topics(self) -> list[TopicEntry]:
        return self.anchors() + self.auto_learned()

    def exclude_keywords(self) -> list[str]:
        return [k.lower() for k in self._data.get("exclude_keywords", [])]

    @property
    def name(self) -> str:
        return self._data.get("name", "Lab")

    @property
    def description(self) -> str:
        return self._data.get("description", "")

    def summary_text(self) -> str:
        return f"{self.name}\n\n{self.description}".strip()

    def raw(self) -> dict[str, Any]:
        return self._data

    def replace_auto_learned(self, topics: list[dict[str, Any]]) -> None:
        """Replace the auto_learned section. Anchors are not touched."""
        self._data["auto_learned"] = topics
        save_lab_profile(self._data)
        self._centroid = None  # force recompute

    # ---- Filtering helpers ------------------------------------------------------------

    def has_excluded_token(self, *texts: str) -> bool:
        joined = " ".join(t.lower() for t in texts if t)
        return any(tok in joined for tok in self.exclude_keywords())

    # ---- Centroid (vector representation of interests) -------------------------------

    def centroid(self, embedder) -> list[float]:
        """Compute (and cache) the weighted centroid of all topics in embedding space.

        `embedder` is the project Embedder; we accept it as a parameter rather than import
        it directly to avoid a circular import.
        """
        if self._centroid is not None:
            return self._centroid

        topics = self.all_topics()
        if not topics:
            return []

        sentences: list[str] = []
        weights: list[float] = []
        for t in topics:
            txt = t.topic + ". " + ", ".join(t.keywords)
            sentences.append(txt)
            weights.append(t.weight)

        vecs = embedder.embed_documents(sentences)
        arr = np.array(vecs)
        w = np.array(weights).reshape(-1, 1)
        centroid = (arr * w).sum(axis=0) / max(w.sum(), 1e-6)
        # normalize
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        self._centroid = centroid.tolist()
        return self._centroid


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    av = np.array(a)
    bv = np.array(b)
    denom = (np.linalg.norm(av) * np.linalg.norm(bv)) or 1e-9
    return float(np.dot(av, bv) / denom)
