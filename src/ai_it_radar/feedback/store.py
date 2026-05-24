"""Thin facade over KnowledgeBase for feedback-related ops."""
from __future__ import annotations

from datetime import datetime, timedelta

from ..memory.kb import KnowledgeBase
from ..schemas import Feedback


class FeedbackStore:
    def __init__(self, kb: KnowledgeBase | None = None) -> None:
        self._kb = kb or KnowledgeBase()

    def record(self, fb: Feedback) -> None:
        self._kb.record_feedback(fb)

    def for_candidate(self, uid: str) -> list[Feedback]:
        return self._kb.feedback_for(uid)

    def since(self, days: int) -> list[Feedback]:
        return self._kb.feedback_since(datetime.utcnow() - timedelta(days=days))
