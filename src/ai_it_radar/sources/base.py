"""Source adapter contract."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable

from ..schemas import Candidate


class SourceAdapter(ABC):
    """Each external feed (arXiv / GitHub / HF / ...) implements this."""

    name: str = ""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def fetch(self) -> Iterable[Candidate]:
        """Yield normalized Candidates. Adapters MUST be tolerant of partial failures."""
