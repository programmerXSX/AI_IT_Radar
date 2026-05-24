"""Short-term working memory: LangGraph SQLite checkpointer.

A cycle's intermediate state (candidates, triage, analyses, scores) is persisted on
each node transition so a crashed cycle can be resumed without re-fetching sources.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator

from langgraph.checkpoint.sqlite import SqliteSaver

from ..settings import get_settings


@contextmanager
def build_checkpointer() -> Iterator[SqliteSaver]:
    """Yield a LangGraph SqliteSaver bound to the configured checkpoint DB.

    Use as a context manager so the underlying connection is closed cleanly.
    """
    settings = get_settings()
    settings.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(settings.checkpoint_path), check_same_thread=False)
    try:
        saver = SqliteSaver(conn)
        yield saver
    finally:
        conn.close()
