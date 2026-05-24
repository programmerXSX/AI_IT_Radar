"""Long-term knowledge base.

Two stores back-to-back:
- SQLite: structured metadata, evaluation history, feedback (queryable, joinable).
- ChromaDB: dense vector index of candidate text for similarity search and dedup.

KnowledgeBase encapsulates both behind a single ergonomic interface.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator

import chromadb
from chromadb.config import Settings as ChromaSettings

from ..schemas import Analysis, Candidate, Feedback, Score
from ..settings import get_settings

# ---- SQLite schema -----------------------------------------------------------------------

DDL = [
    """
    CREATE TABLE IF NOT EXISTS candidates (
        uid TEXT PRIMARY KEY,
        source TEXT NOT NULL,
        kind TEXT NOT NULL,
        title TEXT NOT NULL,
        url TEXT NOT NULL,
        summary TEXT,
        content TEXT,
        authors TEXT,
        published_at TEXT,
        fetched_at TEXT NOT NULL,
        metadata TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS analyses (
        candidate_uid TEXT PRIMARY KEY,
        payload TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (candidate_uid) REFERENCES candidates(uid)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS scores (
        candidate_uid TEXT NOT NULL,
        cycle_id TEXT NOT NULL,
        payload TEXT NOT NULL,
        evaluated_at TEXT NOT NULL,
        PRIMARY KEY (candidate_uid, cycle_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        candidate_uid TEXT NOT NULL,
        tag TEXT NOT NULL,
        note TEXT,
        user TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (candidate_uid) REFERENCES candidates(uid)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS eval_traces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        candidate_uid TEXT NOT NULL,
        cycle_id TEXT NOT NULL,
        dimension_id TEXT NOT NULL,
        prompt TEXT,
        raw_response TEXT,
        critic_response TEXT,
        neighbors TEXT,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cycles (
        id TEXT PRIMARY KEY,
        started_at TEXT NOT NULL,
        finished_at TEXT,
        status TEXT NOT NULL
    )
    """,
]


class KnowledgeBase:
    """Combined SQLite + ChromaDB store for the radar."""

    COLLECTION = "radar_kb"

    def __init__(self) -> None:
        s = get_settings()
        s.ensure_dirs()
        self._sqlite_path = str(s.sqlite_path)
        self._chroma = chromadb.PersistentClient(
            path=str(s.chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        # Use cosine distance: with normalized embeddings, distance ranges 0 (identical)
        # to 2 (opposite). The triage threshold below is interpreted in that space.
        self._collection = self._chroma.get_or_create_collection(
            name=self.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._init_schema()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._sqlite_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as c:
            for stmt in DDL:
                c.execute(stmt)

    # ---- Candidates -------------------------------------------------------------------

    def upsert_candidate(self, c: Candidate, embedding: list[float] | None = None) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO candidates
                   (uid, source, kind, title, url, summary, content, authors,
                    published_at, fetched_at, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    c.uid,
                    c.source.value,
                    c.kind.value,
                    c.title,
                    c.url,
                    c.summary,
                    c.content,
                    json.dumps(c.authors, ensure_ascii=False),
                    c.published_at.isoformat() if c.published_at else None,
                    c.fetched_at.isoformat(),
                    json.dumps(c.metadata, ensure_ascii=False, default=str),
                ),
            )
        if embedding is not None:
            self._collection.upsert(
                ids=[c.uid],
                embeddings=[embedding],
                documents=[self._candidate_doc(c)],
                metadatas=[{
                    "source": c.source.value,
                    "kind": c.kind.value,
                    "title": c.title,
                    "url": c.url,
                }],
            )

    @staticmethod
    def _candidate_doc(c: Candidate) -> str:
        # Keep the doc compact; full content stays in SQLite.
        return f"{c.title}\n\n{c.summary}\n\n{c.content[:2000]}"

    def get_candidate(self, uid: str) -> Candidate | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM candidates WHERE uid = ?", (uid,)).fetchone()
        if not row:
            return None
        return _row_to_candidate(row)

    def has_candidate(self, uid: str) -> bool:
        with self._conn() as conn:
            r = conn.execute("SELECT 1 FROM candidates WHERE uid = ?", (uid,)).fetchone()
        return r is not None

    # ---- Vector search ----------------------------------------------------------------

    def query_similar(
        self,
        embedding: list[float],
        *,
        k: int = 5,
        exclude_uids: list[str] | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Return [(uid, distance, metadata), ...] sorted ascending by distance.

        Distance follows ChromaDB convention (default cosine: 0 == identical, 2 == opposite).
        """
        # Ask Chroma for extra in case some are excluded.
        n = max(k, k + len(exclude_uids or []))
        res = self._collection.query(query_embeddings=[embedding], n_results=n)
        out: list[tuple[str, float, dict[str, Any]]] = []
        ids = res.get("ids", [[]])[0]
        dists = res.get("distances", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        for uid, dist, meta in zip(ids, dists, metas):
            if exclude_uids and uid in exclude_uids:
                continue
            out.append((uid, float(dist), meta or {}))
            if len(out) >= k:
                break
        return out

    # ---- Analyses ---------------------------------------------------------------------

    def upsert_analysis(self, a: Analysis) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO analyses (candidate_uid, payload, created_at)
                   VALUES (?, ?, ?)""",
                (a.candidate_uid, a.model_dump_json(), datetime.utcnow().isoformat()),
            )

    def get_analysis(self, uid: str) -> Analysis | None:
        with self._conn() as conn:
            r = conn.execute("SELECT payload FROM analyses WHERE candidate_uid = ?", (uid,)).fetchone()
        return Analysis.model_validate_json(r["payload"]) if r else None

    # ---- Scores -----------------------------------------------------------------------

    def upsert_score(self, s: Score, cycle_id: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO scores
                   (candidate_uid, cycle_id, payload, evaluated_at)
                   VALUES (?, ?, ?, ?)""",
                (s.candidate_uid, cycle_id, s.model_dump_json(), s.evaluated_at.isoformat()),
            )

    def latest_score(self, uid: str) -> Score | None:
        with self._conn() as conn:
            r = conn.execute(
                """SELECT payload FROM scores WHERE candidate_uid = ?
                   ORDER BY evaluated_at DESC LIMIT 1""",
                (uid,),
            ).fetchone()
        return Score.model_validate_json(r["payload"]) if r else None

    def scores_in_period(self, start: datetime, end: datetime) -> list[Score]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT payload FROM scores
                   WHERE evaluated_at >= ? AND evaluated_at <= ?
                   ORDER BY evaluated_at DESC""",
                (start.isoformat(), end.isoformat()),
            ).fetchall()
        return [Score.model_validate_json(r["payload"]) for r in rows]

    # ---- Feedback ---------------------------------------------------------------------

    def record_feedback(self, fb: Feedback) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO feedback (candidate_uid, tag, note, user, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (fb.candidate_uid, fb.tag.value, fb.note, fb.user, fb.created_at.isoformat()),
            )

    def feedback_for(self, uid: str) -> list[Feedback]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM feedback WHERE candidate_uid = ? ORDER BY created_at DESC",
                (uid,),
            ).fetchall()
        from ..schemas import FeedbackTag
        return [
            Feedback(
                candidate_uid=r["candidate_uid"],
                tag=FeedbackTag(r["tag"]),
                note=r["note"] or "",
                user=r["user"] or "anonymous",
                created_at=datetime.fromisoformat(r["created_at"]),
            )
            for r in rows
        ]

    def feedback_since(self, since: datetime) -> list[Feedback]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM feedback WHERE created_at >= ?",
                (since.isoformat(),),
            ).fetchall()
        from ..schemas import FeedbackTag
        return [
            Feedback(
                candidate_uid=r["candidate_uid"],
                tag=FeedbackTag(r["tag"]),
                note=r["note"] or "",
                user=r["user"] or "anonymous",
                created_at=datetime.fromisoformat(r["created_at"]),
            )
            for r in rows
        ]

    # ---- Eval traces (observability) -------------------------------------------------

    def record_eval_trace(
        self,
        *,
        candidate_uid: str,
        cycle_id: str,
        dimension_id: str,
        prompt: str,
        raw_response: str,
        critic_response: str = "",
        neighbors: list[dict[str, Any]] | None = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO eval_traces
                   (candidate_uid, cycle_id, dimension_id, prompt, raw_response,
                    critic_response, neighbors, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    candidate_uid,
                    cycle_id,
                    dimension_id,
                    prompt,
                    raw_response,
                    critic_response,
                    json.dumps(neighbors or [], ensure_ascii=False, default=str),
                    datetime.utcnow().isoformat(),
                ),
            )

    # ---- Cycles -----------------------------------------------------------------------

    def start_cycle(self, cycle_id: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO cycles (id, started_at, status)
                   VALUES (?, ?, 'running')""",
                (cycle_id, datetime.utcnow().isoformat()),
            )

    def finish_cycle(self, cycle_id: str, status: str = "ok") -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE cycles SET finished_at = ?, status = ? WHERE id = ?",
                (datetime.utcnow().isoformat(), status, cycle_id),
            )


def _row_to_candidate(row: sqlite3.Row) -> Candidate:
    from ..schemas import CandidateKind, SourceKind

    return Candidate(
        uid=row["uid"],
        source=SourceKind(row["source"]),
        kind=CandidateKind(row["kind"]),
        title=row["title"],
        url=row["url"],
        summary=row["summary"] or "",
        content=row["content"] or "",
        authors=json.loads(row["authors"]) if row["authors"] else [],
        published_at=datetime.fromisoformat(row["published_at"]) if row["published_at"] else None,
        fetched_at=datetime.fromisoformat(row["fetched_at"]),
        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
    )
