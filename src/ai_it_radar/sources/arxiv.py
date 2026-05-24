"""arXiv adapter — uses the official `arxiv` PyPI library (fronts arXiv's API)."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Iterable

import arxiv

from ..schemas import Candidate, CandidateKind, SourceKind
from .base import SourceAdapter

log = logging.getLogger(__name__)


class ArxivSource(SourceAdapter):
    name = "arxiv"

    def fetch(self) -> Iterable[Candidate]:
        cfg = self.config
        if not cfg.get("enabled", True):
            return []
        cats: list[str] = cfg.get("categories", []) or []
        keywords: list[str] = cfg.get("keywords", []) or []
        max_per_run: int = int(cfg.get("max_per_run", 30))
        lookback_days: int = int(cfg.get("lookback_days", 7))

        # Build a query: (cat:cs.AI OR cat:cs.CL) AND (abs:"agent" OR abs:"reasoning" OR ...)
        cat_clause = " OR ".join(f"cat:{c}" for c in cats) if cats else ""
        kw_clause = " OR ".join(f'abs:"{k}"' for k in keywords) if keywords else ""
        query_parts = [p for p in (cat_clause, kw_clause) if p]
        query = " AND ".join(f"({p})" for p in query_parts) if query_parts else "cat:cs.AI"

        client = arxiv.Client(page_size=min(max_per_run, 50), delay_seconds=3, num_retries=3)
        search = arxiv.Search(
            query=query,
            max_results=max_per_run,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)
        out: list[Candidate] = []
        try:
            for result in client.results(search):
                published = result.published.replace(tzinfo=None) if result.published else None
                if published and published < cutoff:
                    continue
                paper_id = result.entry_id.rsplit("/", 1)[-1]  # e.g. 2501.12345v1
                base_id = paper_id.split("v", 1)[0]
                uid = f"arxiv:{base_id}"
                cand = Candidate(
                    uid=uid,
                    source=SourceKind.ARXIV,
                    kind=CandidateKind.PAPER,
                    title=result.title.strip(),
                    url=result.entry_id,
                    summary=(result.summary or "").strip()[:1500],
                    content=(result.summary or "").strip(),
                    authors=[a.name for a in result.authors],
                    published_at=published,
                    metadata={
                        "primary_category": result.primary_category,
                        "categories": list(result.categories or []),
                        "pdf_url": result.pdf_url,
                        "doi": result.doi,
                    },
                )
                out.append(cand)
        except Exception as e:
            log.warning("arXiv fetch failed: %s", e)
        return out
