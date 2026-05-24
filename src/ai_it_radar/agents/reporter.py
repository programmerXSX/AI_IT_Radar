"""ReporterAgent — assemble the period's accumulated scores into a graded Report.

The report is a *period* summary, NOT a per-cycle delta:
- It pulls every score evaluated in the last `report_period_days` (default 7) from KB
  so that re-runs within a week still produce a complete report (the previous cycle's
  scored items remain visible even when this cycle had nothing new — e.g. when arXiv
  returned the same papers and Triage flagged them as duplicates).
- It dedupes by candidate_uid keeping the latest score, so re-evaluations don't
  double-count.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from ..memory.kb import KnowledgeBase
from ..reporter.render import render_report
from ..schemas import GraphState, Report, ReportItem, Score
from ..settings import get_settings

log = logging.getLogger(__name__)

REPORT_PERIOD_DAYS = 7


def reporter_node(state: GraphState, config: dict[str, Any] | None = None) -> dict:
    settings = get_settings()
    kb = KnowledgeBase()

    period_end = datetime.utcnow()
    period_start = period_end - timedelta(days=REPORT_PERIOD_DAYS)

    # Pull every score in the period from KB (this naturally includes the scores
    # we just upserted in evaluator_node, plus older ones from prior cycles).
    period_scores: list[Score] = kb.scores_in_period(period_start, period_end)

    # Dedup by uid, keeping the latest evaluation.
    latest_by_uid: dict[str, Score] = {}
    for s in sorted(period_scores, key=lambda x: x.evaluated_at):
        latest_by_uid[s.candidate_uid] = s

    items_by_band: dict[str, list[ReportItem]] = {
        "strong_recommend": [],
        "watch": [],
        "monitor": [],
    }

    # Prefer in-state objects (this cycle's freshest data) but fall back to KB.
    by_uid_cand = {c.uid: c for c in state.candidates}
    by_uid_analysis = {a.candidate_uid: a for a in state.analyses}

    for uid, s in sorted(latest_by_uid.items(), key=lambda kv: -kv[1].aggregate):
        cand = by_uid_cand.get(uid) or kb.get_candidate(uid)
        if cand is None:
            continue
        analysis = by_uid_analysis.get(uid) or kb.get_analysis(uid)
        items_by_band.setdefault(s.band, []).append(
            ReportItem(candidate=cand, analysis=analysis, score=s)
        )

    report = Report(
        period_start=period_start,
        period_end=period_end,
        strong_recommend=items_by_band["strong_recommend"],
        watch=items_by_band["watch"],
        monitor=items_by_band["monitor"],
    )

    paths = render_report(report, out_dir=settings.reports_dir)
    log.info(
        "reporter (period=%dd): %d strong / %d watch / %d monitor -> %s",
        REPORT_PERIOD_DAYS,
        len(report.strong_recommend), len(report.watch), len(report.monitor),
        ", ".join(str(p) for p in paths.values()),
    )

    return {"report": report}
