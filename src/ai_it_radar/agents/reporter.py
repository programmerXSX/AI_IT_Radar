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
import re
from datetime import datetime, timedelta
from typing import Any

from ..memory.kb import KnowledgeBase
from ..reporter.render import render_report
from ..schemas import GraphState, Report, ReportItem, Score
from ..settings import get_settings

log = logging.getLogger(__name__)

REPORT_PERIOD_DAYS = 7
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_ASCII_WORD_RE = re.compile(r"[A-Za-z]{3,}")


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

    _log_language_drift_warnings(report)
    paths = render_report(report, out_dir=settings.reports_dir)
    log.info(
        "reporter (period=%dd): %d strong / %d watch / %d monitor -> %s",
        REPORT_PERIOD_DAYS,
        len(report.strong_recommend), len(report.watch), len(report.monitor),
        ", ".join(str(p) for p in paths.values()),
    )

    return {"report": report}


def _looks_english_heavy(text: str) -> bool:
    if not text:
        return False
    cjk_count = len(_CJK_RE.findall(text))
    ascii_word_count = len(_ASCII_WORD_RE.findall(text))
    return ascii_word_count >= 8 and ascii_word_count > (cjk_count * 2)


def _log_language_drift_warnings(report: Report) -> None:
    for band_items in (report.strong_recommend, report.watch, report.monitor):
        for item in band_items:
            if item.analysis and _looks_english_heavy(item.analysis.method_summary):
                log.warning(
                    "reporter: method_summary appears non-Chinese uid=%s",
                    item.candidate.uid,
                )
            if not item.score:
                continue
            for dim in item.score.dimensions:
                if _looks_english_heavy(dim.rationale):
                    log.warning(
                        "reporter: rationale appears non-Chinese uid=%s dim=%s",
                        item.candidate.uid,
                        dim.dimension_id,
                    )
