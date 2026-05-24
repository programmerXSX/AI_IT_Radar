"""Golden-set regression harness.

Drift in the prompt or LLM version frequently causes silent score shifts. This module
re-runs the EvaluatorAgent over a curated golden set and compares aggregate scores
to expected ranges. CI / `radar regression` exits non-zero on a drift exceeding the
configured tolerance.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from ..schemas import Candidate, CandidateKind, SourceKind

log = logging.getLogger(__name__)


@dataclass
class GoldenItem:
    uid: str
    title: str
    url: str
    kind: CandidateKind
    summary: str
    content: str
    expected_band: str            # "strong_recommend" | "watch" | "monitor"
    expected_aggregate: float
    tolerance: float = 0.7

    def as_candidate(self) -> Candidate:
        return Candidate(
            uid=self.uid,
            source=SourceKind.OTHER,
            kind=self.kind,
            title=self.title,
            url=self.url,
            summary=self.summary,
            content=self.content,
            metadata={"golden": True},
        )


@dataclass
class GoldenSet:
    items: list[GoldenItem]

    @classmethod
    def load(cls, path: Path) -> "GoldenSet":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        items = []
        for it in data.get("items", []):
            items.append(
                GoldenItem(
                    uid=it["uid"],
                    title=it["title"],
                    url=it["url"],
                    kind=CandidateKind(it.get("kind", "repo")),
                    summary=it.get("summary", ""),
                    content=it.get("content", ""),
                    expected_band=it["expected_band"],
                    expected_aggregate=float(it["expected_aggregate"]),
                    tolerance=float(it.get("tolerance", 0.7)),
                )
            )
        return cls(items=items)


@dataclass
class RegressionReport:
    run_at: datetime
    drift_count: int
    failed_items: list[dict[str, Any]]
    passed_items: list[dict[str, Any]]

    @property
    def ok(self) -> bool:
        return self.drift_count == 0


def run_regression(golden: GoldenSet, evaluator_fn) -> RegressionReport:
    """Run `evaluator_fn(candidate) -> Score` over each golden item and report drift.

    `evaluator_fn` is supplied by the caller (typically a thin wrapper over EvaluatorAgent)
    so this module stays free of LangGraph imports for testability.
    """
    failed: list[dict[str, Any]] = []
    passed: list[dict[str, Any]] = []
    for item in golden.items:
        try:
            score = evaluator_fn(item.as_candidate())
        except Exception as e:
            failed.append({"uid": item.uid, "error": str(e)})
            continue
        delta = abs(score.aggregate - item.expected_aggregate)
        record = {
            "uid": item.uid,
            "title": item.title,
            "expected_aggregate": item.expected_aggregate,
            "actual_aggregate": score.aggregate,
            "expected_band": item.expected_band,
            "actual_band": score.band,
            "delta": delta,
            "tolerance": item.tolerance,
        }
        if delta > item.tolerance or score.band != item.expected_band:
            failed.append(record)
        else:
            passed.append(record)

    return RegressionReport(
        run_at=datetime.utcnow(),
        drift_count=len(failed),
        failed_items=failed,
        passed_items=passed,
    )
