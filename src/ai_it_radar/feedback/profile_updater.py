"""Periodic Lab Profile auto-updater.

Reads the most recent feedback (default: last 30 days), extracts topics from adopted
items via an LLM, and merges them into `lab_profile.yaml#auto_learned`. Anchors are
never modified; auto_learned topics are bounded in count and weight to avoid drift.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from ..llm import llm_json, primary_llm
from ..memory.kb import KnowledgeBase
from ..memory.profile import LabProfile
from ..schemas import FeedbackTag

log = logging.getLogger(__name__)

MAX_AUTO_LEARNED = 12
DEFAULT_LOOKBACK_DAYS = 30
DELAYED_VALUE_BOOST = 0.15  # bump topics whose "watch" items later got "adopt"

_TOPIC_EXTRACTION = """You are clustering recently ADOPTED AI resources into a small set of
TOPICS that describe the Lab's evolving interests. Output 3-7 topics, each a short
phrase plus a few keywords. Avoid duplicates of these existing anchors:

{anchors_block}

## Adopted resources (titles + summaries)
{items_block}

Output STRICT JSON:
{{
  "topics": [
    {{"topic": "...", "keywords": ["...", "..."], "rationale": "..."}}
  ]
}}
"""


class ProfileUpdater:
    def __init__(self, kb: KnowledgeBase | None = None, profile: LabProfile | None = None) -> None:
        self._kb = kb or KnowledgeBase()
        self._profile = profile or LabProfile()

    def run(self, lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> dict[str, Any]:
        since = datetime.utcnow() - timedelta(days=lookback_days)
        recent = self._kb.feedback_since(since)
        adopted = [f for f in recent if f.tag == FeedbackTag.ADOPT]
        watched = [f for f in recent if f.tag == FeedbackTag.WATCH]

        if not adopted:
            log.info("profile_updater: no adoptions in last %d days, skipping", lookback_days)
            return {"updated": False, "reason": "no_adoptions"}

        # Build LLM prompt context.
        items_lines: list[str] = []
        for fb in adopted[:30]:
            cand = self._kb.get_candidate(fb.candidate_uid)
            if not cand:
                continue
            items_lines.append(f"- [{cand.kind.value}] {cand.title}\n  {cand.summary[:280]}")

        if not items_lines:
            return {"updated": False, "reason": "no_candidate_records"}

        anchors_block = "\n".join(
            f"- {a.topic}: {', '.join(a.keywords)}" for a in self._profile.anchors()
        )
        prompt = _TOPIC_EXTRACTION.format(
            anchors_block=anchors_block or "(none)",
            items_block="\n".join(items_lines),
        )

        try:
            data = llm_json(primary_llm(), prompt)
        except Exception as e:
            log.exception("profile_updater LLM call failed: %s", e)
            return {"updated": False, "reason": f"llm_error:{e}"}

        topics_raw = data.get("topics") or []
        new_auto: list[dict[str, Any]] = []
        for t in topics_raw[:MAX_AUTO_LEARNED]:
            if not isinstance(t, dict) or "topic" not in t:
                continue
            new_auto.append({
                "topic": str(t["topic"]).strip(),
                "weight": 0.5,  # auto-learned starts at half-weight
                "keywords": [str(k) for k in (t.get("keywords") or []) if k][:8],
            })

        # Delayed-value boost: any "watch" item that later flipped to "adopt" -> bump
        # topics whose keywords overlap that item's title/summary.
        adopted_uids = {f.candidate_uid for f in adopted}
        watched_then_adopted_uids = {f.candidate_uid for f in watched if f.candidate_uid in adopted_uids}
        if watched_then_adopted_uids and new_auto:
            booster_text = " ".join(
                (self._kb.get_candidate(u).title.lower() if self._kb.get_candidate(u) else "")
                for u in watched_then_adopted_uids
            )
            for t in new_auto:
                if any(k.lower() in booster_text for k in t["keywords"]):
                    t["weight"] = min(0.9, t["weight"] + DELAYED_VALUE_BOOST)

        self._profile.replace_auto_learned(new_auto)
        log.info("profile_updater: refreshed auto_learned with %d topics", len(new_auto))
        return {"updated": True, "topic_count": len(new_auto), "topics": new_auto}
