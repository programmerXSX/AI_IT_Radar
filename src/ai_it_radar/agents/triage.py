"""TriageAgent — vector dedup + LLM second pass + profile filtering + exploration budget."""
from __future__ import annotations

import logging
import random
from typing import Any

from ..feedback.ignore_filter import IgnoreFilter
from ..memory.kb import KnowledgeBase
from ..memory.profile import LabProfile, cosine_similarity
from ..rag.embedder import get_embedder
from ..rag.indexer import embedding_text
from ..rag.reranker import LLMReranker
from ..schemas import Candidate, GraphState, TriageDecision, TriageResult
from ..settings import get_settings, load_sources_config

log = logging.getLogger(__name__)


def triage_node(state: GraphState, config: dict[str, Any] | None = None) -> dict:
    settings = get_settings()
    sources_cfg = load_sources_config()
    exploration_budget = float(sources_cfg.get("exploration_budget", settings.exploration_budget))

    rcfg = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    force = bool(rcfg.get("force", False))

    # `--force` bypasses dedup and profile filtering — useful when you've changed the
    # eval prompts / model and want to re-score everything that came in.
    if force:
        log.warning("triage: --force mode, bypassing dedup + profile filter")
        embedder = get_embedder()
        kb = KnowledgeBase()
        results: list[TriageResult] = []
        for c in state.candidates:
            # Re-index the candidate so KB stays current (cheap, idempotent).
            kb.upsert_candidate(c, embedding=embedder.embed_query(embedding_text(c)))
            results.append(TriageResult(
                candidate_uid=c.uid,
                decision=TriageDecision.KEEP,
                profile_match_score=1.0,
                rationale="force mode (re-evaluation requested)",
            ))
        return {"triage": results, "candidates": list(state.candidates)}

    embedder = get_embedder()
    kb = KnowledgeBase()
    profile = LabProfile()
    reranker = LLMReranker()
    ignore_filter = IgnoreFilter(kb)

    profile_centroid = profile.centroid(embedder)
    if ignore_filter.is_active(embedder):
        log.info("triage: IgnoreFilter active (%d ignored items)", ignore_filter.ignore_count)

    results: list[TriageResult] = []
    surviving: list[Candidate] = []
    explore_pool: list[tuple[Candidate, float, list[float]]] = []  # held-out for budget

    for c in state.candidates:
        # Hard exclude: blacklist tokens.
        if profile.has_excluded_token(c.title, c.summary, c.content[:1000]):
            results.append(TriageResult(
                candidate_uid=c.uid,
                decision=TriageDecision.OUT_OF_SCOPE,
                rationale="excluded keyword hit",
            ))
            continue

        text = embedding_text(c)
        vec = embedder.embed_query(text)

        # Already in KB? (deterministic dedup by uid)
        if kb.has_candidate(c.uid):
            results.append(TriageResult(
                candidate_uid=c.uid,
                decision=TriageDecision.DUPLICATE,
                duplicate_of=c.uid,
                rationale="already_in_kb (uid match)",
            ))
            continue

        # Vector dedup: any near-twin existing in KB?
        neighbors_raw = kb.query_similar(vec, k=5, exclude_uids=[c.uid])
        if neighbors_raw and neighbors_raw[0][1] < (1.0 - settings.triage_dedup_threshold):
            # Very close in cosine distance (0 = identical) -> ask LLM.
            from ..rag.retriever import Neighbor
            neighbors = [Neighbor(uid=u, title=m.get("title", ""), kind=m.get("kind", ""),
                                  distance=d, metadata=m) for (u, d, m) in neighbors_raw]
            verdict = reranker.judge_duplicates(c, neighbors, kb=kb)
            if verdict:
                matched, reason = verdict
                results.append(TriageResult(
                    candidate_uid=c.uid,
                    decision=TriageDecision.DUPLICATE,
                    duplicate_of=matched.uid,
                    rationale=reason,
                ))
                continue

        # IgnoreFilter — drop candidates that look like things the user previously
        # marked Ignore. Runs before profile match so we don't waste embedding compute
        # AND so it can override a candidate that would otherwise pass profile match.
        ignore_score = ignore_filter.score(vec, embedder)
        if ignore_score >= settings.triage_ignore_threshold:
            results.append(TriageResult(
                candidate_uid=c.uid,
                decision=TriageDecision.OUT_OF_SCOPE,
                profile_match_score=0.0,
                rationale=(
                    f"matches ignored pattern (cos={ignore_score:.3f} >= "
                    f"{settings.triage_ignore_threshold}, n_ignored="
                    f"{ignore_filter.ignore_count})"
                ),
            ))
            continue

        # Profile match.
        match = cosine_similarity(vec, profile_centroid) if profile_centroid else 0.0
        if match >= settings.triage_profile_threshold:
            results.append(TriageResult(
                candidate_uid=c.uid,
                decision=TriageDecision.KEEP,
                profile_match_score=match,
                rationale=f"profile match cos={match:.3f}",
            ))
            surviving.append(c)
            # Index now so subsequent same-cycle candidates can dedup against it.
            kb.upsert_candidate(c, embedding=vec)
        else:
            explore_pool.append((c, match, vec))

    # Apply exploration budget over the leftover pool: pick top-quality (high embedding norm
    # already handled by `normalize=True`); we use random subsample for diversity.
    n_explore = max(0, int(round(len(state.candidates) * exploration_budget)))
    if explore_pool and n_explore > 0:
        random.shuffle(explore_pool)
        for c, match, vec in explore_pool[:n_explore]:
            results.append(TriageResult(
                candidate_uid=c.uid,
                decision=TriageDecision.EXPLORE,
                profile_match_score=match,
                rationale=f"exploration budget admit (match={match:.3f})",
            ))
            surviving.append(c)
            kb.upsert_candidate(c, embedding=vec)
        for c, match, _ in explore_pool[n_explore:]:
            results.append(TriageResult(
                candidate_uid=c.uid,
                decision=TriageDecision.OUT_OF_SCOPE,
                profile_match_score=match,
                rationale=f"low profile match (cos={match:.3f}) and budget exhausted",
            ))
    else:
        for c, match, _ in explore_pool:
            results.append(TriageResult(
                candidate_uid=c.uid,
                decision=TriageDecision.OUT_OF_SCOPE,
                profile_match_score=match,
                rationale=f"low profile match (cos={match:.3f})",
            ))

    log.info(
        "triage: %d kept (%d explore), %d duplicate, %d oos",
        sum(1 for r in results if r.decision in (TriageDecision.KEEP, TriageDecision.EXPLORE)),
        sum(1 for r in results if r.decision == TriageDecision.EXPLORE),
        sum(1 for r in results if r.decision == TriageDecision.DUPLICATE),
        sum(1 for r in results if r.decision == TriageDecision.OUT_OF_SCOPE),
    )

    return {"triage": results, "candidates": surviving}
