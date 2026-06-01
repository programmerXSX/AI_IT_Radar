"""EvaluatorAgent — multi-dimensional scoring with RAG anchors and Critic."""
from __future__ import annotations

import logging
from typing import Any

from ..harness.critic import Critic
from ..harness.eval_spec import EvalSpec, load_eval_specs
from ..llm import llm_json, primary_llm
from ..memory.kb import KnowledgeBase
from ..memory.profile import LabProfile
from ..rag.embedder import get_embedder
from ..rag.retriever import Retriever
from ..schemas import Candidate, DimensionScore, GraphState, Score
from ..settings import get_settings

log = logging.getLogger(__name__)

# Dimensions whose semantics are inverted (lower raw == worse, but harness rubric is
# already directional so we don't need an inversion — keep this list for documentation).
_INVERTED_DIMS = {"reproduction_cost", "risk"}


def evaluator_node(state: GraphState, config: dict[str, Any] | None = None) -> dict:
    if not state.candidates:
        return {"scores": []}

    settings = get_settings()
    specs = load_eval_specs()
    if not specs:
        log.warning("evaluator: no eval specs found; skipping")
        return {"scores": []}

    kb = KnowledgeBase()
    embedder = get_embedder()
    retriever = Retriever(kb, embedder)
    profile = LabProfile()
    llm = primary_llm()
    critic = Critic()

    scores: list[Score] = []
    for c in state.candidates:
        try:
            s = _evaluate_one(
                c, specs, retriever, kb, profile, llm, critic,
                cycle_id=state.cycle_id,
            )
        except Exception as e:
            log.exception("evaluator: failed for %s: %s", c.uid, e)
            s = Score(
                candidate_uid=c.uid,
                dimensions=[],
                aggregate=0.0,
                band="monitor",
                needs_human=True,
            )
        s.band = _band_of(s.aggregate, settings)
        scores.append(s)
        kb.upsert_score(s, cycle_id=state.cycle_id)

    log.info("evaluator: scored %d candidates", len(scores))
    return {"scores": scores}


def _evaluate_one(
    candidate: Candidate,
    specs: list[EvalSpec],
    retriever: Retriever,
    kb: KnowledgeBase,
    profile: LabProfile,
    llm,
    critic: Critic,
    *,
    cycle_id: str,
) -> Score:
    settings = get_settings()
    dims: list[DimensionScore] = []
    verdicts = []
    needs_human = False

    for spec in specs:
        neighbors_block = _neighbors_block(candidate, spec, retriever, kb)
        prompt_vars = {
            "title": candidate.title,
            "url": candidate.url,
            "kind": candidate.kind.value,
            "summary": candidate.summary[:600],
            "content": candidate.content[:3500],
            "stars": candidate.metadata.get("stars"),
            "last_commit": candidate.metadata.get("last_commit"),
            "license": candidate.metadata.get("license"),
            "contributors": candidate.metadata.get("contributors"),
            "neighbors": neighbors_block,
            "lab_profile_summary": profile.summary_text(),
            "anchors": [a.__dict__ for a in profile.anchors()],
            "auto_learned": [a.__dict__ for a in profile.auto_learned()],
        }
        prompt = spec.render_prompt(**prompt_vars)
        try:
            raw = llm_json(llm, prompt)
        except Exception as e:
            log.warning("evaluator dim=%s failed: %s", spec.id, e)
            raw = {"score": 0, "confidence": 0.0,
                   "rationale": f"模型调用异常：{e}", "quote": ""}

        ds = DimensionScore(
            dimension_id=spec.id,
            score=int(raw.get("score") or 0),
            confidence=float(raw.get("confidence") or 0.0),
            rationale=str(raw.get("rationale") or ""),
            quote=str(raw.get("quote") or ""),
            extras={k: v for k, v in raw.items()
                    if k not in {"score", "confidence", "rationale", "quote"}},
        )

        verdict = critic.review(spec, ds, candidate.content)
        verdicts.append(verdict)
        kb.record_eval_trace(
            candidate_uid=candidate.uid,
            cycle_id=cycle_id,
            dimension_id=spec.id,
            prompt=prompt,
            raw_response=str(raw),
            critic_response=str(verdict.model_dump()),
            neighbors=neighbors_block,
        )

        # Reconcile if disagreement is large.
        if (
            not verdict.agree
            and verdict.suggested_score is not None
            and abs(verdict.suggested_score - ds.score) >= settings.critic_disagreement_threshold
        ):
            needs_human = True
            ds.score = (ds.score + verdict.suggested_score) // 2
            ds.rationale += f" | 复核意见：{verdict.disagreement_reason}"

        dims.append(ds)

    aggregate = _aggregate(dims, specs)
    return Score(
        candidate_uid=candidate.uid,
        dimensions=dims,
        critic_verdicts=verdicts,
        aggregate=aggregate,
        band="monitor",
        needs_human=needs_human,
    )


def _neighbors_block(candidate: Candidate, spec: EvalSpec, retriever: Retriever, kb: KnowledgeBase):
    if spec.rag_neighbors_k <= 0:
        return []
    neighbors = retriever.neighbors(candidate, k=spec.rag_neighbors_k)
    block: list[dict[str, Any]] = []
    for n in neighbors:
        existing = kb.get_candidate(n.uid)
        if not existing:
            continue
        prior = kb.latest_score(n.uid)
        if not prior:
            continue
        prev = next((d for d in prior.dimensions if d.dimension_id == spec.id), None)
        if not prev:
            continue
        block.append({
            "title": existing.title,
            "kind": existing.kind.value,
            "score": prev.score,
            "rationale": prev.rationale[:200],
        })
    return block


def _aggregate(dims: list[DimensionScore], specs: list[EvalSpec]) -> float:
    weights = {s.id: s.weight for s in specs}
    total_w = sum(weights.values()) or 1.0
    total = sum(d.score * weights.get(d.dimension_id, 0.0) for d in dims)
    return round(total / total_w, 3)


def _band_of(aggregate: float, settings):
    if aggregate >= settings.band_strong_recommend:
        return "strong_recommend"
    if aggregate >= settings.band_watch:
        return "watch"
    return "monitor"
