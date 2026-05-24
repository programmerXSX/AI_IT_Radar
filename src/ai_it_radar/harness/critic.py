"""Second-pass LLM critic.

The critic (ideally a different model family from the primary judge) is shown the
candidate, the rubric, and the primary judge's score+rationale. It outputs an
agree/disagree verdict. If disagreement is large enough the case is flagged for human.
"""
from __future__ import annotations

from langchain_openai import ChatOpenAI

from ..llm import critic_llm, llm_json
from ..schemas import CriticVerdict, DimensionScore
from .eval_spec import EvalSpec

_CRITIC_PROMPT = """You are an independent CRITIC. The PRIMARY JUDGE has already scored an
AI resource on the dimension below. Your job is to verify whether their score is
defensible from the evidence provided. You must NOT just agree — push back if the
quoted evidence does not actually support the score.

## Dimension: {dim_name}
Rubric:
{rubric_block}

## Source content (only ground truth)
{content}

## Primary judge output
score: {score}
quote: "{quote}"
rationale: {rationale}

Output STRICT JSON:
{{
  "agree": <true|false>,
  "suggested_score": <int 0-5 or null>,
  "disagreement_reason": "<one sentence; empty if agree=true>"
}}
"""


class Critic:
    def __init__(self, llm: ChatOpenAI | None = None) -> None:
        self._llm = llm or critic_llm()

    def review(
        self,
        spec: EvalSpec,
        primary_score: DimensionScore,
        candidate_content: str,
    ) -> CriticVerdict:
        rubric_block = "\n".join(f"- {k}: {v}" for k, v in sorted(spec.rubric.items()))
        prompt = _CRITIC_PROMPT.format(
            dim_name=spec.display_name,
            rubric_block=rubric_block,
            content=candidate_content[:3000],
            score=primary_score.score,
            quote=primary_score.quote,
            rationale=primary_score.rationale,
        )
        try:
            result = llm_json(self._llm, prompt)
        except Exception as e:
            return CriticVerdict(
                dimension_id=spec.id,
                agree=True,
                suggested_score=primary_score.score,
                disagreement_reason=f"critic_error:{e}",
            )
        return CriticVerdict(
            dimension_id=spec.id,
            agree=bool(result.get("agree", True)),
            suggested_score=result.get("suggested_score"),
            disagreement_reason=result.get("disagreement_reason", ""),
        )
