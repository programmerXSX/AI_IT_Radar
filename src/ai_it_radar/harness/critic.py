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

_CRITIC_PROMPT = """你是一名独立的评审员（CRITIC）。主评审（PRIMARY JUDGE）已对一份 AI 资源
在以下维度上完成了评分。你的任务是验证该评分是否能被现有证据支撑。
你不能一味同意——如果引用的证据实际上不支持该评分，请明确指出。

## 评测维度: {dim_name}
评分标准（Rubric）:
{rubric_block}

## 源内容（唯一的事实依据）
{content}

## 主评审的输出
score: {score}
quote: "{quote}"
rationale: {rationale}

输出 STRICT JSON：
{{
  "agree": <true|false>,
  "suggested_score": <int 0-5 或 null>,
  "disagreement_reason": "<一句话原因；agree=true 时留空>"
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
                disagreement_reason=f"复核模型异常：{e}",
            )
        return CriticVerdict(
            dimension_id=spec.id,
            agree=bool(result.get("agree", True)),
            suggested_score=result.get("suggested_score"),
            disagreement_reason=result.get("disagreement_reason", ""),
        )
