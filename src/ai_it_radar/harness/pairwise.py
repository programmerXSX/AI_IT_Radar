"""Pairwise comparator — relative judgements over absolute ones.

Given a new candidate and a known anchor (e.g. a prior adopted resource on the same
dimension), ask the LLM "which is stronger, by how much". This signal is fed back to
the EvaluatorAgent for calibration: pairwise verdicts are more stable across LLM
versions than direct numeric scoring.
"""
from __future__ import annotations

from langchain_openai import ChatOpenAI

from ..llm import llm_json, primary_llm
from ..schemas import Candidate

_PAIRWISE_PROMPT = """Compare two AI resources on dimension: {dim_name}.

A:
{a_block}

B:
{b_block}

Output STRICT JSON:
{{
  "winner": "A" | "B" | "tie",
  "delta": <int 0-3>,    // 0=tie, 3=overwhelming
  "reason": "<one sentence>"
}}
"""


class PairwiseComparator:
    def __init__(self, llm: ChatOpenAI | None = None) -> None:
        self._llm = llm or primary_llm()

    def compare(self, dim_name: str, a: Candidate, b: Candidate) -> dict:
        prompt = _PAIRWISE_PROMPT.format(
            dim_name=dim_name,
            a_block=_block(a),
            b_block=_block(b),
        )
        try:
            return llm_json(self._llm, prompt)
        except Exception as e:
            return {"winner": "tie", "delta": 0, "reason": f"error:{e}"}


def _block(c: Candidate) -> str:
    return f"title: {c.title}\nurl: {c.url}\nsummary: {c.summary[:300]}\ncontent: {c.content[:600]}"
