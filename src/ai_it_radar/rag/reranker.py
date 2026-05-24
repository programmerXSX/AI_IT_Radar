"""LLM-based reranker / dedup judge.

Pure vector similarity is too coarse: it routinely conflates "same paper, different
version" and "two different papers in the same line of work". This stage takes the
top-K vector hits and asks the LLM to render a final yes/no on duplication.
"""
from __future__ import annotations

from langchain_openai import ChatOpenAI

from ..llm import llm_json, primary_llm
from ..memory.kb import KnowledgeBase
from ..schemas import Candidate

_DEDUP_PROMPT = """You decide whether a NEW resource is essentially the SAME as an existing
resource we have already catalogued. Treat as DUPLICATE only if it is the same paper,
project, or model release (different versions/mirrors of the SAME thing count as duplicates).
Two resources tackling related problems but with distinct contributions are NOT duplicates.

## NEW
Title: {new_title}
URL: {new_url}
Summary: {new_summary}
Content excerpt: {new_content}

## CANDIDATES (vector neighbors)
{neighbors_block}

Output STRICT JSON:
{{
  "duplicate_uid": "<uid from candidates, or null>",
  "reason": "<one sentence>"
}}
"""


class LLMReranker:
    def __init__(self, llm: ChatOpenAI | None = None) -> None:
        self._llm = llm or primary_llm()

    def judge_duplicates(
        self,
        candidate: Candidate,
        neighbors: list,
        *,
        kb: KnowledgeBase,
    ):
        blocks: list[str] = []
        for n in neighbors:
            existing = kb.get_candidate(n.uid)
            if not existing:
                continue
            blocks.append(
                f"- uid={n.uid} | title={existing.title}\n"
                f"  url={existing.url}\n"
                f"  summary={existing.summary[:300]}"
            )
        if not blocks:
            return None
        prompt = _DEDUP_PROMPT.format(
            new_title=candidate.title,
            new_url=candidate.url,
            new_summary=candidate.summary[:400],
            new_content=candidate.content[:600],
            neighbors_block="\n".join(blocks),
        )
        try:
            result = llm_json(self._llm, prompt)
        except Exception:
            return None
        dup_uid = result.get("duplicate_uid")
        if not dup_uid:
            return None
        for n in neighbors:
            if n.uid == dup_uid:
                return (n, result.get("reason", ""))
        return None
