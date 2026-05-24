"""AnalystAgent — structured extraction with RAG-supplied analogies."""
from __future__ import annotations

import logging
from typing import Any

from ..llm import llm_json, primary_llm
from ..memory.kb import KnowledgeBase
from ..rag.embedder import get_embedder
from ..rag.retriever import Retriever
from ..schemas import Analysis, Candidate, GraphState

log = logging.getLogger(__name__)


_ANALYST_PROMPT = """You extract a STRUCTURED ENGINEERING SUMMARY from an AI resource.
Stay STRICTLY grounded in the provided content; if a field is unsupported, use an empty
list/string rather than guessing.

## Candidate
Title: {title}
Type: {kind}
URL: {url}
Summary: {summary}

## Source content (truncated)
{content}

## Analogous resources we have catalogued (use them only to phrase comparable terms; do NOT copy)
{analogies_block}

Output STRICT JSON with these keys:
{{
  "key_capabilities": ["..."],          // 3-7 short capability statements
  "method_summary": "...",              // 1-3 sentences, technical
  "dependency_stack": ["..."],          // libraries / runtimes / hardware mentioned
  "data_requirements": "...",           // training/eval data, if any
  "license": "...",                     // SPDX-style if knowable, else null
  "known_limitations": ["..."]          // explicit limitations stated by authors
}}
"""


def analyst_node(state: GraphState, config: dict[str, Any] | None = None) -> dict:
    if not state.candidates:
        return {"analyses": []}

    kb = KnowledgeBase()
    embedder = get_embedder()
    retriever = Retriever(kb, embedder)
    llm = primary_llm()

    analyses: list[Analysis] = []
    for c in state.candidates:
        try:
            analysis = _analyze_one(c, retriever, kb, llm)
        except Exception as e:
            log.exception("analyst: failed for %s: %s", c.uid, e)
            analysis = Analysis(candidate_uid=c.uid, method_summary=f"analyst_error: {e}")
        analyses.append(analysis)
        kb.upsert_analysis(analysis)

    log.info("analyst: produced %d analyses", len(analyses))
    return {"analyses": analyses}


def _analyze_one(c: Candidate, retriever: Retriever, kb: KnowledgeBase, llm) -> Analysis:
    neighbors = retriever.neighbors(c, k=3)
    block_lines: list[str] = []
    for n in neighbors:
        existing = kb.get_candidate(n.uid)
        if not existing:
            continue
        block_lines.append(
            f"- [{existing.kind.value}] {existing.title}: {existing.summary[:200]}"
        )
    analogies_block = "\n".join(block_lines) or "(none)"

    prompt = _ANALYST_PROMPT.format(
        title=c.title,
        kind=c.kind.value,
        url=c.url,
        summary=c.summary[:600],
        content=c.content[:3500],
        analogies_block=analogies_block,
    )
    data = llm_json(llm, prompt)
    return Analysis(
        candidate_uid=c.uid,
        key_capabilities=list(data.get("key_capabilities") or []),
        method_summary=str(data.get("method_summary") or ""),
        dependency_stack=list(data.get("dependency_stack") or []),
        data_requirements=str(data.get("data_requirements") or ""),
        license=data.get("license") or c.metadata.get("license"),
        known_limitations=list(data.get("known_limitations") or []),
        related_uids=[n.uid for n in neighbors],
    )
