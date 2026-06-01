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


_ANALYST_PROMPT = """你从一份 AI 资源中提取结构化的工程摘要。
必须严格基于提供的内容；如果某个字段没有依据，请使用空列表/空字符串，不要猜测。

## 候选条目
Title: {title}
Type: {kind}
URL: {url}
Summary: {summary}

## 源内容（已截断）
{content}

## 我们已收录的类似资源（仅用于措辞参考，不要直接照抄）
{analogies_block}

输出 STRICT JSON，包含以下字段：
{{
  "key_capabilities": ["..."],          // 3-7 条核心能力简述
  "method_summary": "...",              // 1-3 句技术性总结
  "dependency_stack": ["..."],          // 涉及的库 / 运行时 / 硬件
  "data_requirements": "...",           // 训练或评估数据需求（如有）
  "license": "...",                     // SPDX 格式，无法确定则 null
  "known_limitations": ["..."]          // 作者明确指出的局限性
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
