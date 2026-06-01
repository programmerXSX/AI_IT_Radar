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

_DEDUP_PROMPT = """你判断一个新资源是否与我们已收录的某个资源本质上是同一个。
只有当它是同一篇论文、同一个项目或同一个模型发布时才判定为 DUPLICATE
（同一事物的不同版本/镜像视为重复）。两个处理相关但贡献不同的资源不应判定为重复。

## 新资源
Title: {new_title}
URL: {new_url}
Summary: {new_summary}
Content excerpt: {new_content}

## 候选匹配项（向量近邻）
{neighbors_block}

输出 STRICT JSON：
{{
  "duplicate_uid": "<候选列表中的 uid，或 null>",
  "reason": "<一句话原因>"
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
