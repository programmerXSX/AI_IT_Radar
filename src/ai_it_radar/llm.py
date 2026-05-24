"""LLM provider abstraction.

Wraps any OpenAI-compatible endpoint via langchain-openai's ChatOpenAI.
This covers OpenAI, DeepSeek, Together, vLLM-served models, and Ollama (>=0.4 with /v1).
"""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .settings import LLMSettings, get_settings


def build_chat_model(cfg: LLMSettings) -> ChatOpenAI:
    return ChatOpenAI(
        model=cfg.model,
        base_url=cfg.base_url,
        api_key=cfg.api_key or "EMPTY",
        temperature=cfg.temperature,
        timeout=cfg.timeout_s,
    )


def primary_llm() -> ChatOpenAI:
    return build_chat_model(get_settings().llm)


def critic_llm() -> ChatOpenAI:
    return build_chat_model(get_settings().critic)


_JSON_BLOCK = re.compile(r"\{[\s\S]*\}")
# Match any backslash that is NOT the start of a valid JSON escape sequence.
# Valid: \" \\ \/ \b \f \n \r \t \uXXXX
_BAD_BACKSLASH = re.compile(r'\\(?!["\\/bfnrtu])')


def parse_json_response(text: str) -> dict[str, Any]:
    """Best-effort extract a JSON object from an LLM completion.

    Handles common failure modes:
    - markdown fences (```json ... ```)
    - leading prose / trailing commentary (greedy `{...}` extraction)
    - LLM-emitted bare backslashes that violate JSON escape rules
      (e.g. dumping a Windows path or a code snippet inside a string value)
    """
    original = text
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = text.rstrip("`").rstrip()
        if text.endswith("```"):
            text = text[:-3].rstrip()

    first_err: Exception | None = None
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        first_err = e

    m = _JSON_BLOCK.search(text)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        # Last resort: double any bare backslash that isn't a valid JSON escape.
        cleaned = _BAD_BACKSLASH.sub(r"\\\\", candidate)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Failed to parse JSON from LLM output. "
        f"Underlying error: {first_err}. "
        f"Output (truncated 300 chars): {original[:300]!r}"
    )


DEFAULT_SYSTEM_PROMPT = """你是一名严谨、克制的 AI 技术评测助手，为内部 AI Lab 决策者提供辅助分析。

【硬性输出语言规则 — 必须遵守，不可妥协】
即使本条用户消息的指令是英文写的，你也必须按以下规则输出。这是用户的明确要求，不是建议：

1. 所有「自然语言字段」一律用**简体中文**：rationale, reason, summary, method_summary,
   disagreement_reason, note, topic, 以及任何字符串列表中表达观点/描述/原因的元素。
2. JSON 的「键名」保持英文不变（"score" / "confidence" / "rationale" 等）。
3. `quote` 字段**原样保留原文**：原文是英文就保留英文，原文是中文就保留中文，绝不翻译。
   这是为了让人工核查时能在 README/abstract 里搜到该片段。
4. 数字 / 布尔 / URL 等机器值保持不变。

【输出格式硬性规则】
- 当被要求 STRICT JSON 时只输出一个 JSON 对象，前后无任何文字、无 markdown 代码块。"""


# Appended to every user prompt as a final reinforcement. Crucial because the eval
# rubric prompts themselves are written in English, and LLMs strongly tend to mirror
# the user-message language. A short Chinese reminder right before output generation
# is the single most reliable way to enforce Chinese rationales.
LANG_REMINDER = """

──────────────
【输出语言最终提醒】
- rationale, reason, summary, disagreement_reason, note 等所有自然语言字段必须用**简体中文**。
- quote 字段保留原文（不翻译）。
- JSON 键名保持英文。
- 仅输出 JSON 对象本身，不要任何前后文。"""


def llm_json(
    llm: ChatOpenAI,
    prompt: str,
    *,
    system: str | None = None,
    lang_reminder: bool = True,
) -> dict[str, Any]:
    """Run a single prompt and parse a JSON response.

    A default Chinese-output system prompt is applied when `system` is None.
    Pass system="" to disable it explicitly.

    `lang_reminder=True` (default) appends a strong Chinese-output reminder at the
    tail of the user prompt — required because system prompts alone are routinely
    ignored when the user message is itself in English with concrete task language.
    """
    msgs: list[Any] = []
    sys_text = DEFAULT_SYSTEM_PROMPT if system is None else system
    if sys_text:
        msgs.append(SystemMessage(content=sys_text))
    final_prompt = prompt + LANG_REMINDER if lang_reminder else prompt
    msgs.append(HumanMessage(content=final_prompt))
    resp = llm.invoke(msgs)
    return parse_json_response(resp.content if isinstance(resp.content, str) else str(resp.content))
