"""LangGraph topology for the radar.

State flows linearly: scout -> triage -> analyst -> evaluator -> reporter.
Each node persists via the SqliteSaver checkpoint, so a crashed cycle is resumable.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator

from langgraph.graph import END, StateGraph

from .agents import analyst_node, evaluator_node, reporter_node, scout_node, triage_node
from .memory.kb import KnowledgeBase
from .memory.short_term import build_checkpointer
from .schemas import GraphState

log = logging.getLogger(__name__)


def _build_graph_uncompiled() -> StateGraph:
    """Construct the StateGraph without compiling — caller adds the checkpointer."""
    g: StateGraph = StateGraph(GraphState)
    g.add_node("scout", scout_node)
    g.add_node("triage", triage_node)
    g.add_node("analyst", analyst_node)
    g.add_node("evaluator", evaluator_node)
    g.add_node("reporter", reporter_node)

    g.set_entry_point("scout")
    g.add_edge("scout", "triage")
    g.add_edge("triage", "analyst")
    g.add_edge("analyst", "evaluator")
    g.add_edge("evaluator", "reporter")
    g.add_edge("reporter", END)
    return g


@contextmanager
def build_radar_graph() -> Iterator[Any]:
    """Yield a compiled LangGraph app with SqliteSaver attached."""
    g = _build_graph_uncompiled()
    with build_checkpointer() as saver:
        app = g.compile(checkpointer=saver)
        yield app


def run_cycle(*, sources: list[str] | None = None, force: bool = False) -> GraphState:
    """One end-to-end pass. Returns the terminal GraphState (which carries the Report).

    `force=True` makes Triage skip dedup + profile filtering, forcing every fetched
    candidate to be re-analyzed and re-evaluated. Use sparingly — burns LLM budget.
    """
    cycle_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    initial = GraphState(cycle_id=cycle_id)

    kb = KnowledgeBase()
    kb.start_cycle(cycle_id)
    log.info("=== radar cycle %s start (force=%s) ===", cycle_id, force)

    cfg: dict[str, Any] = {"configurable": {"thread_id": cycle_id}}
    if sources:
        cfg["configurable"]["sources"] = sources
    if force:
        cfg["configurable"]["force"] = True

    try:
        with build_radar_graph() as app:
            final_state = app.invoke(initial, config=cfg)
    except Exception:
        kb.finish_cycle(cycle_id, status="error")
        raise

    kb.finish_cycle(cycle_id, status="ok")
    log.info("=== radar cycle %s done ===", cycle_id)

    if isinstance(final_state, GraphState):
        return final_state
    return GraphState(**final_state)
