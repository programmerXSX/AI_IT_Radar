"""ScoutAgent — fan out to enabled source adapters and normalize Candidates."""
from __future__ import annotations

import logging
from typing import Any

from ..schemas import Candidate, GraphState
from ..settings import load_sources_config
from ..sources import all_sources

log = logging.getLogger(__name__)


def scout_node(state: GraphState, config: dict[str, Any] | None = None) -> dict:
    """Pull from each enabled source. Idempotent w.r.t. uid."""
    config = config or {}
    selected: set[str] | None = None
    rcfg = (config.get("configurable") or {}) if isinstance(config, dict) else {}
    if rcfg.get("sources"):
        selected = set(rcfg["sources"])

    sources_cfg = load_sources_config()
    out: dict[str, Candidate] = {c.uid: c for c in state.candidates}

    adapters = all_sources()
    for key, AdapterCls in adapters.items():
        if selected and key not in selected:
            continue
        sub_cfg = sources_cfg.get(key, {}) or {}
        if not sub_cfg.get("enabled", True):
            continue
        log.info("scout: fetching from %s", key)
        try:
            adapter = AdapterCls(sub_cfg)
            for c in adapter.fetch():
                out[c.uid] = c
        except Exception as e:
            log.exception("scout: adapter %s failed: %s", key, e)

    log.info("scout: %d unique candidates collected", len(out))
    return {"candidates": list(out.values())}
