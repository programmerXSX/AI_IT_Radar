"""Declarative evaluation rubric loader."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from jinja2 import Template

from ..settings import load_eval_specs as _load_eval_specs_raw


@dataclass
class EvalSpec:
    id: str
    display_name: str
    weight: float
    rubric: dict[int, str]
    rag_neighbors_k: int
    prompt_template: str
    raw: dict[str, Any] = field(default_factory=dict)

    def render_prompt(self, **vars: Any) -> str:
        # Make the rubric available as a sorted dict
        ctx = dict(vars)
        ctx.setdefault("rubric", {int(k): v for k, v in self.rubric.items()})
        return Template(self.prompt_template).render(**ctx)


def load_eval_specs() -> list[EvalSpec]:
    out: list[EvalSpec] = []
    for raw in _load_eval_specs_raw():
        if not raw:
            continue
        out.append(
            EvalSpec(
                id=raw["id"],
                display_name=raw.get("display_name", raw["id"]),
                weight=float(raw.get("weight", 1.0)),
                rubric={int(k): v for k, v in (raw.get("rubric") or {}).items()},
                rag_neighbors_k=int(raw.get("rag_neighbors_k", 0)),
                prompt_template=raw.get("prompt_template", ""),
                raw=raw,
            )
        )
    return out
