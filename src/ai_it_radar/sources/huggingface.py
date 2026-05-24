"""HuggingFace Hub adapter — trending models (and optionally datasets)."""
from __future__ import annotations

import logging
from typing import Iterable

from huggingface_hub import HfApi, hf_hub_download

from ..schemas import Candidate, CandidateKind, SourceKind
from .base import SourceAdapter

log = logging.getLogger(__name__)


class HuggingFaceSource(SourceAdapter):
    name = "huggingface"

    def fetch(self) -> Iterable[Candidate]:
        cfg = self.config
        if not cfg.get("enabled", True):
            return []
        api = HfApi()
        out: list[Candidate] = []

        models_cfg = cfg.get("models", {}) or {}
        if models_cfg.get("enabled", True):
            out.extend(self._fetch_models(api, models_cfg))

        ds_cfg = cfg.get("datasets", {}) or {}
        if ds_cfg.get("enabled", False):
            out.extend(self._fetch_datasets(api, ds_cfg))

        return out

    def _fetch_models(self, api: HfApi, cfg: dict) -> list[Candidate]:
        limit = int(cfg.get("limit", 25))
        sort = cfg.get("sort", "trendingScore")
        direction = int(cfg.get("direction", -1))
        filter_tags = cfg.get("filter_tags", []) or []
        out: list[Candidate] = []
        try:
            for m in api.list_models(
                sort=sort,
                direction=direction,
                limit=limit,
                filter=filter_tags or None,
                full=True,
            ):
                model_id = m.id
                uid = f"hf:model:{model_id}"
                tags = list(m.tags or [])
                summary = ", ".join(tags[:8])
                content = self._fetch_model_card(model_id)
                out.append(
                    Candidate(
                        uid=uid,
                        source=SourceKind.HUGGINGFACE,
                        kind=CandidateKind.MODEL,
                        title=model_id,
                        url=f"https://huggingface.co/{model_id}",
                        summary=summary,
                        content=content,
                        published_at=getattr(m, "created_at", None),
                        metadata={
                            "downloads": getattr(m, "downloads", None),
                            "likes": getattr(m, "likes", None),
                            "pipeline_tag": getattr(m, "pipeline_tag", None),
                            "library_name": getattr(m, "library_name", None),
                            "tags": tags,
                        },
                    )
                )
        except Exception as e:
            log.warning("hf models fetch error: %s", e)
        return out

    def _fetch_datasets(self, api: HfApi, cfg: dict) -> list[Candidate]:
        limit = int(cfg.get("limit", 10))
        sort = cfg.get("sort", "trendingScore")
        direction = int(cfg.get("direction", -1))
        out: list[Candidate] = []
        try:
            for d in api.list_datasets(sort=sort, direction=direction, limit=limit, full=True):
                ds_id = d.id
                out.append(
                    Candidate(
                        uid=f"hf:ds:{ds_id}",
                        source=SourceKind.HUGGINGFACE,
                        kind=CandidateKind.DATASET,
                        title=ds_id,
                        url=f"https://huggingface.co/datasets/{ds_id}",
                        summary=", ".join(list(d.tags or [])[:8]),
                        content=self._fetch_dataset_card(ds_id),
                        published_at=getattr(d, "created_at", None),
                        metadata={
                            "downloads": getattr(d, "downloads", None),
                            "likes": getattr(d, "likes", None),
                            "tags": list(d.tags or []),
                        },
                    )
                )
        except Exception as e:
            log.warning("hf datasets fetch error: %s", e)
        return out

    def _fetch_model_card(self, model_id: str) -> str:
        try:
            path = hf_hub_download(repo_id=model_id, filename="README.md")
            with open(path, encoding="utf-8") as f:
                return f.read()[:8000]
        except Exception:
            return ""

    def _fetch_dataset_card(self, ds_id: str) -> str:
        try:
            path = hf_hub_download(repo_id=ds_id, filename="README.md", repo_type="dataset")
            with open(path, encoding="utf-8") as f:
                return f.read()[:8000]
        except Exception:
            return ""
