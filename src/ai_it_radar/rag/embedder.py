"""Embedding model abstraction.

Supported providers (set via `RADAR_EMBEDDING__PROVIDER` in `.env`):
- `local`     : sentence-transformers (default `BAAI/bge-m3` — strong CN/EN multi-lingual).
                Requires no API; ~2.3GB on first run.
- `openai`    : Any OpenAI-compatible `/embeddings` endpoint.
- `dashscope` : Alibaba Tongyi DashScope native API (recommended for CN content).
                Reads `DASHSCOPE_API_KEY` from env if `RADAR_EMBEDDING__API_KEY` is empty.
                Default model: `text-embedding-v3` (8192 ctx, 100+ languages).
"""
from __future__ import annotations

from typing import Protocol

from ..settings import get_settings


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class _LocalEmbedder:
    def __init__(self, model: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode([text], normalize_embeddings=True)[0].tolist()


class _OpenAIEmbedder:
    def __init__(self, model: str, base_url: str, api_key: str) -> None:
        from langchain_openai import OpenAIEmbeddings

        self._inner = OpenAIEmbeddings(
            model=model,
            base_url=base_url,
            api_key=api_key or "EMPTY",
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(text)


class _DashScopeEmbedder:
    """Tongyi DashScope embedding via `langchain_community.embeddings.DashScopeEmbeddings`.

    DashScope's text-embedding-v3 has a hard per-call batch limit of 6 documents
    (older v1/v2 allow 25). We chunk defensively here so the rest of the codebase
    can pass arbitrary list sizes without HTTP 400s.
    """

    # text-embedding-v3 limit; safe lower bound for v1/v2 too.
    BATCH_SIZE = 6

    def __init__(self, model: str, api_key: str = "") -> None:
        from langchain_community.embeddings import DashScopeEmbeddings

        kwargs: dict = {"model": model}
        if api_key:
            kwargs["dashscope_api_key"] = api_key
        # If api_key is empty, the underlying SDK reads `DASHSCOPE_API_KEY` env var.
        self._inner = DashScopeEmbeddings(**kwargs)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            chunk = texts[i : i + self.BATCH_SIZE]
            out.extend(self._inner.embed_documents(chunk))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(text)


_singleton: Embedder | None = None


def get_embedder() -> Embedder:
    global _singleton
    if _singleton is not None:
        return _singleton
    cfg = get_settings().embedding
    if cfg.provider == "local":
        _singleton = _LocalEmbedder(cfg.model)
    elif cfg.provider == "dashscope":
        _singleton = _DashScopeEmbedder(cfg.model, cfg.api_key)
    else:
        _singleton = _OpenAIEmbedder(cfg.model, cfg.base_url, cfg.api_key)
    return _singleton
