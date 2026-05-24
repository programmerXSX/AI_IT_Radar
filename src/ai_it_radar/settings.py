"""Centralized configuration — loaded from .env (RADAR_*) and YAML."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseModel):
    provider: Literal["openai", "ollama", "deepseek", "azure"] = "openai"
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    temperature: float = 0.2
    timeout_s: int = 60


class EmbeddingSettings(BaseModel):
    provider: Literal["local", "openai", "dashscope"] = "local"
    model: str = "BAAI/bge-m3"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    dim: int | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="RADAR_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    llm: LLMSettings = LLMSettings()
    critic: LLMSettings = LLMSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()

    github_token: str = ""

    data_dir: Path = Path("./data")
    config_dir: Path = Path("./config")
    reports_dir: Path = Path("./reports")

    # Pipeline tuning
    triage_dedup_threshold: float = 0.86
    triage_profile_threshold: float = 0.32
    # If a candidate's cosine similarity to the IgnoreFilter centroid (built from past
    # Ignore feedback) is >= this value, drop it before evaluation. 0.62 is moderately
    # strict — high enough to block "more papers like the one I disliked" but not so
    # high that it constantly fires on adjacent topics.
    triage_ignore_threshold: float = 0.62
    exploration_budget: float = 0.15
    critic_disagreement_threshold: int = 2
    band_strong_recommend: float = 3.5
    band_watch: float = 2.5

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "radar.sqlite"

    @property
    def chroma_path(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def checkpoint_path(self) -> Path:
        return self.data_dir / "checkpoints.sqlite"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_dirs()
    return _settings


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_sources_config() -> dict[str, Any]:
    return load_yaml(get_settings().config_dir / "sources.yaml")


def load_lab_profile() -> dict[str, Any]:
    return load_yaml(get_settings().config_dir / "lab_profile.yaml")


def save_lab_profile(data: dict[str, Any]) -> None:
    path = get_settings().config_dir / "lab_profile.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def load_eval_specs() -> list[dict[str, Any]]:
    specs_dir = get_settings().config_dir / "eval_specs"
    out: list[dict[str, Any]] = []
    if not specs_dir.exists():
        return out
    for p in sorted(specs_dir.glob("*.yaml")):
        out.append(load_yaml(p))
    return out
