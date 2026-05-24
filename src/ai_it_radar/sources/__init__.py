from .base import SourceAdapter
from .arxiv import ArxivSource
from .github_trending import GitHubTrendingSource
from .huggingface import HuggingFaceSource

__all__ = ["SourceAdapter", "ArxivSource", "GitHubTrendingSource", "HuggingFaceSource"]


def all_sources() -> dict[str, type[SourceAdapter]]:
    return {
        "arxiv": ArxivSource,
        "github_trending": GitHubTrendingSource,
        "huggingface": HuggingFaceSource,
    }
