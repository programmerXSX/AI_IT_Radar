"""GitHub Trending adapter.

GitHub does not expose Trending via the official API. We use a two-path strategy:
1. Primary: scrape https://github.com/trending (HTML; may break — kept defensive).
2. Fallback: use the official Search API with a heuristic
   "stars:>N created:>{date} topic:llm OR topic:ai", then enrich each repo with README.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Iterable

import httpx
from bs4 import BeautifulSoup

from ..schemas import Candidate, CandidateKind, SourceKind
from ..settings import get_settings
from .base import SourceAdapter

log = logging.getLogger(__name__)

USER_AGENT = "ai-it-radar/0.1 (+internal)"


class GitHubTrendingSource(SourceAdapter):
    name = "github_trending"

    def fetch(self) -> Iterable[Candidate]:
        cfg = self.config
        if not cfg.get("enabled", True):
            return []
        languages: list[str] = cfg.get("languages", [""]) or [""]
        since: str = cfg.get("since", "weekly")
        spoken: str = cfg.get("spoken_language", "") or ""
        max_per_run: int = int(cfg.get("max_per_run", 20))

        repos: list[Candidate] = []
        seen: set[str] = set()
        for lang in languages:
            scraped = self._scrape_trending(lang, since, spoken)
            for c in scraped:
                if c.uid in seen:
                    continue
                seen.add(c.uid)
                repos.append(c)
                if len(repos) >= max_per_run:
                    return repos

        if not repos and cfg.get("fallback_search_api", True):
            log.info("GitHub Trending scrape returned nothing; falling back to Search API.")
            for c in self._search_api_fallback(cfg, max_per_run):
                if c.uid not in seen:
                    seen.add(c.uid)
                    repos.append(c)
                if len(repos) >= max_per_run:
                    break

        # Enrich with README (best-effort).
        for c in repos:
            self._enrich_readme(c)
        return repos

    # ---- HTML scrape -----------------------------------------------------------------

    def _scrape_trending(self, language: str, since: str, spoken: str) -> list[Candidate]:
        url = "https://github.com/trending"
        if language:
            url += f"/{language}"
        params: dict[str, str] = {"since": since}
        if spoken:
            params["spoken_language_code"] = spoken
        out: list[Candidate] = []
        try:
            r = httpx.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=20.0)
            if r.status_code != 200:
                log.warning("trending scrape status=%s url=%s", r.status_code, r.url)
                return []
            soup = BeautifulSoup(r.text, "lxml")
            for article in soup.select("article.Box-row"):
                a = article.select_one("h2 a")
                if not a:
                    continue
                full_name = a.get_text(strip=True).replace(" ", "").replace("\n", "")
                href = a.get("href", "").strip("/")
                if not full_name or "/" not in full_name:
                    continue
                desc_el = article.select_one("p")
                description = desc_el.get_text(strip=True) if desc_el else ""
                lang_el = article.select_one('span[itemprop="programmingLanguage"]')
                lang_name = lang_el.get_text(strip=True) if lang_el else ""
                stars_el = article.select_one('a[href$="/stargazers"]')
                stars_text = stars_el.get_text(strip=True).replace(",", "") if stars_el else "0"
                stars = _safe_int(stars_text)
                uid = f"github:{href}"
                out.append(
                    Candidate(
                        uid=uid,
                        source=SourceKind.GITHUB,
                        kind=CandidateKind.REPO,
                        title=full_name,
                        url=f"https://github.com/{href}",
                        summary=description,
                        content="",
                        metadata={"language": lang_name, "stars": stars, "source_path": "trending"},
                    )
                )
        except Exception as e:
            log.warning("trending scrape error: %s", e)
        return out

    # ---- Search API fallback ---------------------------------------------------------

    def _search_api_fallback(self, cfg: dict, max_n: int) -> list[Candidate]:
        token = get_settings().github_token
        headers = {"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        lookback = (datetime.utcnow() - timedelta(days=7)).date().isoformat()
        query_template: str = cfg.get(
            "search_api_query", "stars:>50 created:>{lookback_iso} topic:llm OR topic:ai"
        )
        q = query_template.replace("{lookback_iso}", lookback)
        try:
            r = httpx.get(
                "https://api.github.com/search/repositories",
                params={"q": q, "sort": "stars", "order": "desc", "per_page": max_n},
                headers=headers,
                timeout=20.0,
            )
            if r.status_code != 200:
                log.warning("github search status=%s body=%s", r.status_code, r.text[:200])
                return []
            data = r.json()
        except Exception as e:
            log.warning("github search error: %s", e)
            return []

        out: list[Candidate] = []
        for repo in data.get("items", [])[:max_n]:
            full_name = repo.get("full_name", "")
            if not full_name:
                continue
            out.append(
                Candidate(
                    uid=f"github:{full_name}",
                    source=SourceKind.GITHUB,
                    kind=CandidateKind.REPO,
                    title=full_name,
                    url=repo.get("html_url", f"https://github.com/{full_name}"),
                    summary=repo.get("description") or "",
                    content="",
                    metadata={
                        "language": repo.get("language") or "",
                        "stars": int(repo.get("stargazers_count") or 0),
                        "license": (repo.get("license") or {}).get("spdx_id"),
                        "open_issues": int(repo.get("open_issues_count") or 0),
                        "forks": int(repo.get("forks_count") or 0),
                        "topics": list(repo.get("topics") or []),
                        "source_path": "search_api",
                    },
                    published_at=_parse_dt(repo.get("created_at")),
                )
            )
        return out

    # ---- README enrichment -----------------------------------------------------------

    def _enrich_readme(self, c: Candidate) -> None:
        full_name = c.title
        token = get_settings().github_token
        headers = {"User-Agent": USER_AGENT, "Accept": "application/vnd.github.raw"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            r = httpx.get(
                f"https://api.github.com/repos/{full_name}/readme",
                headers=headers,
                timeout=15.0,
            )
            if r.status_code == 200:
                c.content = r.text[:8000]
        except Exception as e:
            log.debug("readme fetch failed for %s: %s", full_name, e)


def _safe_int(s: str) -> int:
    try:
        return int(s)
    except ValueError:
        return 0


def _parse_dt(s: str | None):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None
