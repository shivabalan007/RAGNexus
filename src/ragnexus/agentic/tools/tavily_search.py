from __future__ import annotations

import logging
import re
from time import perf_counter
from typing import Any
from urllib.parse import urlparse

from ragnexus.infrastructure.cache.redis_cache import make_cache_key
from ragnexus.observability.tracing import start_span

logger = logging.getLogger(__name__)


class TavilySearchTool:
    """Read-only Tavily adapter with optional Redis result caching."""

    def __init__(
        self,
        api_key: str | None,
        *,
        max_results: int = 5,
        search_depth: str = "basic",
        cache=None,
        cache_ttl_seconds: int = 900,
        fresh_cache_ttl_seconds: int = 60,
    ):
        if hasattr(api_key, "get_secret_value"):
            api_key = api_key.get_secret_value()
        self.api_key = str(api_key).strip() if api_key else None
        self.max_results = max(1, min(int(max_results), 10))
        self.search_depth = search_depth if search_depth in {"basic", "advanced"} else "basic"
        self.cache = cache
        self.cache_ttl_seconds = max(30, int(cache_ttl_seconds))
        self.fresh_cache_ttl_seconds = max(15, int(fresh_cache_ttl_seconds))
        self._client = None

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _get_client(self):
        if not self.api_key:
            raise RuntimeError("TAVILY_API_KEY is not configured")
        if self._client is None:
            try:
                from tavily import TavilyClient
            except ImportError as exc:
                raise RuntimeError("Install tavily-python to enable Tavily web search") from exc
            self._client = TavilyClient(api_key=self.api_key)
        return self._client

    @staticmethod
    def _normalize_results(response: dict | None) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in (response or {}).get("results", []):
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            content = str(item.get("content") or "").strip()
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc or not content or url in seen:
                continue
            seen.add(url)
            normalized.append({
                "title": str(item.get("title") or url)[:300],
                "url": url,
                "content": content[:5000],
                "score": item.get("score"),
                "source_type": "web",
            })
        return normalized

    def search_with_metadata(self, query: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not query or not query.strip():
            return [], {"provider": "Tavily", "status": "empty_query", "cache_hit": False, "latency_ms": 0.0}
        normalized_query = " ".join(query.casefold().split())
        freshness_sensitive = bool(re.search(
            r"\b(latest|current|currently|today|tonight|yesterday|tomorrow|recent|breaking|news|price now)\b",
            normalized_query,
        ))
        cache_ttl = (
            min(self.cache_ttl_seconds, self.fresh_cache_ttl_seconds)
            if freshness_sensitive else self.cache_ttl_seconds
        )
        key = make_cache_key("tavily-search", {
            "query": normalized_query,
            "search_depth": self.search_depth,
            "max_results": self.max_results,
        })
        started = perf_counter()
        if self.cache is not None:
            try:
                cached = self.cache.get(key)
            except Exception:
                cached = None
            if isinstance(cached, list):
                return cached, {
                    "provider": "Tavily", "status": "cache_hit", "cache_hit": True,
                    "result_count": len(cached),
                    "latency_ms": round((perf_counter() - started) * 1000, 2),
                }

        with start_span("ragnexus.tavily.search", {
            "ragnexus.tavily.max_results": self.max_results,
            "ragnexus.tavily.search_depth": self.search_depth,
        }):
            response = self._get_client().search(
                query=query.strip(),
                search_depth=self.search_depth,
                max_results=self.max_results,
                include_answer=False,
                include_raw_content=False,
            )
        normalized = self._normalize_results(response)
        if self.cache is not None:
            try:
                # Also cache an empty successful result briefly to avoid repeating
                # an expensive no-result query on every rerun.
                self.cache.set(key, normalized, cache_ttl)
            except Exception:
                logger.debug("Tavily cache write failed", exc_info=True)
        return normalized, {
            "provider": "Tavily", "status": "success" if normalized else "no_results",
            "cache_hit": False, "result_count": len(normalized),
            "latency_ms": round((perf_counter() - started) * 1000, 2),
        }

    def search(self, query: str) -> list[dict[str, Any]]:
        """Backward-compatible list-only interface."""
        return self.search_with_metadata(query)[0]
