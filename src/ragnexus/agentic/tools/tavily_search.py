from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from ragnexus.observability.tracing import start_span


class TavilySearchTool:
    """Read-only Tavily search adapter. Web results are always treated as untrusted evidence."""

    def __init__(self, api_key: str | None, *, max_results: int = 5, search_depth: str = "basic"):
        self.api_key = api_key
        self.max_results = max(1, min(int(max_results), 10))
        self.search_depth = search_depth
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

    def search(self, query: str) -> list[dict[str, Any]]:
        if not query or not query.strip():
            return []
        with start_span("ragnexus.tavily.search", {"ragnexus.tavily.max_results": self.max_results, "ragnexus.tavily.search_depth": self.search_depth}):
            response = self._get_client().search(
                query=query.strip(),
                search_depth=self.search_depth,
                max_results=self.max_results,
                include_answer=False,
                include_raw_content=False,
            )
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in (response or {}).get("results", []):
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
