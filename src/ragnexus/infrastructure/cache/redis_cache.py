from __future__ import annotations

import hashlib
import json
import logging
from typing import Callable

logger = logging.getLogger(__name__)


class NoopCache:
    def get(self, key: str):
        return None
    def set(self, key: str, value, ttl: int = 300):
        return False


class RedisJsonCache:
    def __init__(self, url: str, default_ttl: int = 1800):
        from redis import Redis
        self.client = Redis.from_url(url, decode_responses=True, socket_connect_timeout=2, socket_timeout=2)
        self.default_ttl = default_ttl

    def get(self, key: str):
        raw = self.client.get(key)
        return json.loads(raw) if raw else None

    def set(self, key: str, value, ttl: int | None = None):
        self.client.setex(key, ttl or self.default_ttl, json.dumps(value))
        return True


def build_cache(settings):
    if not settings.REDIS_URL:
        return NoopCache()
    try:
        cache = RedisJsonCache(settings.REDIS_URL, settings.REDIS_TTL_SECONDS)
        cache.client.ping()
        return cache
    except Exception as exc:
        logger.warning("Redis unavailable; caching disabled (%s)", type(exc).__name__)
        return NoopCache()


class CachedEmbeddingFunction:
    """Cache query vectors by model + normalized query; raw queries are not cache keys."""
    def __init__(self, embed_query: Callable[[str], list[float]], *, model_name: str, cache, ttl: int = 1800):
        self.embed_query = embed_query
        self.model_name = model_name
        self.cache = cache
        self.ttl = ttl

    def __call__(self, query: str) -> list[float]:
        normalized = " ".join(query.casefold().split())
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        key = f"ragnexus:query-embedding:{self.model_name}:{digest}"
        try:
            cached = self.cache.get(key)
        except Exception:
            cached = None
        if cached is not None:
            return [float(v) for v in cached]
        vector = self.embed_query(query)
        try:
            self.cache.set(key, vector, self.ttl)
        except Exception:
            logger.debug("Embedding cache write failed", exc_info=True)
        return vector
