from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


class NoopCache:
    """Cache fallback used when Redis is not configured or is unavailable."""

    enabled = False
    available = False

    def get(self, key: str):
        return None

    def set(self, key: str, value: Any, ttl: int = 300):
        return False

    def delete(self, key: str):
        return False


class RedisJsonCache:
    """Small JSON cache with fail-open behaviour.

    PostgreSQL remains the source of truth. Redis failures must never prevent
    chat, retrieval, ingestion or provider execution from continuing.
    """

    enabled = True

    def __init__(self, url: str, default_ttl: int = 1800, retry_interval_seconds: float = 5.0):
        from redis import Redis

        self.client = Redis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
            health_check_interval=30,
        )
        self.default_ttl = max(1, int(default_ttl))
        self.retry_interval_seconds = max(1.0, float(retry_interval_seconds))
        self.available = True
        self._retry_after = 0.0

    def _can_attempt(self) -> bool:
        return self.available or time.monotonic() >= self._retry_after

    def _mark_success(self) -> None:
        self.available = True
        self._retry_after = 0.0

    def _mark_failure(self) -> None:
        self.available = False
        self._retry_after = time.monotonic() + self.retry_interval_seconds

    def get(self, key: str):
        if not self._can_attempt():
            return None
        try:
            raw = self.client.get(key)
            self._mark_success()
        except Exception as exc:
            self._mark_failure()
            logger.debug("Redis cache read unavailable (%s)", type(exc).__name__)
            return None
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            # Corrupt entries are disposable cache data, not application state.
            self.delete(key)
            return None

    def set(self, key: str, value: Any, ttl: int | None = None):
        if not self._can_attempt():
            return False
        try:
            encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
            self.client.setex(key, max(1, int(ttl or self.default_ttl)), encoded)
            self._mark_success()
            return True
        except Exception as exc:
            self._mark_failure()
            logger.debug("Redis cache write unavailable (%s)", type(exc).__name__)
            return False

    def delete(self, key: str):
        if not self._can_attempt():
            return False
        try:
            self.client.delete(key)
            self._mark_success()
            return True
        except Exception as exc:
            self._mark_failure()
            logger.debug("Redis cache delete unavailable (%s)", type(exc).__name__)
            return False


def build_cache(settings):
    """Build a lazy Redis cache; cache outages do not disable the application."""
    if not getattr(settings, "REDIS_CACHE_ENABLED", True) or not settings.REDIS_URL:
        return NoopCache()
    try:
        # Do not ping here. Redis may become available after API startup and the
        # client reconnects on subsequent operations. Every operation fails open.
        return RedisJsonCache(settings.REDIS_URL, settings.REDIS_TTL_SECONDS)
    except Exception as exc:
        logger.warning("Redis cache could not be initialized (%s); caching disabled", type(exc).__name__)
        return NoopCache()


def make_cache_key(namespace: str, payload: Any) -> str:
    """Return a deterministic, non-reversible cache key for structured input."""
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return f"ragnexus:{namespace}:v1:{digest}"


class CachedEmbeddingFunction:
    """Cache query vectors by embedding model + normalized query."""

    def __init__(
        self,
        embed_query: Callable[[str], list[float]],
        *,
        model_name: str,
        cache,
        ttl: int = 1800,
        embedding_dimension: int | None = None,
    ):
        self.embed_query = embed_query
        self.model_name = model_name
        self.embedding_dimension = embedding_dimension
        self.cache = cache
        self.ttl = max(1, int(ttl))

    def __call__(self, query: str) -> list[float]:
        if not query or not query.strip():
            raise ValueError("query cannot be empty")
        normalized = " ".join(query.split())
        key = make_cache_key("query-embedding", {
            "model": self.model_name,
            "dimension": self.embedding_dimension,
            "query": normalized,
        })
        try:
            cached = self.cache.get(key)
        except Exception:
            cached = None
        if isinstance(cached, list) and cached:
            try:
                return [float(value) for value in cached]
            except (TypeError, ValueError):
                self.cache.delete(key)
        vector = self.embed_query(query)
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        # Cache JSON-native floats. Serializing a numpy array with json's
        # default=str produces a string, which can never be read as a vector.
        if vector and isinstance(vector[0], (list, tuple)):
            if len(vector) != 1:
                raise ValueError("Expected one query embedding, received a batch")
            vector = vector[0]
        vector = [float(value) for value in vector]
        try:
            self.cache.set(key, vector, self.ttl)
        except Exception:
            logger.debug("Embedding cache write failed", exc_info=True)
        return vector
