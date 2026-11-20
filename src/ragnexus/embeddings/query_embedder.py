from __future__ import annotations

from functools import lru_cache
from threading import RLock

import numpy as np
from sentence_transformers import SentenceTransformer

from ragnexus.config import get_settings


# Model objects are process-wide resources. RetrievalService (and therefore a
# QueryEmbedder wrapper) may be constructed more than once during the lifetime
# of the API process, so cache the actual SentenceTransformer, not only the
# wrapper returned by get_query_embedder().
_MODEL_CACHE: dict[str, SentenceTransformer] = {}
_MODEL_CACHE_LOCK = RLock()


def _load_sentence_transformer(model_name: str) -> SentenceTransformer:
    """Return one loaded embedding model per model name per process."""
    with _MODEL_CACHE_LOCK:
        model = _MODEL_CACHE.get(model_name)
        if model is None:
            model = SentenceTransformer(model_name)
            _MODEL_CACHE[model_name] = model
        return model


class QueryEmbedder:
    """Query-only adapter compatible with RetrievalService.embed_query.

    Multiple adapters share the same underlying SentenceTransformer instance,
    preventing model weights from being loaded again when services are rebuilt.
    """

    def __init__(self, model_name: str | None = None, dimension: int | None = None):
        settings = get_settings()
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.dimension = dimension or settings.EMBEDDING_DIMENSION
        self.model = _load_sentence_transformer(self.model_name)
        actual = self.model.get_sentence_embedding_dimension()
        if actual != self.dimension:
            raise ValueError(
                f"Embedding model dimension {actual} does not match configured dimension {self.dimension}"
            )

    def embed_query(self, query: str) -> list[float]:
        if not query or not query.strip():
            raise ValueError("query cannot be empty")
        vector = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False,
        )
        vector = np.asarray(vector, dtype=np.float32)
        if vector.ndim != 2 or vector.shape != (1, self.dimension):
            raise ValueError(f"Expected embedding shape (1, {self.dimension}), received {vector.shape}")
        return vector[0].tolist()


@lru_cache(maxsize=8)
def get_query_embedder() -> QueryEmbedder:
    """Return a cached default adapter (custom adapters share the same model)."""
    return QueryEmbedder()
