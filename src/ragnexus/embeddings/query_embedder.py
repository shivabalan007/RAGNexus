from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from ragnexus.config import get_settings


class QueryEmbedder:
    """Small query-only adapter compatible with RetrievalService.embed_query."""

    def __init__(self, model_name: str | None = None, dimension: int | None = None):
        settings = get_settings()
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.dimension = dimension or settings.EMBEDDING_DIMENSION
        self.model = SentenceTransformer(self.model_name)
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


@lru_cache(maxsize=1)
def get_query_embedder() -> QueryEmbedder:
    return QueryEmbedder()
