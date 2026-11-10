from functools import lru_cache
import numpy as np
from sentence_transformers import SentenceTransformer
from ragnexus.config import get_settings

class Embedder:
    def __init__(self, model_name: str | None = None):
        settings = get_settings()
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.dimension = settings.EMBEDDING_DIMENSION
        self.model = SentenceTransformer(self.model_name)
        actual = self.model.get_sentence_embedding_dimension()
        if actual != self.dimension:
            raise ValueError(f"Embedding model dimension {actual} does not match database dimension {self.dimension}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError("Unexpected embedding dimensions")
        return vectors.astype(np.float32).tolist()

@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder()



"""
Loads sentence transformer model and provides embed_documents() and embed_query() methods. Converts text chunks and queries into 384-dimensional vectors.
"""