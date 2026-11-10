import re
import numpy as np
from ragnexus.ingestion.base import Chunk, LoadedDocument

class SemanticChunker:
    """Groups adjacent sentences using cosine similarity of sentence embeddings."""
    def __init__(self, embedder, similarity_threshold: float = 0.70, max_chars: int = 1800):
        if embedder is None:
            raise ValueError("SemanticChunker requires an embedder")
        if not -1.0 <= similarity_threshold <= 1.0 or max_chars < 100:
            raise ValueError("Invalid semantic chunker configuration")
        self.embedder = embedder
        self.threshold = similarity_threshold
        self.max_chars = max_chars

    def split(self, documents: list[LoadedDocument]) -> list[Chunk]:
        output: list[Chunk] = []
        index = 0
        for doc in documents:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", doc.text) if s.strip()]
            if not sentences:
                continue
            vectors = np.asarray(self.embedder.embed_documents(sentences), dtype=np.float32)
            groups: list[list[str]] = [[sentences[0]]]
            for i in range(1, len(sentences)):
                a, b = vectors[i - 1], vectors[i]
                denom = float(np.linalg.norm(a) * np.linalg.norm(b))
                similarity = float(np.dot(a, b) / denom) if denom else 0.0
                candidate_len = len(" ".join(groups[-1])) + 1 + len(sentences[i])
                if similarity < self.threshold or candidate_len > self.max_chars:
                    groups.append([sentences[i]])
                else:
                    groups[-1].append(sentences[i])
            for group in groups:
                text = " ".join(group).strip()
                if text:
                    output.append(Chunk(text, index, dict(doc.metadata)))
                    index += 1
        return output


"""
Splits text at semantic boundaries — paragraph breaks and topic shifts. Preserves meaning by keeping related sentences together in the same chunk.
"""