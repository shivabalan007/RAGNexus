from dataclasses import dataclass, field
from typing import List, Optional, Any


@dataclass
class RetrievedChunk:
    """
    Stores one retrieved document together with
    all scores collected during retrieval.
    """

    chunk: Any

    vector_score: float = 0.0

    bm25_score: float = 0.0

    rerank_score: float = 0.0


@dataclass
class RetrievalResult:
    """
    Stores the complete output of the retrieval stage.

    This object is passed through the pipeline so
    Router, Monitoring, Memory and Evaluation
    can reuse the same retrieval information.
    """

    # Retrieved chunks with all scores
    retrieved_chunks: List[RetrievedChunk] = field(default_factory=list)

    # Metadata
    retrieved_count: int = 0

    retrieval_latency: float = 0.0

    rerank_latency: float = 0.0

    # --------------------------
    # Helper Methods
    # --------------------------

    def add_chunk(
        self,
        chunk,
        vector_score: float = 0.0,
        bm25_score: float = 0.0,
    ):

        self.retrieved_chunks.append(
            RetrievedChunk(
                chunk=chunk,
                vector_score=vector_score,
                bm25_score=bm25_score,
            )
        )

        self.retrieved_count = len(self.retrieved_chunks)

    def get_chunks(self):

        return [
            item.chunk
            for item in self.retrieved_chunks
        ]

    def top_vector_score(self):

        if not self.retrieved_chunks:
            return 0.0

        return max(
            item.vector_score
            for item in self.retrieved_chunks
        )

    def top_bm25_score(self):

        if not self.retrieved_chunks:
            return 0.0

        return max(
            item.bm25_score
            for item in self.retrieved_chunks
        )

    def top_rerank_score(self):

        if not self.retrieved_chunks:
            return 0.0

        return max(
            item.rerank_score
            for item in self.retrieved_chunks
        )

    def has_documents(self):

        return len(self.retrieved_chunks) > 0
    
"""
Stores all retrieval outputs (chunks, vector scores, BM25 scores, rerank scores, latency, metadata) in a single object. Prevents information loss between retrieval stages and supports routing, monitoring, memory, and future production features.
"""