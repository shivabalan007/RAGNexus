from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RetrievedChunk:
    chunk: Any
    dense_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None

    @property
    def content(self) -> str:
        return self.chunk.content

    @property
    def chunk_id(self):
        return self.chunk.id

    @property
    def document_id(self):
        return self.chunk.document_id

    @property
    def metadata(self) -> dict:
        chunk_meta = getattr(self.chunk, "chunk_metadata", None) or {}
        doc = getattr(self.chunk, "document", None)
        doc_meta = getattr(doc, "document_metadata", None) or {}
        return {
            **chunk_meta,
            "document_id": str(self.chunk.document_id),
            "chunk_id": str(self.chunk.id),
            "chunk_index": self.chunk.chunk_index,
            "filename": getattr(doc, "filename", None),
            "document_metadata": doc_meta,
        }


@dataclass(slots=True)
class RetrievalResult:
    query: str
    rewritten_query: str | None = None
    chunks: list[RetrievedChunk] = field(default_factory=list)
    dense_latency_ms: float = 0.0
    bm25_latency_ms: float = 0.0
    fusion_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    @property
    def retrieved_count(self) -> int:
        return len(self.chunks)

    def get_chunks(self) -> list[Any]:
        return [item.chunk for item in self.chunks]

    def has_documents(self) -> bool:
        return bool(self.chunks)

    def citations(self) -> list[dict]:
        return [
            {
                "document_id": str(item.document_id),
                "chunk_id": str(item.chunk_id),
                "chunk_index": item.chunk.chunk_index,
                "filename": getattr(getattr(item.chunk, "document", None), "filename", None),
                "content": item.content,
            }
            for item in self.chunks
        ]
