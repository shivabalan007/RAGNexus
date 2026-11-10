from __future__ import annotations

from time import perf_counter
from uuid import UUID

from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.orm import Session

from ragnexus.infrastructure.database.models.document import Document, DocumentChunk
from ragnexus.retrieval.tokenization import tokenize


class SparseRetriever:
    """Builds a request-scoped BM25 corpus from one authorized workspace.

    This deliberately avoids the old V3 process-global BM25 index, which
    could become stale and mix tenant corpora. A later optimization can add
    a versioned workspace index/cache without changing this interface.
    """

    def __init__(self, max_corpus_chunks: int = 50000):
        self.max_corpus_chunks = max_corpus_chunks

    def search(
        self,
        db: Session,
        *,
        query: str,
        workspace_id: UUID,
        document_ids: list[UUID] | None,
        top_k: int,
        metadata_filters: dict | None = None,
    ) -> tuple[list[tuple[DocumentChunk, float]], float]:
        started = perf_counter()
        if document_ids is not None and not document_ids:
            return [], (perf_counter() - started) * 1000

        stmt = (
            select(DocumentChunk, Document.filename, Document.document_metadata)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(
                Document.workspace_id == workspace_id,
                Document.status == "indexed",
                Document.document_metadata["is_current"].as_boolean().is_(True),
            )
            .order_by(DocumentChunk.id)
            .limit(self.max_corpus_chunks + 1)
        )
        if document_ids is not None:
            stmt = stmt.where(Document.id.in_(document_ids))
        if metadata_filters:
            stmt = stmt.where(DocumentChunk.chunk_metadata.contains(metadata_filters))

        from types import SimpleNamespace
        rows = db.execute(stmt).all()
        chunks = []
        for chunk, filename, document_metadata in rows:
            chunk.document = SimpleNamespace(filename=filename, document_metadata=document_metadata or {})
            chunks.append(chunk)
        if len(chunks) > self.max_corpus_chunks:
            raise RuntimeError(
                f"Workspace BM25 corpus exceeds configured limit "
                f"({self.max_corpus_chunks}). Implement a persisted, versioned "
                "workspace index before increasing this limit."
            )

        query_tokens = tokenize(query)
        if not chunks or not query_tokens:
            return [], (perf_counter() - started) * 1000

        tokenized = [tokenize(chunk.content) for chunk in chunks]
        # BM25Okapi cannot produce useful scores if every corpus document is empty.
        if not any(tokenized):
            return [], (perf_counter() - started) * 1000

        model = BM25Okapi(tokenized)
        scores = model.get_scores(query_tokens)
        ranked = sorted(
            ((chunks[i], float(score)) for i, score in enumerate(scores) if float(score) > 0.0),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return ranked[:top_k], (perf_counter() - started) * 1000
