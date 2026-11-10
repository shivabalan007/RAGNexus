from __future__ import annotations

from time import perf_counter
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from ragnexus.infrastructure.database.models.document import Document, DocumentChunk


class DenseRetriever:
    """Workspace-scoped pgvector cosine retrieval.

    The caller supplies an authenticated user and workspace. Membership is
    checked by RetrievalService before this class is invoked; the SQL query
    independently constrains every result to that workspace and optional
    document allow-list.
    """

    def __init__(self, embedding_dimension: int = 384):
        self.embedding_dimension = embedding_dimension

    def search(
        self,
        db: Session,
        *,
        query_vector: list[float],
        workspace_id: UUID,
        document_ids: list[UUID] | None,
        top_k: int,
        metadata_filters: dict | None = None,
    ) -> tuple[list[tuple[DocumentChunk, float]], float]:
        if len(query_vector) != self.embedding_dimension:
            raise ValueError(
                f"Expected {self.embedding_dimension}-dimensional query vector; "
                f"received {len(query_vector)}"
            )

        started = perf_counter()
        distance = DocumentChunk.embedding.cosine_distance(query_vector)
        stmt = (
            select(DocumentChunk, Document.filename, Document.document_metadata, (1.0 - distance).label("similarity"))
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(
                Document.workspace_id == workspace_id,
                Document.status == "indexed",
                Document.document_metadata["is_current"].as_boolean().is_(True),
                DocumentChunk.embedding.is_not(None),
            )
        )
        if document_ids is not None:
            if not document_ids:
                return [], (perf_counter() - started) * 1000
            stmt = stmt.where(Document.id.in_(document_ids))
        if metadata_filters:
            stmt = stmt.where(DocumentChunk.chunk_metadata.contains(metadata_filters))

        rows = db.execute(stmt.order_by(distance.asc()).limit(top_k)).all()
        from types import SimpleNamespace
        results = []
        for chunk, filename, document_metadata, score in rows:
            chunk.document = SimpleNamespace(filename=filename, document_metadata=document_metadata or {})
            results.append((chunk, float(score)))
        return results, (perf_counter() - started) * 1000
