from __future__ import annotations

from time import perf_counter
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from ragnexus.infrastructure.database.models.document import Document
from ragnexus.infrastructure.database.models.workspace import WorkspaceMember
from ragnexus.retrieval.dense import DenseRetriever
from ragnexus.retrieval.fusion import reciprocal_rank_fusion
from ragnexus.retrieval.query_rewriter import QueryRewriter
from ragnexus.retrieval.reranker import CrossEncoderReranker
from ragnexus.retrieval.result import RetrievalResult
from ragnexus.retrieval.sparse import SparseRetriever


class WorkspaceAccessDenied(Exception):
    pass


class InvalidDocumentScope(Exception):
    pass


class RetrievalService:
    """Secure hybrid retrieval service for an authenticated workspace member."""

    def __init__(
        self,
        *,
        embed_query,
        embedding_dimension: int = 384,
        dense_retriever: DenseRetriever | None = None,
        sparse_retriever: SparseRetriever | None = None,
        reranker: CrossEncoderReranker | None = None,
        query_rewriter: QueryRewriter | None = None,
        rrf_k: int = 60,
    ):
        self.embed_query = embed_query
        self.embedding_dimension = embedding_dimension
        self.dense = dense_retriever or DenseRetriever(embedding_dimension)
        self.sparse = sparse_retriever or SparseRetriever()
        self.reranker = reranker
        self.query_rewriter = query_rewriter or QueryRewriter()
        self.rrf_k = rrf_k

    @staticmethod
    def _verify_membership(db: Session, workspace_id: UUID, user_id: UUID) -> None:
        membership = db.get(WorkspaceMember, (workspace_id, user_id))
        if membership is None:
            raise WorkspaceAccessDenied("User is not a member of this workspace")

    @staticmethod
    def _verify_document_scope(
        db: Session,
        workspace_id: UUID,
        document_ids: list[UUID] | None,
    ) -> None:
        if document_ids is None:
            return
        if not document_ids:
            return
        found = set(
            db.scalars(
                select(Document.id).where(
                    Document.workspace_id == workspace_id,
                    Document.id.in_(document_ids),
                    Document.status != "deleted",
                )
            ).all()
        )
        if found != set(document_ids):
            raise InvalidDocumentScope(
                "One or more selected documents do not belong to this workspace or are unavailable"
            )

    @staticmethod
    def _vector_to_list(vector) -> list[float]:
        # Supports SentenceTransformer's (1, dim) query output and flat vectors.
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        if vector and isinstance(vector[0], list):
            vector = vector[0]
        return [float(value) for value in vector]

    def search(
        self,
        db: Session,
        *,
        user_id: UUID,
        workspace_id: UUID,
        query: str,
        document_ids: list[UUID] | None = None,
        metadata_filters: dict | None = None,
        candidate_k: int = 20,
        final_k: int = 5,
        use_query_rewrite: bool = True,
        use_reranker: bool = True,
    ) -> RetrievalResult:
        total_started = perf_counter()
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if candidate_k <= 0 or final_k <= 0:
            raise ValueError("candidate_k and final_k must be positive")

        # Security boundary: verify membership before embedding or querying chunks.
        self._verify_membership(db, workspace_id, user_id)
        self._verify_document_scope(db, workspace_id, document_ids)

        rewritten = (
            self.query_rewriter.rewrite(query)
            if use_query_rewrite
            else query.strip()
        )
        result = RetrievalResult(query=query, rewritten_query=rewritten)

        query_vector = self._vector_to_list(self.embed_query(rewritten))
        if len(query_vector) != self.embedding_dimension:
            raise ValueError(
                f"Expected {self.embedding_dimension}-dimensional embedding, "
                f"received {len(query_vector)}"
            )

        dense_rows, result.dense_latency_ms = self.dense.search(
            db,
            query_vector=query_vector,
            workspace_id=workspace_id,
            document_ids=document_ids,
            top_k=candidate_k,
            metadata_filters=metadata_filters,
        )
        sparse_rows, result.bm25_latency_ms = self.sparse.search(
            db,
            query=rewritten,
            workspace_id=workspace_id,
            document_ids=document_ids,
            top_k=candidate_k,
            metadata_filters=metadata_filters,
        )

        fusion_started = perf_counter()
        fused = reciprocal_rank_fusion(
            dense_rows,
            sparse_rows,
            rrf_k=self.rrf_k,
        )
        result.fusion_latency_ms = (perf_counter() - fusion_started) * 1000

        if use_reranker and self.reranker is not None and fused:
            result.chunks, result.rerank_latency_ms = self.reranker.rerank(
                rewritten,
                fused,
                top_k=final_k,
            )
        else:
            result.chunks = fused[:final_k]

        result.total_latency_ms = (perf_counter() - total_started) * 1000
        return result
