from __future__ import annotations

from collections import defaultdict
import logging
from time import perf_counter
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from ragnexus.config import get_settings
from ragnexus.infrastructure.database.models.document import Document
from ragnexus.infrastructure.database.models.workspace import WorkspaceMember
from ragnexus.retrieval.advanced import ContextualCompressor, EvidenceSufficiencyChecker, MultiQueryGenerator
from ragnexus.retrieval.dense import DenseRetriever
from ragnexus.retrieval.query_rewriter import QueryRewriter
from ragnexus.retrieval.reranker import CrossEncoderReranker
from ragnexus.retrieval.result import RetrievalResult, RetrievedChunk
from ragnexus.retrieval.sparse import SparseRetriever
from ragnexus.observability.tracing import start_span

logger = logging.getLogger(__name__)


class WorkspaceAccessDenied(Exception):
    pass


class InvalidDocumentScope(Exception):
    pass


class RetrievalService:
    """Secure, configurable hybrid retrieval with bounded advanced strategies."""

    def __init__(
        self,
        *,
        embed_query,
        embedding_dimension: int = 384,
        dense_retriever: DenseRetriever | None = None,
        sparse_retriever: SparseRetriever | None = None,
        reranker: CrossEncoderReranker | None = None,
        query_rewriter: QueryRewriter | None = None,
        multi_query_generator: MultiQueryGenerator | None = None,
        compressor: ContextualCompressor | None = None,
        sufficiency_checker: EvidenceSufficiencyChecker | None = None,
        rrf_k: int = 60,
        settings=None,
    ):
        self.settings = settings or get_settings()
        self.embed_query = embed_query
        self.embedding_dimension = embedding_dimension
        self.dense = dense_retriever or DenseRetriever(embedding_dimension)
        self.sparse = sparse_retriever or SparseRetriever()
        self.reranker = reranker
        self.query_rewriter = query_rewriter or QueryRewriter()
        self.multi_query_generator = multi_query_generator or MultiQueryGenerator(
            max_queries=self.settings.MULTI_QUERY_MAX_QUERIES
        )
        self.compressor = compressor or ContextualCompressor()
        self.sufficiency_checker = sufficiency_checker or EvidenceSufficiencyChecker(
            min_token_overlap=self.settings.EVIDENCE_MIN_TOKEN_OVERLAP,
            min_dense_similarity=self.settings.EVIDENCE_MIN_DENSE_SIMILARITY,
        )
        self.rrf_k = rrf_k

    @staticmethod
    def _verify_membership(db: Session, workspace_id: UUID, user_id: UUID) -> None:
        membership = db.get(WorkspaceMember, (workspace_id, user_id))
        if membership is None:
            raise WorkspaceAccessDenied("User is not a member of this workspace")

    @staticmethod
    def _verify_document_scope(db: Session, workspace_id: UUID, document_ids: list[UUID] | None) -> None:
        if document_ids is None:
            return
        if not document_ids:
            return
        found = set(db.scalars(select(Document.id).where(
            Document.workspace_id == workspace_id,
            Document.id.in_(document_ids),
            Document.status == "indexed",
            Document.document_metadata["is_current"].as_boolean().is_(True),
        )).all())
        if found != set(document_ids):
            raise InvalidDocumentScope("One or more selected documents are unavailable, not indexed, or not current")

    @staticmethod
    def _vector_to_list(vector) -> list[float]:
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        if vector is None:
            raise ValueError("Embedding provider returned no vector")
        if len(vector) > 0 and isinstance(vector[0], (list, tuple)):
            if len(vector) != 1:
                raise ValueError("Expected one query embedding, received a batch")
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
        use_multi_query: bool | None = None,
        use_parent_expansion: bool | None = None,
        use_compression: bool | None = None,
        retrieval_mode: str = "hybrid",
    ) -> RetrievalResult:
        total_started = perf_counter()
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if candidate_k <= 0 or final_k <= 0:
            raise ValueError("candidate_k and final_k must be positive")
        if retrieval_mode not in {"dense", "bm25", "hybrid"}:
            raise ValueError("retrieval_mode must be dense, bm25, or hybrid")

        # Hard authorization boundary before any query expansion, embedding, or chunk access.
        self._verify_membership(db, workspace_id, user_id)
        self._verify_document_scope(db, workspace_id, document_ids)
        use_multi_query = self.settings.ENABLE_MULTI_QUERY if use_multi_query is None else use_multi_query
        use_parent_expansion = self.settings.ENABLE_PARENT_EXPANSION if use_parent_expansion is None else use_parent_expansion
        use_compression = self.settings.ENABLE_CONTEXTUAL_COMPRESSION if use_compression is None else use_compression

        rewrite_started = perf_counter()
        with start_span("ragnexus.retrieval.query_planning", {"ragnexus.retrieval.multi_query_enabled": bool(use_multi_query)}):
            rewritten = self.query_rewriter.rewrite(query) if use_query_rewrite else query.strip()
            queries = [rewritten]
            if use_multi_query and (len(rewritten.split()) >= 7 or any(term in rewritten.casefold() for term in ("compare", "versus", "difference", " and "))):
                queries = self.multi_query_generator.generate_queries(rewritten)
            queries = list(dict.fromkeys(q.strip() for q in queries if q and q.strip()))[:self.settings.MULTI_QUERY_MAX_QUERIES]
        result = RetrievalResult(query=query, rewritten_query=rewritten, multi_query_count=len(queries), retrieval_mode=retrieval_mode)
        result.query_planning_latency_ms = (perf_counter() - rewrite_started) * 1000

        dense_rankings: list[list[tuple[object, float]]] = []
        sparse_rankings: list[list[tuple[object, float]]] = []
        for candidate_query in queries:
            if retrieval_mode in {"dense", "hybrid"}:
                vector = self._vector_to_list(self.embed_query(candidate_query))
                if len(vector) != self.embedding_dimension:
                    raise ValueError(f"Expected {self.embedding_dimension}-dimensional embedding, received {len(vector)}")
                with start_span("ragnexus.retrieval.dense", {"ragnexus.workspace_id": str(workspace_id), "ragnexus.query_index": len(dense_rankings), "ragnexus.candidate_k": candidate_k}):
                    rows, elapsed = self.dense.search(
                        db, query_vector=vector, workspace_id=workspace_id, document_ids=document_ids,
                        top_k=candidate_k, metadata_filters=metadata_filters,
                    )
                result.dense_latency_ms += elapsed
                dense_rankings.append(rows)
            if retrieval_mode in {"bm25", "hybrid"}:
                with start_span("ragnexus.retrieval.bm25", {"ragnexus.workspace_id": str(workspace_id), "ragnexus.query_index": len(sparse_rankings), "ragnexus.candidate_k": candidate_k}):
                    rows, elapsed = self.sparse.search(
                        db, query=candidate_query, workspace_id=workspace_id, document_ids=document_ids,
                        top_k=candidate_k, metadata_filters=metadata_filters,
                    )
                result.bm25_latency_ms += elapsed
                sparse_rankings.append(rows)

        fusion_started = perf_counter()
        with start_span("ragnexus.retrieval.rrf_fusion", {"ragnexus.rrf_k": self.rrf_k}):
            by_id: dict[object, RetrievedChunk] = {}
            rrf_scores: dict[object, float] = defaultdict(float)
            for source, rankings in (("dense", dense_rankings), ("sparse", sparse_rankings)):
                for ranking in rankings:
                    for rank, (chunk, raw_score) in enumerate(ranking, start=1):
                        key = chunk.id
                        if key not in by_id:
                            by_id[key] = RetrievedChunk(chunk=chunk)
                        item = by_id[key]
                        rrf_scores[key] += 1.0 / (self.rrf_k + rank)
                        if source == "dense":
                            item.dense_score = float(raw_score) if item.dense_score is None else max(item.dense_score, float(raw_score))
                        else:
                            item.bm25_score = float(raw_score) if item.bm25_score is None else max(item.bm25_score, float(raw_score))
            fused = sorted(by_id.values(), key=lambda item: rrf_scores[item.chunk_id], reverse=True)
            for item in fused:
                item.rrf_score = rrf_scores[item.chunk_id]
        result.fusion_latency_ms = (perf_counter() - fusion_started) * 1000

        if use_reranker and self.reranker is not None and fused:
            with start_span("ragnexus.retrieval.rerank", {"ragnexus.final_k": final_k, "ragnexus.model": self.reranker.model_name}):
                result.chunks, result.rerank_latency_ms = self.reranker.rerank(rewritten, fused, top_k=final_k)
        else:
            result.chunks = fused[:final_k]

        if use_parent_expansion:
            started = perf_counter()
            with start_span("ragnexus.retrieval.parent_expansion", {"ragnexus.result_count": len(result.chunks)}):
                for item in result.chunks:
                    metadata = getattr(item.chunk, "chunk_metadata", None) or {}
                    parent = metadata.get("parent_content")
                    if parent:
                        item.context_content_override = str(parent)
            result.parent_expansion_latency_ms = (perf_counter() - started) * 1000

        if use_compression:
            started = perf_counter()
            with start_span("ragnexus.retrieval.contextual_compression", {"ragnexus.result_count": len(result.chunks)}):
                for item in result.chunks:
                    original = item.context_content
                    compressed = self.compressor.compress(rewritten, original)
                    if compressed:
                        item.context_content_override = compressed
            result.compression_latency_ms = (perf_counter() - started) * 1000

        with start_span("ragnexus.retrieval.evidence_sufficiency", {"ragnexus.result_count": len(result.chunks)}):
            result.evidence_sufficient, result.evidence_reason = self.sufficiency_checker.assess(
                rewritten, result.chunks, enabled=self.settings.ENABLE_EVIDENCE_SUFFICIENCY
            )
        result.total_latency_ms = (perf_counter() - total_started) * 1000
        logger.info("retrieval_completed", extra={
            "workspace_id": str(workspace_id), "pipeline": retrieval_mode,
            "stage": "retrieval", "latency_ms": round(result.total_latency_ms, 2),
        })
        return result
