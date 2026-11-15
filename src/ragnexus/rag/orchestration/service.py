from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from time import perf_counter
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from ragnexus.infrastructure.database.models.conversation import SessionDocument
from ragnexus.llm.providers.base import ChatModel
from ragnexus.memory.repository import ConversationRepository
from ragnexus.rag.orchestration.citations import validate_citations
from ragnexus.rag.orchestration.followup import FollowUpResolver
from ragnexus.rag.orchestration.prompts import build_rag_messages
from ragnexus.retrieval.result import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ChatResponse:
    session_id: str
    answer: str
    citations: list[dict]
    rewritten_query: str
    retrieved_count: int
    abstained: bool
    latency_ms: float
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    diagnostics: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class RAGOrchestrationService:
    """Coordinates authorization, session memory, retrieval and response generation."""

    ABSTENTION = "The available documents do not provide enough information to answer this question."

    def __init__(self, *, retrieval_service, chat_model: ChatModel, agentic_workflow=None, history_limit: int = 12):
        self.retrieval_service = retrieval_service
        self.chat_model = chat_model
        self.agentic_workflow = agentic_workflow
        self.history_limit = history_limit
        self.followup_resolver = FollowUpResolver(chat_model)

    def chat(
        self,
        db: Session,
        *,
        user_id: UUID,
        session_id: UUID,
        query: str,
        candidate_k: int = 20,
        final_k: int = 5,
        pipeline_mode: str = "standard",
        enable_web_search: bool = False,
    ) -> ChatResponse:
        started = perf_counter()
        if not query or not query.strip():
            raise ValueError("query cannot be empty")
        if pipeline_mode not in {"standard", "agentic"}:
            raise ValueError("pipeline_mode must be 'standard' or 'agentic'")

        # Authorization precedes history access and retrieval.
        chat_session = ConversationRepository.get_authorized_session(db, session_id=session_id, user_id=user_id)
        history_rows = ConversationRepository.get_recent_messages(db, session_id=session_id, limit=self.history_limit)
        history = [{"role": row.role, "content": row.content} for row in history_rows]
        ConversationRepository.append_message(
            db, session_id=session_id, role="user", content=query.strip(),
            metadata={"user_id": str(user_id), "pipeline_mode": pipeline_mode},
        )

        standalone_query = self.followup_resolver.resolve(query.strip(), history)
        selected_document_ids: list[UUID] | None = None
        if chat_session.document_scope == "selected":
            selected_document_ids = list(db.scalars(select(SessionDocument.document_id).where(
                SessionDocument.session_id == session_id,
                SessionDocument.workspace_id == chat_session.workspace_id,
            )).all())
            # Never convert an empty selected-document scope into workspace-wide retrieval.
            if not selected_document_ids:
                retrieval = RetrievalResult(query=standalone_query, rewritten_query=standalone_query)
            else:
                retrieval = None
        else:
            retrieval = None

        model_name = None
        prompt_tokens = None
        completion_tokens = None
        citations: list[dict] = []
        abstained = False
        extra_diagnostics: dict = {}
        answer = self.ABSTENTION
        rewritten_query = standalone_query

        if pipeline_mode == "agentic":
            if self.agentic_workflow is None:
                raise RuntimeError("Agentic workflow is not configured. Install the agentic dependencies and restart the API.")
            if retrieval is None:
                state = self.agentic_workflow.invoke(
                    db=db, user_id=user_id, workspace_id=chat_session.workspace_id,
                    query=standalone_query, document_ids=selected_document_ids,
                    candidate_k=candidate_k, final_k=final_k,
                    enable_web_search=enable_web_search, history=history,
                )
                retrieval = state.get("retrieval") or RetrievalResult(query=standalone_query)
                answer = state.get("answer") or self.ABSTENTION
                citations = state.get("citations") or []
                abstained = bool(state.get("abstained", False))
                model_name = state.get("model")
                prompt_tokens = state.get("prompt_tokens")
                completion_tokens = state.get("completion_tokens")
                rewritten_query = str(state.get("query") or standalone_query)
                extra_diagnostics = {
                    "agent_route": state.get("route"),
                    "agent_retries": state.get("retry_count", 0),
                    "web_result_count": len(state.get("web_sources") or []),
                    "web_error": state.get("web_error"),
                    "answer_verification": state.get("verification"),
                    **(state.get("diagnostics") or {}),
                }
        else:
            if retrieval is None:
                retrieval = self.retrieval_service.search(
                    db, user_id=user_id, workspace_id=chat_session.workspace_id,
                    query=standalone_query, document_ids=selected_document_ids,
                    candidate_k=candidate_k, final_k=final_k,
                )
            if not retrieval.has_documents() or not retrieval.evidence_sufficient:
                abstained = True
                answer = self.ABSTENTION
            else:
                messages = build_rag_messages(query=query.strip(), retrieval=retrieval, history=history)
                generated = self.chat_model.complete(messages, temperature=0.2, max_tokens=1400)
                answer, citations = validate_citations(generated.text, retrieval)
                if not answer:
                    answer = self.ABSTENTION
                    citations = []
                    abstained = True
                model_name = generated.model
                prompt_tokens = generated.prompt_tokens
                completion_tokens = generated.completion_tokens

        elapsed = round((perf_counter() - started) * 1000, 2)
        diagnostics = {
            "dense_latency_ms": round(retrieval.dense_latency_ms, 2),
            "bm25_latency_ms": round(retrieval.bm25_latency_ms, 2),
            "fusion_latency_ms": round(retrieval.fusion_latency_ms, 2),
            "rerank_latency_ms": round(retrieval.rerank_latency_ms, 2),
            "query_planning_latency_ms": round(retrieval.query_planning_latency_ms, 2),
            "parent_expansion_latency_ms": round(retrieval.parent_expansion_latency_ms, 2),
            "compression_latency_ms": round(retrieval.compression_latency_ms, 2),
            "retrieval_latency_ms": round(retrieval.total_latency_ms, 2),
            "retrieval_mode": retrieval.retrieval_mode,
            "multi_query_count": retrieval.multi_query_count,
            "evidence_sufficient": retrieval.evidence_sufficient,
            "evidence_reason": retrieval.evidence_reason,
            "document_scope": chat_session.document_scope,
            "visibility": chat_session.visibility,
            "pipeline_mode": pipeline_mode,
            "citation_count": len(citations),
            **extra_diagnostics,
        }
        metadata = {
            "citations": citations, "retrieved_count": retrieval.retrieved_count,
            "abstained": abstained, "rewritten_query": rewritten_query,
            "latency_ms": elapsed, "model": model_name,
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "diagnostics": diagnostics,
        }
        ConversationRepository.append_message(
            db, session_id=session_id, role="assistant", content=answer, metadata=metadata,
        )
        logger.info("chat_completed", extra={
            "workspace_id": str(chat_session.workspace_id), "session_id": str(session_id),
            "pipeline": pipeline_mode, "stage": "chat", "latency_ms": elapsed,
        })
        return ChatResponse(
            session_id=str(session_id), answer=answer, citations=citations,
            rewritten_query=rewritten_query, retrieved_count=retrieval.retrieved_count,
            abstained=abstained, latency_ms=elapsed, model=model_name,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            diagnostics=diagnostics,
        )
