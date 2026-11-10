from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from ragnexus.infrastructure.database.models.conversation import SessionDocument
from ragnexus.llm.providers.base import ChatModel
from ragnexus.memory.repository import ConversationNotFound, ConversationRepository
from ragnexus.rag.orchestration.citations import validate_citations
from ragnexus.rag.orchestration.followup import FollowUpResolver
from ragnexus.rag.orchestration.prompts import build_rag_messages
from ragnexus.retrieval.result import RetrievalResult


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

    def to_dict(self) -> dict:
        return asdict(self)


class RAGOrchestrationService:
    """Coordinates session memory, follow-up resolution, retrieval and generation.

    This service expects the caller to pass an authenticated user ID. It also
    independently validates session ownership/membership before reading memory
    or retrieving documents. It does not expose an unauthenticated API route.
    """

    ABSTENTION = "The available documents do not provide enough information to answer this question."

    def __init__(self, *, retrieval_service, chat_model: ChatModel, history_limit: int = 12):
        self.retrieval_service = retrieval_service
        self.chat_model = chat_model
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
    ) -> ChatResponse:
        started = perf_counter()
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        # Authorization precedes history access and retrieval.
        chat_session = ConversationRepository.get_authorized_session(
            db, session_id=session_id, user_id=user_id
        )
        history_rows = ConversationRepository.get_recent_messages(
            db, session_id=session_id, limit=self.history_limit
        )
        history = [{"role": row.role, "content": row.content} for row in history_rows]
        ConversationRepository.append_message(
            db, session_id=session_id, role="user", content=query.strip(),
            metadata={"user_id": str(user_id)},
        )

        standalone_query = self.followup_resolver.resolve(query.strip(), history)
        selected_document_ids: list[UUID] | None = None
        if chat_session.document_scope == "selected":
            selected_document_ids = list(db.scalars(
                select(SessionDocument.document_id).where(
                    SessionDocument.session_id == session_id,
                    SessionDocument.workspace_id == chat_session.workspace_id,
                )
            ).all())
            # An empty selected scope must never fall back to workspace-wide search.
            if not selected_document_ids:
                retrieval = RetrievalResult(query=standalone_query, rewritten_query=standalone_query)
            else:
                retrieval = self.retrieval_service.search(
                    db,
                    user_id=user_id,
                    workspace_id=chat_session.workspace_id,
                    query=standalone_query,
                    document_ids=selected_document_ids,
                    candidate_k=candidate_k,
                    final_k=final_k,
                )
        else:
            retrieval = self.retrieval_service.search(
                db,
                user_id=user_id,
                workspace_id=chat_session.workspace_id,
                query=standalone_query,
                candidate_k=candidate_k,
                final_k=final_k,
            )

        model_name = None
        prompt_tokens = None
        completion_tokens = None
        abstained = not retrieval.has_documents()
        if abstained:
            answer = self.ABSTENTION
            citations: list[dict] = []
        else:
            messages = build_rag_messages(
                query=query.strip(), retrieval=retrieval, history=history
            )
            generated = self.chat_model.complete(messages, temperature=0.2, max_tokens=1400)
            answer, citations = validate_citations(generated.text, retrieval)
            if not answer:
                answer = self.ABSTENTION
                citations = []
                abstained = True
            model_name = generated.model
            prompt_tokens = generated.prompt_tokens
            completion_tokens = generated.completion_tokens

        metadata = {
            "citations": citations,
            "retrieved_count": retrieval.retrieved_count,
            "abstained": abstained,
            "rewritten_query": standalone_query,
            "latency_ms": round((perf_counter() - started) * 1000, 2),
            "model": model_name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
        ConversationRepository.append_message(
            db, session_id=session_id, role="assistant", content=answer, metadata=metadata
        )
        return ChatResponse(
            session_id=str(session_id),
            answer=answer,
            citations=citations,
            rewritten_query=standalone_query,
            retrieved_count=retrieval.retrieved_count,
            abstained=abstained,
            latency_ms=metadata["latency_ms"],
            model=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
