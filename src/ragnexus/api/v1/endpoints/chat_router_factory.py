from __future__ import annotations

from typing import Callable
import re
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select

from ragnexus.dependencies import get_db
from ragnexus.memory.repository import ConversationNotFound, ConversationRepository
from ragnexus.infrastructure.database.models.document import Document, DocumentChunk
from ragnexus.rag.orchestration.service import RAGOrchestrationService
from ragnexus.schemas.chat import ChatRequest, ChatResponseSchema, CreateSessionRequest

def create_chat_router(*, get_current_user: Callable, orchestration: RAGOrchestrationService) -> APIRouter:
    router = APIRouter(prefix="/chat", tags=["Chat"])

    @router.post("/sessions", status_code=201)
    def create_session(payload: CreateSessionRequest, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
        try:
            session = ConversationRepository.create_session(
                db, workspace_id=payload.workspace_id, user_id=current_user.id,
                title=payload.title, visibility=payload.visibility,
                document_scope=payload.document_scope, document_ids=payload.document_ids,
            )
            return {
                "id": str(session.id), "workspace_id": str(session.workspace_id),
                "title": session.title, "visibility": session.visibility,
                "document_scope": session.document_scope,
            }
        except ConversationNotFound:
            raise HTTPException(status_code=404, detail="Workspace not found")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @router.get("/sessions")
    def list_sessions(workspace_id: UUID, search: str | None = None, limit: int = 50, offset: int = 0,
                      db: Session = Depends(get_db), current_user=Depends(get_current_user)):
        rows = ConversationRepository.list_sessions(db, user_id=current_user.id, workspace_id=workspace_id,
                                                     search=search, limit=limit, offset=offset)
        return [{"id": str(row.id), "workspace_id": str(row.workspace_id), "title": row.title or "New conversation",
                 "visibility": row.visibility, "document_scope": row.document_scope,
                 "created_at": row.created_at.isoformat() if row.created_at else None,
                 "updated_at": row.updated_at.isoformat() if row.updated_at else None}
                for row in rows]

    @router.patch("/sessions/{session_id}")
    def rename_session(session_id: UUID, payload: dict, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
        title = str(payload.get("title", "")).strip()
        if not title or len(title) > 255:
            raise HTTPException(status_code=422, detail="Title must contain 1 to 255 characters")
        try:
            row = ConversationRepository.rename_session(db, session_id=session_id, user_id=current_user.id, title=title)
            return {"id": str(row.id), "title": row.title}
        except ConversationNotFound:
            raise HTTPException(status_code=404, detail="Conversation not found")

    @router.delete("/sessions/{session_id}", status_code=204)
    def delete_session(session_id: UUID, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
        try:
            ConversationRepository.delete_session(db, session_id=session_id, user_id=current_user.id)
            invalidate_cache = getattr(orchestration, "invalidate_session_cache", None)
            if callable(invalidate_cache):
                invalidate_cache(user_id=current_user.id, session_id=session_id)
        except ConversationNotFound:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return None

    @router.get("/sessions/{session_id}/messages")
    def get_session_messages(session_id: UUID, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
        try:
            ConversationRepository.get_authorized_session(
                db, session_id=session_id, user_id=current_user.id
            )
            rows = ConversationRepository.get_recent_messages(
                db, session_id=session_id, limit=100
            )
            # Historical messages can outlive their source documents. Only
            # expose citations and evidence previews whose exact chunk still
            # belongs to an indexed, current document.
            original_by_message = {}
            chunk_ids = set()
            for row in rows:
                metadata = row.message_metadata or {}
                citations = metadata.get("citations") or []
                diagnostics = metadata.get("diagnostics") or {}
                retrieved = diagnostics.get("retrieved_chunks") or []
                original_by_message[str(row.id)] = (citations, retrieved)
                for item in [*citations, *retrieved]:
                    raw_chunk_id = item.get("chunk_id")
                    if raw_chunk_id:
                        try:
                            chunk_ids.add(UUID(str(raw_chunk_id)))
                        except (TypeError, ValueError):
                            continue

            valid_pairs = set()
            if chunk_ids:
                valid_pairs = {
                    (str(chunk_id), str(document_id))
                    for chunk_id, document_id in db.execute(
                        select(DocumentChunk.id, DocumentChunk.document_id)
                        .join(Document, Document.id == DocumentChunk.document_id)
                        .where(
                            DocumentChunk.id.in_(list(chunk_ids)),
                            Document.status == "indexed",
                            Document.document_metadata["is_current"].as_boolean().is_(True),
                        )
                    ).all()
                }

            payload = []
            for row in rows:
                metadata = dict(row.message_metadata or {})
                original_citations, original_retrieved = original_by_message[str(row.id)]

                valid_citations = [
                    citation for citation in original_citations
                    if (
                        str(citation.get("chunk_id")),
                        str(citation.get("document_id")),
                    ) in valid_pairs
                ]
                valid_retrieved = [
                    item for item in original_retrieved
                    if (
                        str(item.get("chunk_id")),
                        str(item.get("document_id")),
                    ) in valid_pairs
                ]
                removed = (
                    len(valid_citations) != len(original_citations)
                    or len(valid_retrieved) != len(original_retrieved)
                )

                metadata["citations"] = valid_citations
                diagnostics = dict(metadata.get("diagnostics") or {})
                if original_retrieved or "retrieved_chunks" in diagnostics:
                    diagnostics["retrieved_chunks"] = valid_retrieved
                    metadata["retrieved_count"] = len(valid_retrieved)
                diagnostics["citation_count"] = len(valid_citations)
                if removed:
                    diagnostics["historical_sources_removed"] = True
                    invalid_numbers = {
                        citation.get("source_number")
                        for citation in original_citations
                        if citation not in valid_citations and citation.get("source_number") is not None
                    }
                    content = row.content
                    for number in invalid_numbers:
                        content = re.sub(rf"\[{re.escape(str(number))}\]", "", content)
                else:
                    content = row.content

                if diagnostics:
                    metadata["diagnostics"] = diagnostics
                payload.append({
                    "id": str(row.id), "role": row.role, "content": content,
                    "message_metadata": metadata,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                })
            return payload
        except ConversationNotFound:
            raise HTTPException(status_code=404, detail="Conversation not found")

    @router.post("/messages", response_model=ChatResponseSchema)
    def send_message(payload: ChatRequest, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
        try:
            if payload.pipeline_mode == "agentic":
                agentic_chat = getattr(orchestration, "agentic_chat", None)
                if not callable(agentic_chat):
                    raise HTTPException(status_code=503, detail="Agentic RAG is not available. Check LangGraph installation and service configuration.")
                result = agentic_chat(
                    db, user_id=current_user.id, session_id=payload.session_id,
                    query=payload.query, candidate_k=payload.candidate_k,
                    final_k=payload.final_k, enable_web_search=payload.enable_web_search,
                )
            else:
                result = orchestration.chat(
                    db, user_id=current_user.id, session_id=payload.session_id,
                    query=payload.query, candidate_k=payload.candidate_k, final_k=payload.final_k,
                )
            return result.to_dict()
        except ConversationNotFound:
            raise HTTPException(status_code=404, detail="Conversation not found")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc))

    return router
