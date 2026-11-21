from __future__ import annotations

from typing import Callable
import re
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select, text

from ragnexus.dependencies import get_db
from ragnexus.memory.repository import ConversationNotFound, ConversationRepository
from ragnexus.infrastructure.database.models.document import Document, DocumentChunk
from ragnexus.infrastructure.database.models.conversation import Message
from ragnexus.rag.orchestration.service import RAGOrchestrationService
from ragnexus.rag.orchestration.citations import filter_active_citations
from ragnexus.schemas.chat import ChatRequest, ChatResponseSchema, CreateSessionRequest




def _find_idempotent_chat_response(db: Session, *, session_id: UUID, request_id: UUID):
    rows = db.scalars(
        select(Message)
        .where(
            Message.session_id == session_id,
            Message.message_metadata["client_request_id"].as_string() == str(request_id),
        )
        .order_by(Message.created_at.asc(), Message.id.asc())
    ).all()
    assistant = next((row for row in reversed(rows) if row.role == "assistant"), None)
    if assistant is None:
        if rows:
            raise HTTPException(
                status_code=409,
                detail="This chat request has already started and is still pending or failed. "
                       "Refresh the conversation before submitting it again.",
            )
        return None

    metadata = assistant.message_metadata or {}
    diagnostics = metadata.get("diagnostics") or {}
    user_message = next((row for row in rows if row.role == "user"), None)
    return {
        "session_id": str(session_id),
        "answer": assistant.content,
        "citations": metadata.get("citations") or [],
        "rewritten_query": metadata.get("rewritten_query") or (
            user_message.content if user_message is not None else ""
        ),
        "retrieved_count": int(metadata.get("retrieved_count") or 0),
        "abstained": bool(metadata.get("abstained", False)),
        "latency_ms": float(metadata.get("latency_ms") or diagnostics.get("total_latency_ms") or 0),
        "model": metadata.get("model"),
        "prompt_tokens": metadata.get("prompt_tokens"),
        "completion_tokens": metadata.get("completion_tokens"),
        "retry_count": int(metadata.get("retry_count") or diagnostics.get("retry_count") or 0),
        "diagnostics": diagnostics,
    }


def _acquire_request_lock(db: Session, *, session_id: UUID, request_id: UUID):
    """Hold a PostgreSQL session advisory lock across internal DB commits."""
    bind = db.get_bind()
    if getattr(getattr(bind, "dialect", None), "name", None) != "postgresql":
        return None, None
    connection = bind.connect()
    lock_key = f"ragnexus-chat:{session_id}:{request_id}"
    try:
        connection.execute(
            text("SELECT pg_advisory_lock(hashtextextended(:lock_key, 0))"),
            {"lock_key": lock_key},
        )
        connection.commit()
        return connection, lock_key
    except Exception:
        connection.close()
        raise


def _release_request_lock(connection, lock_key: str | None) -> None:
    if connection is None or not lock_key:
        return
    try:
        connection.execute(
            text("SELECT pg_advisory_unlock(hashtextextended(:lock_key, 0))"),
            {"lock_key": lock_key},
        )
        connection.commit()
    finally:
        connection.close()


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
                    raw_ids = [item.get("chunk_id"), *(item.get("supporting_chunk_ids") or [])]
                    for raw_chunk_id in raw_ids:
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

                valid_citations = filter_active_citations(original_citations, valid_pairs)
                valid_retrieved = [
                    item for item in original_retrieved
                    if (
                        str(item.get("chunk_id")),
                        str(item.get("document_id")),
                    ) in valid_pairs
                ]
                removed = (
                    valid_citations != original_citations
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
                    valid_source_numbers = {
                        citation.get("source_number") for citation in valid_citations
                    }
                    invalid_numbers = {
                        citation.get("source_number")
                        for citation in original_citations
                        if citation.get("source_number") is not None
                        and citation.get("source_number") not in valid_source_numbers
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
        lock_connection = None
        lock_key = None
        try:
            ConversationRepository.get_authorized_session(
                db, session_id=payload.session_id, user_id=current_user.id
            )
            if payload.request_id is not None:
                lock_connection, lock_key = _acquire_request_lock(
                    db, session_id=payload.session_id, request_id=payload.request_id
                )
                existing = _find_idempotent_chat_response(
                    db, session_id=payload.session_id, request_id=payload.request_id
                )
                if existing is not None:
                    return existing

            if payload.pipeline_mode == "agentic":
                agentic_chat = getattr(orchestration, "agentic_chat", None)
                if not callable(agentic_chat):
                    raise HTTPException(status_code=503, detail="Agentic RAG is not available. Check LangGraph installation and service configuration.")
                result = agentic_chat(
                    db, user_id=current_user.id, session_id=payload.session_id,
                    query=payload.query, candidate_k=payload.candidate_k,
                    final_k=payload.final_k, enable_web_search=payload.enable_web_search,
                    request_id=payload.request_id,
                )
            else:
                result = orchestration.chat(
                    db, user_id=current_user.id, session_id=payload.session_id,
                    query=payload.query, candidate_k=payload.candidate_k, final_k=payload.final_k,
                    request_id=payload.request_id,
                )
            return result.to_dict()
        except ConversationNotFound:
            raise HTTPException(status_code=404, detail="Conversation not found")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        finally:
            _release_request_lock(lock_connection, lock_key)

    return router
