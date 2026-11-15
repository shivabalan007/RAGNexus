from __future__ import annotations

from typing import Callable
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ragnexus.dependencies import get_db
from ragnexus.memory.repository import ConversationNotFound, ConversationRepository
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
            return [{
                "id": str(row.id), "role": row.role, "content": row.content,
                "message_metadata": row.message_metadata or {},
                "created_at": row.created_at.isoformat() if row.created_at else None,
            } for row in rows]
        except ConversationNotFound:
            raise HTTPException(status_code=404, detail="Conversation not found")

    @router.post("/messages", response_model=ChatResponseSchema)
    def send_message(payload: ChatRequest, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
        try:
            result = orchestration.chat(
                db, user_id=current_user.id, session_id=payload.session_id,
                query=payload.query, candidate_k=payload.candidate_k, final_k=payload.final_k,
                pipeline_mode=payload.pipeline_mode, enable_web_search=payload.enable_web_search,
            )
            return result.to_dict()
        except ConversationNotFound:
            raise HTTPException(status_code=404, detail="Conversation not found")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    return router
