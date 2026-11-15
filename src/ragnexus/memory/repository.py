from __future__ import annotations

from uuid import UUID
from datetime import datetime, timezone

from sqlalchemy import select, or_
from sqlalchemy.orm import Session

from ragnexus.infrastructure.database.models.conversation import Message, Session as ChatSession


class ConversationNotFound(PermissionError):
    """Used for missing or inaccessible sessions to avoid leaking their existence."""


class ConversationRepository:
    """PostgreSQL-backed durable conversation history.

    PostgreSQL is the source of truth. Redis caching can be added later without
    changing the repository contract.
    """

    @staticmethod
    def get_authorized_session(db: Session, *, session_id: UUID, user_id: UUID) -> ChatSession:
        session = db.get(ChatSession, session_id)
        if session is None:
            raise ConversationNotFound("Conversation not found")

        from ragnexus.infrastructure.database.models.workspace import WorkspaceMember
        membership = db.get(WorkspaceMember, (session.workspace_id, user_id))
        if membership is None:
            raise ConversationNotFound("Conversation not found")

        if session.visibility == "private" and session.created_by_user_id != user_id:
            raise ConversationNotFound("Conversation not found")

        return session

    @staticmethod
    def create_session(
        db: Session,
        *,
        workspace_id: UUID,
        user_id: UUID,
        title: str | None = None,
        visibility: str = "private",
        document_scope: str = "workspace",
        document_ids: list[UUID] | None = None,
    ) -> ChatSession:
        from ragnexus.infrastructure.database.models.document import Document
        from ragnexus.infrastructure.database.models.conversation import SessionDocument
        from ragnexus.infrastructure.database.models.workspace import WorkspaceMember

        membership = db.get(WorkspaceMember, (workspace_id, user_id))
        if membership is None:
            raise ConversationNotFound("Workspace not found")
        if visibility not in {"private", "workspace"}:
            raise ValueError("visibility must be 'private' or 'workspace'")
        if document_scope not in {"workspace", "selected"}:
            raise ValueError("document_scope must be 'workspace' or 'selected'")
        selected_ids = document_ids or []
        if document_scope == "selected" and not selected_ids:
            raise ValueError("selected document scope requires at least one document")
        if document_scope == "workspace" and selected_ids:
            raise ValueError("document_ids are only valid for selected document scope")

        if selected_ids:
            found = set(db.scalars(select(Document.id).where(
                Document.workspace_id == workspace_id,
                Document.id.in_(selected_ids),
                Document.status == "indexed",
            )).all())
            if found != set(selected_ids):
                raise ValueError("Every selected document must be indexed and belong to this workspace")

        session = ChatSession(
            workspace_id=workspace_id,
            created_by_user_id=user_id,
            title=title,
            visibility=visibility,
            document_scope=document_scope,
        )
        try:
            db.add(session)
            db.flush()
            for document_id in selected_ids:
                db.add(SessionDocument(
                    session_id=session.id,
                    document_id=document_id,
                    workspace_id=workspace_id,
                    added_by_user_id=user_id,
                ))
            db.commit()
            db.refresh(session)
            return session
        except Exception:
            db.rollback()
            raise

    @staticmethod
    def get_recent_messages(db: Session, *, session_id: UUID, limit: int = 12) -> list[Message]:
        rows = db.scalars(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at.desc(), Message.id.desc())
            .limit(limit)
        ).all()
        return list(reversed(rows))

    @staticmethod
    def append_message(
        db: Session,
        *,
        session_id: UUID,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> Message:
        if role not in {"user", "assistant"}:
            raise ValueError("Only user and assistant messages can be appended by chat orchestration")
        session = db.get(ChatSession, session_id)
        if session is None:
            raise ConversationNotFound("Conversation not found")
        if role == "user" and not session.title:
            session.title = content.strip().replace("\n", " ")[:72] or "New conversation"
        session.updated_at = datetime.now(timezone.utc)
        message = Message(
            session_id=session_id,
            role=role,
            content=content,
            message_metadata=metadata or {},
        )
        try:
            db.add(message)
            db.commit()
            db.refresh(message)
            return message
        except Exception:
            db.rollback()
            raise

    @staticmethod
    def list_sessions(db: Session, *, user_id: UUID, workspace_id: UUID,
                      search: str | None = None, limit: int = 50, offset: int = 0) -> list[ChatSession]:
        from ragnexus.infrastructure.database.models.workspace import WorkspaceMember
        stmt = (select(ChatSession)
                .join(WorkspaceMember, WorkspaceMember.workspace_id == ChatSession.workspace_id)
                .where(WorkspaceMember.user_id == user_id, ChatSession.workspace_id == workspace_id)
                .where((ChatSession.visibility == "workspace") | (ChatSession.created_by_user_id == user_id)))
        if search and search.strip():
            term = f"%{search.strip()}%"
            matching_sessions = select(Message.session_id).where(Message.content.ilike(term))
            stmt = stmt.where(or_(ChatSession.title.ilike(term), ChatSession.id.in_(matching_sessions)))
        return list(db.scalars(stmt.order_by(ChatSession.updated_at.desc(), ChatSession.created_at.desc())
                               .offset(offset).limit(min(max(limit, 1), 100))).all())

    @staticmethod
    def rename_session(db: Session, *, session_id: UUID, user_id: UUID, title: str) -> ChatSession:
        session = ConversationRepository.get_authorized_session(db, session_id=session_id, user_id=user_id)
        if session.created_by_user_id != user_id:
            raise ConversationNotFound("Conversation not found")
        session.title = title.strip()[:255]
        db.commit(); db.refresh(session)
        return session

    @staticmethod
    def delete_session(db: Session, *, session_id: UUID, user_id: UUID) -> None:
        session = ConversationRepository.get_authorized_session(db, session_id=session_id, user_id=user_id)
        if session.created_by_user_id != user_id:
            raise ConversationNotFound("Conversation not found")
        db.delete(session)
        db.commit()
