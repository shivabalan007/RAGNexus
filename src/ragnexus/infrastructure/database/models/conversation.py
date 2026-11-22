
# src/ragnexus/infrastructure/database/models/conversation.py

import uuid
from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from ragnexus.infrastructure.database.base import Base

from sqlalchemy.dialects.postgresql import JSONB


class Session(Base):
    __tablename__ = "sessions"

    __table_args__ = (
        CheckConstraint(
            "visibility IN ('private', 'workspace')",
            name="session_visibility",
        ),
        CheckConstraint(
            "document_scope IN ('workspace', 'selected')",
            name="session_document_scope",
        ),
        UniqueConstraint(
            "id",
            "workspace_id",
            name="uq_sessions_id_workspace",
        ),
        Index(
            "ix_sessions_workspace_creator",
            "workspace_id",
            "created_by_user_id",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4,
    )

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
    )

    created_by_user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
    )

    title: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    visibility: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="private",
        server_default="private",
    )

    document_scope: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="workspace",
        server_default="workspace",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class SessionDocument(Base):
    __tablename__ = "session_documents"

    __table_args__ = (
        ForeignKeyConstraint(
            ["session_id", "workspace_id"],
            ["sessions.id", "sessions.workspace_id"],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["document_id", "workspace_id"],
            ["documents.id", "documents.workspace_id"],
            ondelete="CASCADE",
        ),
        Index("ix_session_documents_document_id", "document_id"),
    )

    session_id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
    )

    document_id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
    )

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        nullable=False,
    )

    added_by_user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class Message(Base):
    __tablename__ = "messages"

    __table_args__ = (
        CheckConstraint(
            "role IN ('system', 'user', 'assistant', 'tool')",
            name="message_role",
        ),
        Index("ix_messages_session_created_at", "session_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4,
    )

    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )

    role: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    message_metadata: Mapped[dict] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
