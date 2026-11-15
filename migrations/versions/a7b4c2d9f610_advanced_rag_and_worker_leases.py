"""advanced RAG metadata, worker leases, approvals and active-content uniqueness

Revision ID: a7b4c2d9f610
Revises: e4c52b3911cc
Create Date: 2026-11-13
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "a7b4c2d9f610"
down_revision = "e4c52b3911cc"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Allow a previously soft-deleted file to be uploaded again while still
    # preventing duplicate active content inside a workspace.
    op.drop_constraint("uq_documents_workspace_content_hash", "documents", schema="ragnexus", type_="unique")
    op.create_index(
        "uq_documents_workspace_content_hash_active", "documents",
        ["workspace_id", "content_hash"], unique=True, schema="ragnexus",
        postgresql_where=sa.text("status <> 'deleted'"),
    )

    op.add_column("ingestion_jobs", sa.Column("worker_id", sa.String(length=100), nullable=True), schema="ragnexus")
    op.add_column("ingestion_jobs", sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True), schema="ragnexus")
    op.add_column("ingestion_jobs", sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True), schema="ragnexus")
    op.create_index("ix_ingestion_jobs_lease_expires_at", "ingestion_jobs", ["lease_expires_at"], schema="ragnexus")

    op.create_table(
        "agent_approval_requests",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("session_id", sa.Uuid(), nullable=True),
        sa.Column("requested_by_user_id", sa.Uuid(), nullable=False),
        sa.Column("action_name", sa.String(length=120), nullable=False),
        sa.Column("arguments", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=20), server_default="pending", nullable=False),
        sa.Column("decided_by_user_id", sa.Uuid(), nullable=True),
        sa.Column("decision_reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("status IN ('pending', 'approved', 'rejected', 'expired')", name=op.f("ck_agent_approval_requests_agent_approval_status")),
        sa.ForeignKeyConstraint(["workspace_id"], ["ragnexus.workspaces.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["session_id"], ["ragnexus.sessions.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["requested_by_user_id"], ["ragnexus.users.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["decided_by_user_id"], ["ragnexus.users.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        schema="ragnexus",
    )
    op.add_column("agent_approval_requests", sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True), schema="ragnexus")
    op.create_index("ix_agent_approvals_workspace_status", "agent_approval_requests", ["workspace_id", "status", "created_at"], schema="ragnexus")


def downgrade() -> None:
    op.drop_index("ix_agent_approvals_workspace_status", table_name="agent_approval_requests", schema="ragnexus")
    op.drop_table("agent_approval_requests", schema="ragnexus")
    op.drop_index("ix_ingestion_jobs_lease_expires_at", table_name="ingestion_jobs", schema="ragnexus")
    op.drop_column("ingestion_jobs", "lease_expires_at", schema="ragnexus")
    op.drop_column("ingestion_jobs", "heartbeat_at", schema="ragnexus")
    op.drop_column("ingestion_jobs", "worker_id", schema="ragnexus")
    op.drop_index("uq_documents_workspace_content_hash_active", table_name="documents", schema="ragnexus")
    op.create_unique_constraint("uq_documents_workspace_content_hash", "documents", ["workspace_id", "content_hash"], schema="ragnexus")
