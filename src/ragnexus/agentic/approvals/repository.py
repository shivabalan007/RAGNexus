from __future__ import annotations

from datetime import datetime, timezone, timedelta
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ragnexus.agentic.approvals.gate import canonicalize_arguments
from ragnexus.infrastructure.database.models.approval import AgentApprovalRequest
from ragnexus.infrastructure.database.models.conversation import Session as ChatSession
from ragnexus.infrastructure.database.models.workspace import WorkspaceMember

ALLOWED_SENSITIVE_ACTIONS = {
    "documents.delete", "workspace.member_add", "workspace.member_role_change",
    "workspace.settings_update", "external.write",
}


def create_approval(db: Session, *, workspace_id: UUID, user_id: UUID,
                    action_name: str, arguments: dict, rationale: str | None,
                    session_id: UUID | None = None) -> AgentApprovalRequest:
    member = db.get(WorkspaceMember, (workspace_id, user_id))
    if member is None:
        raise HTTPException(status_code=404, detail="Workspace not found")
    if action_name not in ALLOWED_SENSITIVE_ACTIONS:
        raise HTTPException(status_code=422, detail="Unsupported sensitive action")
    try:
        encoded_arguments = canonicalize_arguments(arguments)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Approval arguments must be valid JSON") from exc
    if len(encoded_arguments) > 12000:
        raise HTTPException(status_code=422, detail="Approval arguments are too large")
    if session_id is not None:
        session = db.get(ChatSession, session_id)
        if session is None or session.workspace_id != workspace_id:
            raise HTTPException(status_code=404, detail="Conversation not found")
        if session.visibility == "private" and session.created_by_user_id != user_id:
            raise HTTPException(status_code=404, detail="Conversation not found")
    row = AgentApprovalRequest(
        workspace_id=workspace_id, session_id=session_id, requested_by_user_id=user_id,
        action_name=action_name, arguments=arguments or {}, rationale=rationale,
        status="pending",
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def list_approvals(db: Session, *, workspace_id: UUID, status: str | None = "pending") -> list[AgentApprovalRequest]:
    stmt = select(AgentApprovalRequest).where(AgentApprovalRequest.workspace_id == workspace_id)
    if status:
        stmt = stmt.where(AgentApprovalRequest.status == status)
    return list(db.scalars(stmt.order_by(AgentApprovalRequest.created_at.desc()).limit(100)).all())


def decide_approval(db: Session, *, workspace_id: UUID, approval_id: UUID,
                    actor_id: UUID, decision: str, reason: str | None) -> AgentApprovalRequest:
    membership = db.get(WorkspaceMember, (workspace_id, actor_id))
    if membership is None:
        raise HTTPException(status_code=404, detail="Approval request not found")
    if membership.role not in {"owner", "admin"}:
        raise HTTPException(status_code=403, detail="Only workspace owners or admins can approve sensitive actions")
    row = db.scalar(select(AgentApprovalRequest).where(
        AgentApprovalRequest.id == approval_id,
        AgentApprovalRequest.workspace_id == workspace_id,
    ).with_for_update())
    if row is None:
        raise HTTPException(status_code=404, detail="Approval request not found")
    if row.status != "pending":
        raise HTTPException(status_code=409, detail="Approval request has already been decided")
    if row.requested_by_user_id == actor_id:
        raise HTTPException(status_code=403, detail="A user cannot approve their own sensitive action request")
    row.status = decision
    row.decided_by_user_id = actor_id
    row.decision_reason = reason
    row.decided_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(row)
    return row


def assert_approved(db: Session, *, approval_id: UUID, workspace_id: UUID,
                    action_name: str, requester_user_id: UUID,
                    arguments: dict) -> AgentApprovalRequest:
    """Consume a single-use approval immediately before a tool side effect.

    The caller must still perform the normal permission check for the requested
    operation. An approval is not a substitute for authorization.
    """
    return consume_approved_request(
        db, approval_id=approval_id, workspace_id=workspace_id,
        action_name=action_name, requester_user_id=requester_user_id,
        arguments=arguments,
    )


def consume_approved_request(db: Session, *, approval_id: UUID, workspace_id: UUID,
                             action_name: str, requester_user_id: UUID,
                             arguments: dict) -> AgentApprovalRequest:
    """Atomically consume a one-time approval immediately before tool execution.

    Call this after the tool has independently authorized the acting user and
    immediately before performing the sensitive side effect.
    """
    row = db.scalar(select(AgentApprovalRequest).where(
        AgentApprovalRequest.id == approval_id,
        AgentApprovalRequest.workspace_id == workspace_id,
        AgentApprovalRequest.action_name == action_name,
    ).with_for_update())
    if row is None or row.status != "approved":
        raise PermissionError("A matching approved request is required")
    if row.consumed_at is not None:
        raise PermissionError("This approval request has already been consumed")
    if row.requested_by_user_id != requester_user_id:
        raise PermissionError("This approval request belongs to a different user")
    try:
        approved_arguments = canonicalize_arguments(row.arguments or {})
        requested_arguments = canonicalize_arguments(arguments or {})
    except (TypeError, ValueError) as exc:
        raise PermissionError("Action arguments are not valid JSON") from exc
    if approved_arguments != requested_arguments:
        raise PermissionError("Action arguments do not match the approved request")
    if row.decided_at is None or row.decided_at < datetime.now(timezone.utc) - timedelta(hours=24):
        raise PermissionError("This approval request has expired")
    requester = db.get(WorkspaceMember, (workspace_id, requester_user_id))
    if requester is None:
        raise PermissionError("The requesting user is no longer a workspace member")
    # WorkspaceMember uses a composite key; load with the complete identity.
    approver = None
    if row.decided_by_user_id is not None:
        approver = db.get(WorkspaceMember, (workspace_id, row.decided_by_user_id))
    if approver is None or approver.role not in {"owner", "admin"}:
        raise PermissionError("Approval is no longer valid because the approver lacks the required role")
    row.consumed_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(row)
    return row
