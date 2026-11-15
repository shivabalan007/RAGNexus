from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ragnexus.agentic.approvals.repository import create_approval, decide_approval, list_approvals
from ragnexus.dependencies import get_current_user, get_db
from ragnexus.infrastructure.database.models.user import User
from ragnexus.infrastructure.database.models.workspace import WorkspaceMember
from ragnexus.schemas.approval import ApprovalCreate, ApprovalDecision, ApprovalRead

router = APIRouter(prefix="/workspaces/{workspace_id}/agent-approvals", tags=["Agent approvals"])


@router.post("", response_model=ApprovalRead, status_code=201)
def post_approval(workspace_id: UUID, payload: ApprovalCreate,
                  db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return create_approval(
        db, workspace_id=workspace_id, user_id=user.id,
        action_name=payload.action_name, arguments=payload.arguments,
        rationale=payload.rationale, session_id=payload.session_id,
    )


@router.get("", response_model=list[ApprovalRead])
def get_approvals(workspace_id: UUID, status: str | None = "pending",
                  db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    membership = db.get(WorkspaceMember, (workspace_id, user.id))
    if membership is None:
        raise HTTPException(status_code=404, detail="Workspace not found")
    if membership.role not in {"owner", "admin"}:
        raise HTTPException(status_code=403, detail="Only owners or admins can review approval requests")
    if status not in {None, "pending", "approved", "rejected", "expired"}:
        raise HTTPException(status_code=422, detail="Invalid approval status")
    return list_approvals(db, workspace_id=workspace_id, status=status)


@router.post("/{approval_id}/decision", response_model=ApprovalRead)
def post_approval_decision(workspace_id: UUID, approval_id: UUID, payload: ApprovalDecision,
                           db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return decide_approval(
        db, workspace_id=workspace_id, approval_id=approval_id,
        actor_id=user.id, decision=payload.decision, reason=payload.reason,
    )
