from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ragnexus.application.workspaces.service import (
    MemberAlreadyExists, UserNotFound, WorkspaceNotFound,
    add_member, change_member_role, create_team_workspace,
    get_membership, list_workspaces, remove_member,
)
from ragnexus.core.permissions import require_permission
from ragnexus.dependencies import get_current_user, get_db
from ragnexus.infrastructure.database.models.user import User
from ragnexus.infrastructure.database.models.workspace import Workspace, WorkspaceMember
from ragnexus.schemas.workspace import (
    WorkspaceCreate, WorkspaceMemberAdd, WorkspaceMemberRead,
    WorkspaceMemberRoleUpdate, WorkspaceRead,
)

router = APIRouter(prefix="/workspaces", tags=["Workspaces"])


@router.get("", response_model=list[WorkspaceRead])
def get_workspaces(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return [workspace for workspace, _role in list_workspaces(db, user.id)]


@router.post("", response_model=WorkspaceRead, status_code=status.HTTP_201_CREATED)
def post_workspace(payload: WorkspaceCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return create_team_workspace(db, name=payload.name, owner=user)


@router.get("/{workspace_id}", response_model=WorkspaceRead)
def get_workspace(workspace_id: UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    membership = get_membership(db, workspace_id, user.id)
    require_permission(membership.role, "workspace:read")
    workspace = db.get(Workspace, workspace_id)
    if workspace is None:
        raise HTTPException(status_code=404, detail="Workspace not found.")
    return workspace


@router.get("/{workspace_id}/members", response_model=list[WorkspaceMemberRead])
def get_members(workspace_id: UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    membership = get_membership(db, workspace_id, user.id)
    require_permission(membership.role, "members:read")
    rows = db.execute(
        select(WorkspaceMember.user_id, User.email, WorkspaceMember.role, WorkspaceMember.joined_at)
        .join(User, User.id == WorkspaceMember.user_id)
        .where(WorkspaceMember.workspace_id == workspace_id)
        .order_by(WorkspaceMember.joined_at)
    ).all()
    return [WorkspaceMemberRead(user_id=r.user_id, email=r.email, role=r.role, joined_at=r.joined_at) for r in rows]


@router.post("/{workspace_id}/members", response_model=WorkspaceMemberRead, status_code=201)
def post_member(workspace_id: UUID, payload: WorkspaceMemberAdd, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    try:
        member = add_member(db, workspace_id=workspace_id, actor_id=user.id, email=payload.email, role=payload.role)
    except WorkspaceNotFound:
        raise HTTPException(status_code=404, detail="Workspace not found.")
    except UserNotFound:
        raise HTTPException(status_code=404, detail="User account not found. Register the user before adding them.")
    except MemberAlreadyExists:
        raise HTTPException(status_code=409, detail="User is already a workspace member.")
    member_user = db.get(User, member.user_id)
    return WorkspaceMemberRead(user_id=member.user_id, email=member_user.email, role=member.role, joined_at=member.joined_at)


@router.patch("/{workspace_id}/members/{target_user_id}", response_model=WorkspaceMemberRead)
def patch_member(workspace_id: UUID, target_user_id: UUID, payload: WorkspaceMemberRoleUpdate, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    try:
        member = change_member_role(db, workspace_id=workspace_id, actor_id=user.id, target_user_id=target_user_id, role=payload.role)
    except WorkspaceNotFound:
        raise HTTPException(status_code=404, detail="Workspace or member not found.")
    member_user = db.get(User, member.user_id)
    return WorkspaceMemberRead(user_id=member.user_id, email=member_user.email, role=member.role, joined_at=member.joined_at)


@router.delete("/{workspace_id}/members/{target_user_id}", status_code=204)
def delete_member(workspace_id: UUID, target_user_id: UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    try:
        remove_member(db, workspace_id=workspace_id, actor_id=user.id, target_user_id=target_user_id)
    except WorkspaceNotFound:
        raise HTTPException(status_code=404, detail="Workspace or member not found.")
    return None
