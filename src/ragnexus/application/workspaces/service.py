
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from ragnexus.core.permissions import require_permission
from ragnexus.infrastructure.database.models.user import User
from ragnexus.infrastructure.database.models.workspace import (
    Workspace,
    WorkspaceMember,
)


class WorkspaceNotFound(Exception):
    pass


class MemberAlreadyExists(Exception):
    pass


class UserNotFound(Exception):
    pass


def get_membership(
    db: Session,
    workspace_id: UUID,
    user_id: UUID,
) -> WorkspaceMember:
    membership = db.get(WorkspaceMember, (workspace_id, user_id))

    if membership is None:
        raise WorkspaceNotFound

    return membership


def list_workspaces(
    db: Session,
    user_id: UUID,
) -> list[tuple[Workspace, str]]:
    rows = db.execute(
        select(Workspace, WorkspaceMember.role)
        .join(
            WorkspaceMember,
            WorkspaceMember.workspace_id == Workspace.id,
        )
        .where(WorkspaceMember.user_id == user_id)
        .order_by(Workspace.created_at)
    ).all()

    return [(workspace, role) for workspace, role in rows]


def create_team_workspace(
    db: Session,
    *,
    name: str,
    owner: User,
) -> Workspace:
    workspace = Workspace(
        name=name.strip(),
        workspace_type="team",
        created_by=owner.id,
    )

    try:
        db.add(workspace)
        db.flush()

        membership = WorkspaceMember(
            workspace_id=workspace.id,
            user_id=owner.id,
            role="owner",
        )

        db.add(membership)
        db.commit()
        db.refresh(workspace)

        return workspace

    except Exception:
        db.rollback()
        raise


def add_member(
    db: Session,
    *,
    workspace_id: UUID,
    actor_id: UUID,
    email: str,
    role: str,
) -> WorkspaceMember:
    actor_membership = get_membership(
        db,
        workspace_id,
        actor_id,
    )

    require_permission(actor_membership.role, "members:add")

    if actor_membership.role == "admin" and role == "admin":
        from fastapi import HTTPException

        raise HTTPException(
            status_code=403,
            detail="Admins cannot grant admin role.",
        )

    user = db.scalar(
        select(User).where(
            User.email == email.strip().lower()
        )
    )

    if user is None:
        raise UserNotFound

    existing = db.get(
        WorkspaceMember,
        (workspace_id, user.id),
    )

    if existing is not None:
        raise MemberAlreadyExists

    member = WorkspaceMember(
        workspace_id=workspace_id,
        user_id=user.id,
        role=role,
    )

    try:
        db.add(member)
        db.commit()
        db.refresh(member)

        return member

    except Exception:
        db.rollback()
        raise


def change_member_role(
    db: Session,
    *,
    workspace_id: UUID,
    actor_id: UUID,
    target_user_id: UUID,
    role: str,
) -> WorkspaceMember:
    actor = get_membership(
        db,
        workspace_id,
        actor_id,
    )

    require_permission(actor.role, "members:role_change")

    target = get_membership(
        db,
        workspace_id,
        target_user_id,
    )

    if target.role == "owner":
        from fastapi import HTTPException

        raise HTTPException(
            status_code=400,
            detail="Workspace owner role cannot be changed.",
        )

    target.role = role

    try:
        db.commit()
        db.refresh(target)
        return target

    except Exception:
        db.rollback()
        raise


def remove_member(
    db: Session,
    *,
    workspace_id: UUID,
    actor_id: UUID,
    target_user_id: UUID,
) -> None:
    actor = get_membership(
        db,
        workspace_id,
        actor_id,
    )

    require_permission(actor.role, "members:remove")

    target = get_membership(
        db,
        workspace_id,
        target_user_id,
    )

    if target.role == "owner":
        from fastapi import HTTPException

        raise HTTPException(
            status_code=400,
            detail="Workspace owner cannot be removed.",
        )

    try:
        db.delete(target)
        db.commit()

    except Exception:
        db.rollback()
        raise
