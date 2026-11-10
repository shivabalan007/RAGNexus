from enum import StrEnum
from fastapi import HTTPException, status

class WorkspaceRole(StrEnum):
    OWNER = "owner"
    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"

ROLE_PERMISSIONS: dict[WorkspaceRole, frozenset[str]] = {
    WorkspaceRole.OWNER: frozenset({
        "workspace:read", "workspace:update", "workspace:delete",
        "members:read", "members:add", "members:role_change", "members:remove",
        "documents:read", "documents:upload", "documents:delete_any",
        "sessions:read_workspace",
    }),
    WorkspaceRole.ADMIN: frozenset({
        "workspace:read", "workspace:update", "members:read", "members:add",
        "documents:read", "documents:upload", "documents:delete_any",
        "sessions:read_workspace",
    }),
    WorkspaceRole.EDITOR: frozenset({
        "workspace:read", "documents:read", "documents:upload",
        "documents:delete_own", "sessions:read_workspace",
    }),
    WorkspaceRole.VIEWER: frozenset({
        "workspace:read", "documents:read", "sessions:read_workspace",
    }),
}

def has_permission(role: str | WorkspaceRole, permission: str) -> bool:
    try:
        normalized = WorkspaceRole(role)
    except ValueError:
        return False
    return permission in ROLE_PERMISSIONS[normalized]

def require_permission(role: str | WorkspaceRole, permission: str) -> None:
    if not has_permission(role, permission):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to perform this action.",
        )
