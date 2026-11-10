from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class WorkspaceCreate(BaseModel):
    name: str = Field(min_length=2, max_length=150)


class WorkspaceRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    name: str
    workspace_type: str
    created_by: UUID
    created_at: datetime


class WorkspaceMemberAdd(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    role: str = Field(default="viewer", pattern="^(admin|editor|viewer)$")


class WorkspaceMemberRoleUpdate(BaseModel):
    role: str = Field(pattern="^(admin|editor|viewer)$")


class WorkspaceMemberRead(BaseModel):
    user_id: UUID
    email: str
    role: str
    joined_at: datetime
