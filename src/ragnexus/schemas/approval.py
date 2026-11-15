from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field


class ApprovalCreate(BaseModel):
    session_id: UUID | None = None
    action_name: str = Field(min_length=1, max_length=120)
    arguments: dict = Field(default_factory=dict)
    rationale: str | None = Field(default=None, max_length=2000)


class ApprovalDecision(BaseModel):
    decision: Literal["approved", "rejected"]
    reason: str | None = Field(default=None, max_length=2000)


class ApprovalRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    workspace_id: UUID
    session_id: UUID | None
    requested_by_user_id: UUID
    action_name: str
    arguments: dict
    rationale: str | None
    status: str
    decided_by_user_id: UUID | None
    decision_reason: str | None
    created_at: datetime
    decided_at: datetime | None
    consumed_at: datetime | None
