from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    workspace_id: UUID
    title: str | None = Field(default=None, max_length=255)
    visibility: str = Field(default="private", pattern="^(private|workspace)$")
    document_scope: str = Field(default="workspace", pattern="^(workspace|selected)$")
    document_ids: list[UUID] = Field(default_factory=list)


class ChatRequest(BaseModel):
    session_id: UUID
    query: str = Field(min_length=1, max_length=12000)
    candidate_k: int = Field(default=20, ge=1, le=100)
    final_k: int = Field(default=5, ge=1, le=20)
    pipeline_mode: str = Field(default="standard", pattern="^(standard|agentic)$")
    enable_web_search: bool = False


class CitationResponse(BaseModel):
    source_number: int
    source_type: str = "document"
    document_id: str | None = None
    chunk_id: str | None = None
    chunk_index: int | None = None
    filename: str | None = None
    title: str | None = None
    url: str | None = None


class ChatResponseSchema(BaseModel):
    session_id: str
    answer: str
    citations: list[CitationResponse]
    rewritten_query: str
    retrieved_count: int
    abstained: bool
    latency_ms: float
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    retry_count: int = 0
    diagnostics: dict | None = None
