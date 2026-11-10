from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, ConfigDict

class DocumentUploadResponse(BaseModel):
    document_id: UUID
    workspace_id: UUID
    filename: str
    status: str
    content_hash: str
    version_number: int
    duplicate: bool
    ingestion_job_id: UUID | None

class IngestionJobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    document_id: UUID
    status: str
    attempts: int
    error_code: str | None
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
