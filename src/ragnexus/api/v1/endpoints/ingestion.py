from pathlib import Path
from uuid import UUID
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session
from ragnexus.config import get_settings
from ragnexus.dependencies import get_db, get_current_user
from ragnexus.infrastructure.database.models.user import User
from ragnexus.infrastructure.database.models.document import Document
from ragnexus.infrastructure.database.models.ingestion import IngestionJob
from ragnexus.infrastructure.database.models.workspace import WorkspaceMember
from ragnexus.application.ingestion.service import create_upload, WorkspaceAccessDenied
from ragnexus.infrastructure.storage.local_storage import LocalDocumentStorage
from ragnexus.schemas.ingestion import DocumentUploadResponse, IngestionJobResponse

router = APIRouter(tags=["Ingestion"])

@router.post("/workspaces/{workspace_id}/documents", response_model=DocumentUploadResponse, status_code=status.HTTP_202_ACCEPTED)
def upload_document(
    workspace_id: UUID,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    settings = get_settings()
    storage = LocalDocumentStorage(settings.DOCUMENT_STORAGE_DIR)
    try:
        document, job, duplicate = create_upload(
            db,
            workspace_id=workspace_id,
            user_id=current_user.id,
            filename=file.filename or "upload.bin",
            stream=file.file,
            storage=storage,
            max_bytes=settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024,
        )
    except WorkspaceAccessDenied:
        raise HTTPException(status_code=404, detail="Workspace not found")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        file.file.close()
    metadata = document.document_metadata or {}
    return DocumentUploadResponse(
        document_id=document.id,
        workspace_id=document.workspace_id,
        filename=document.filename,
        status=document.status,
        content_hash=document.content_hash,
        version_number=int(metadata.get("version_number", 1)),
        duplicate=duplicate,
        ingestion_job_id=job.id if job else None,
    )

@router.get("/ingestion/jobs/{job_id}", response_model=IngestionJobResponse)
def get_ingestion_job(
    job_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    job = db.get(IngestionJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Ingestion job not found")
    document = db.get(Document, job.document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    membership = db.get(WorkspaceMember, (document.workspace_id, current_user.id))
    if membership is None:
        raise HTTPException(status_code=404, detail="Ingestion job not found")
    return job

@router.get("/workspaces/{workspace_id}/documents")
def list_documents(
    workspace_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    membership = db.get(WorkspaceMember, (workspace_id, current_user.id))
    if membership is None:
        raise HTTPException(status_code=404, detail="Workspace not found")
    rows = db.scalars(select(Document).where(
        Document.workspace_id == workspace_id
    ).order_by(Document.created_at.desc())).all()
    return [{
        "id": row.id,
        "filename": row.filename,
        "status": row.status,
        "content_hash": row.content_hash,
        "version_number": (row.document_metadata or {}).get("version_number", 1),
        "is_current": (row.document_metadata or {}).get("is_current", True),
        "created_at": row.created_at,
    } for row in rows]
