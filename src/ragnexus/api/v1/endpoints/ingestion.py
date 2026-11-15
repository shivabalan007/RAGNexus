from uuid import UUID
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select, update
from sqlalchemy.orm import Session
from ragnexus.config import get_settings
from ragnexus.dependencies import get_db, get_current_user
from ragnexus.infrastructure.database.models.user import User
from ragnexus.infrastructure.database.models.document import Document, DocumentChunk
from ragnexus.infrastructure.database.models.ingestion import IngestionJob
from ragnexus.infrastructure.database.models.workspace import WorkspaceMember
from ragnexus.application.ingestion.service import create_upload, WorkspaceAccessDenied
from ragnexus.infrastructure.storage.factory import build_document_storage
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
    storage = build_document_storage(settings)
    try:
        document, job, duplicate = create_upload(
            db,
            workspace_id=workspace_id,
            user_id=current_user.id,
            filename=file.filename or "upload.bin",
            stream=file.file,
            storage=storage,
            max_bytes=settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024,
            chunking_strategy=settings.CHUNKING_STRATEGY,
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
        Document.workspace_id == workspace_id, Document.status != "deleted"
    ).order_by(Document.created_at.desc())).all()
    return [{
        "id": row.id,
        "filename": row.filename,
        "status": row.status,
        "content_hash": row.content_hash,
        "version_number": (row.document_metadata or {}).get("version_number", 1),
        "is_current": (row.document_metadata or {}).get("is_current", True),
        "file_size_bytes": row.file_size,
        "mime_type": row.mime_type,
        "created_at": row.created_at,
    } for row in rows]


@router.delete("/workspaces/{workspace_id}/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    workspace_id: UUID, document_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Soft-delete a document, immediately remove its searchable chunks, then remove its stored blob."""
    membership = db.get(WorkspaceMember, (workspace_id, current_user.id))
    if membership is None:
        raise HTTPException(status_code=404, detail="Document not found")
    document = db.scalar(select(Document).where(
        Document.id == document_id, Document.workspace_id == workspace_id
    ).with_for_update())
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    allowed = (
        membership.role in {"owner", "admin"}
        or (membership.role == "editor" and document.uploaded_by == current_user.id)
    )
    if not allowed:
        raise HTTPException(status_code=403, detail="You do not have permission to delete this document")

    metadata = dict(document.document_metadata or {})
    storage_path = metadata.get("storage_path")
    if document.status != "deleted":
        document.status = "deleted"
        metadata["is_current"] = False
        metadata["deleted_at"] = datetime.now(timezone.utc).isoformat()
        metadata["storage_cleanup_pending"] = bool(storage_path)
        document.document_metadata = metadata
        db.execute(update(IngestionJob).where(
            IngestionJob.document_id == document.id,
            IngestionJob.status.in_(["pending", "running"]),
        ).values(status="failed", error_code="DOCUMENT_DELETED",
                 error_message="Document was deleted before ingestion completed."))
        db.query(DocumentChunk).filter(DocumentChunk.document_id == document.id).delete(synchronize_session=False)
        db.commit()

    # Idempotent retry: the tombstone stays excluded from retrieval while physical cleanup is retried.
    if storage_path:
        try:
            build_document_storage(get_settings()).delete(storage_path)
            metadata["storage_cleanup_pending"] = False
            metadata.pop("storage_cleanup_error", None)
            document.document_metadata = metadata
            db.commit()
        except Exception as exc:
            metadata["storage_cleanup_pending"] = True
            metadata["storage_cleanup_error"] = type(exc).__name__
            document.document_metadata = metadata
            db.commit()
            raise HTTPException(status_code=503, detail="Document is excluded from retrieval, but stored-file cleanup failed. Retry deletion to complete cleanup.")
    return None
