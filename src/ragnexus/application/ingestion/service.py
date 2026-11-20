from pathlib import Path
from uuid import UUID
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from ragnexus.core.permissions import require_permission
from ragnexus.infrastructure.database.models.document import Document
from ragnexus.infrastructure.database.models.ingestion import IngestionJob
from ragnexus.infrastructure.database.models.workspace import Workspace, WorkspaceMember
from ragnexus.config import get_settings


ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv", ".docx", ".html", ".htm"}

class WorkspaceAccessDenied(Exception): pass
class DuplicateDocument(Exception): pass


def create_upload(db: Session, *, workspace_id: UUID, user_id: UUID, filename: str,
                  stream, storage, max_bytes: int) -> tuple[Document, IngestionJob, bool]:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}")
    # Serialize same-workspace version allocation. Two simultaneous uploads
    # of different content under the same filename must not both become current.
    membership = db.execute(
        select(WorkspaceMember)
        .where(WorkspaceMember.workspace_id == workspace_id, WorkspaceMember.user_id == user_id)
        .with_for_update()
    ).scalar_one_or_none()
    if membership is None:
        raise WorkspaceAccessDenied
    require_permission(membership.role, "documents:upload")
    # Lock the workspace row (not only the caller's membership row) so uploads
    # by different workspace members serialize version-number allocation.
    workspace = db.scalar(
        select(Workspace).where(Workspace.id == workspace_id).with_for_update()
    )
    if workspace is None:
        raise WorkspaceAccessDenied

    stored_path, content_hash, size = storage.save_stream(
        stream, workspace_id=str(workspace_id), max_bytes=max_bytes
    )
    try:
        existing = db.scalar(select(Document).where(
            Document.workspace_id == workspace_id,
            Document.content_hash == content_hash,
            Document.status != "deleted",
        ))
        if existing is not None:
            storage.delete(stored_path)
            return existing, db.scalar(select(IngestionJob).where(
                IngestionJob.document_id == existing.id
            ).order_by(IngestionJob.created_at.desc())), True

        logical_key = Path(filename).name.casefold()
        siblings = db.scalars(select(Document).where(
            Document.workspace_id == workspace_id
        )).all()
        same_name = [d for d in siblings if (d.document_metadata or {}).get("logical_document_key") == logical_key]
        version = max([int((d.document_metadata or {}).get("version_number", 0)) for d in same_name] or [0]) + 1
        for old in same_name:
            metadata = dict(old.document_metadata or {})
            metadata["is_current"] = False
            old.document_metadata = metadata

        document = Document(
            workspace_id=workspace_id,
            uploaded_by=user_id,
            filename=Path(filename).name[:255],
            content_hash=content_hash,
            mime_type=None,
            file_size=size,
            status="pending",
            parser_version="ingestion-v1",
            chunking_strategy=get_settings().CHUNKING_STRATEGY,
            embedding_model=None,
            embedding_dimension=None,
            document_metadata={
                "storage_path": str(stored_path),
                "logical_document_key": logical_key,
                "version_number": version,
                "is_current": True,
                "original_filename": Path(filename).name,
            },
        )
        db.add(document)
        db.flush()
        job = IngestionJob(document_id=document.id, status="pending", attempts=0)
        db.add(job)
        db.commit()
        db.refresh(document)
        db.refresh(job)
        return document, job, False
    except IntegrityError:
        # A concurrent request may have inserted the same active content after
        # our initial lookup. The partial unique index is the final authority.
        db.rollback()
        existing = db.scalar(select(Document).where(
            Document.workspace_id == workspace_id,
            Document.content_hash == content_hash,
            Document.status != "deleted",
        ))
        storage.delete(stored_path)
        if existing is not None:
            latest_job = db.scalar(select(IngestionJob).where(
                IngestionJob.document_id == existing.id
            ).order_by(IngestionJob.created_at.desc()))
            return existing, latest_job, True
        raise
    except Exception:
        db.rollback()
        storage.delete(stored_path)
        raise
