from __future__ import annotations

import hashlib
from datetime import datetime, timezone, timedelta
from pathlib import Path
from uuid import UUID

from sqlalchemy import or_, select, update
from sqlalchemy.orm import Session

from ragnexus.config import get_settings
from ragnexus.infrastructure.database.models.document import Document, DocumentChunk
from ragnexus.infrastructure.database.models.ingestion import IngestionJob
from ragnexus.ingestion.loaders.registry import load_file
from ragnexus.ingestion.chunking.factory import build_chunker
from ragnexus.embeddings.embedder import get_embedder
from ragnexus.infrastructure.storage.factory import build_document_storage
from ragnexus.observability.tracing import start_span


def recover_stale_jobs(db: Session, *, lease_seconds: int | None = None, max_attempts: int | None = None) -> int:
    """Requeue jobs whose database lease expired; fail jobs after bounded attempts."""
    settings = get_settings()
    lease_seconds = lease_seconds or settings.INGESTION_LEASE_SECONDS
    max_attempts = max_attempts or settings.INGESTION_MAX_ATTEMPTS
    now = datetime.now(timezone.utc)
    stale_cutoff = now - timedelta(seconds=lease_seconds)
    stale = db.scalars(
        select(IngestionJob).where(
            IngestionJob.status == "running",
            or_(
                IngestionJob.lease_expires_at < now,
                (IngestionJob.lease_expires_at.is_(None)) & (IngestionJob.started_at < stale_cutoff),
            ),
        ).with_for_update(skip_locked=True)
    ).all()
    recovered = 0
    for job in stale:
        document = db.get(Document, job.document_id)
        job.worker_id = None
        job.lease_expires_at = None
        job.heartbeat_at = None
        if document is None or document.status == "deleted":
            job.status = "failed"
            job.error_code = "DOCUMENT_DELETED"
            job.error_message = "Document was deleted while the worker was unavailable."
            job.finished_at = now
        elif job.attempts >= max_attempts:
            job.status = "failed"
            job.error_code = "WORKER_LEASE_EXPIRED"
            job.error_message = "Worker lease expired too many times; manual retry is required."
            job.finished_at = now
            document.status = "failed"
        else:
            job.status = "pending"
            job.error_code = "WORKER_LEASE_EXPIRED"
            job.error_message = "Worker lease expired; job returned to queue."
            job.finished_at = None
            document.status = "pending"
            recovered += 1
    if stale:
        db.commit()
    return recovered


def claim_job(db: Session, *, worker_id: str = "worker", lease_seconds: int | None = None) -> IngestionJob | None:
    settings = get_settings()
    now = datetime.now(timezone.utc)
    lease_seconds = lease_seconds or settings.INGESTION_LEASE_SECONDS
    job = db.scalar(
        select(IngestionJob)
        .where(IngestionJob.status == "pending")
        .order_by(IngestionJob.created_at)
        .with_for_update(skip_locked=True)
        .limit(1)
    )
    if job is None:
        return None
    job.status = "running"
    job.attempts += 1
    job.started_at = now
    job.heartbeat_at = now
    job.lease_expires_at = now + timedelta(seconds=lease_seconds)
    job.worker_id = worker_id
    job.error_code = None
    job.error_message = None
    document = db.get(Document, job.document_id)
    if document and document.status != "deleted":
        document.status = "processing"
    db.commit()
    db.refresh(job)
    return job


def heartbeat_job(db: Session, *, job_id: UUID, worker_id: str, lease_seconds: int | None = None) -> bool:
    settings = get_settings()
    now = datetime.now(timezone.utc)
    lease_seconds = lease_seconds or settings.INGESTION_LEASE_SECONDS
    result = db.execute(
        update(IngestionJob)
        .where(IngestionJob.id == job_id, IngestionJob.status == "running", IngestionJob.worker_id == worker_id)
        .values(heartbeat_at=now, lease_expires_at=now + timedelta(seconds=lease_seconds))
    )
    db.commit()
    return bool(result.rowcount)


def process_job(db: Session, job_id, *, worker_id: str | None = None) -> None:
    with start_span("ragnexus.ingestion.process_job", {
        "ragnexus.ingestion.job_id": str(job_id),
        "ragnexus.ingestion.worker_id": worker_id or "manual",
    }):
        _process_job(db, job_id, worker_id=worker_id)


def _process_job(db: Session, job_id, *, worker_id: str | None = None) -> None:
    settings = get_settings()
    storage = build_document_storage(settings)
    job = db.get(IngestionJob, job_id)
    if job is None or job.status != "running":
        return
    if worker_id is not None and job.worker_id != worker_id:
        return
    document = db.get(Document, job.document_id)
    if document is None:
        job.status = "failed"
        job.error_code = "DOCUMENT_MISSING"
        job.error_message = "Document record no longer exists."
        job.finished_at = datetime.now(timezone.utc)
        db.commit()
        return
    try:
        path = storage.resolve(document.document_metadata["storage_path"])
        try:
            loaded = load_file(path, original_filename=document.filename)
        finally:
            if settings.STORAGE_BACKEND == "s3":
                path.unlink(missing_ok=True)
        embedder = get_embedder()
        chunker = build_chunker(document.chunking_strategy or settings.CHUNKING_STRATEGY, embedder=embedder)
        chunks = chunker.split(loaded)
        if not chunks:
            raise ValueError("Document produced no chunks")
        texts = [chunk.text for chunk in chunks]
        vectors = embedder.embed_documents(texts)
        if len(vectors) != len(chunks):
            raise RuntimeError("Embedding count does not match chunk count")

        document = db.scalar(
            select(Document).where(Document.id == document.id).with_for_update().execution_options(populate_existing=True)
        )
        job = db.scalar(select(IngestionJob).where(IngestionJob.id == job_id).with_for_update().execution_options(populate_existing=True))
        if (document is None or document.status == "deleted" or job is None or job.status != "running"
                or (worker_id is not None and job.worker_id != worker_id)):
            db.rollback()
            return

        # Atomic/idempotent replacement of all chunks from any partial attempt.
        db.query(DocumentChunk).filter(DocumentChunk.document_id == document.id).delete(synchronize_session=False)
        for chunk, vector in zip(chunks, vectors):
            chunk_hash = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
            db.add(DocumentChunk(
                document_id=document.id,
                chunk_index=chunk.index,
                content=chunk.text,
                embedding=vector,
                chunk_metadata=chunk.metadata,
                content_hash=chunk_hash,
            ))
        document.status = "indexed"
        document.embedding_model = settings.EMBEDDING_MODEL
        document.embedding_dimension = settings.EMBEDDING_DIMENSION
        document.document_metadata = {**(document.document_metadata or {}), "chunk_count": len(chunks)}
        job.status = "completed"
        job.finished_at = datetime.now(timezone.utc)
        job.worker_id = None
        job.heartbeat_at = None
        job.lease_expires_at = None
        job.error_code = None
        job.error_message = None
        db.commit()
    except Exception as exc:
        db.rollback()
        job = db.get(IngestionJob, job_id)
        # A worker whose lease was reassigned must not overwrite the newer
        # owner's retry/completion state while handling its own late failure.
        if job is None or (worker_id is not None and job.worker_id != worker_id):
            db.rollback()
            raise
        document = db.get(Document, job.document_id)
        if job:
            retry = job.attempts < settings.INGESTION_MAX_ATTEMPTS and document is not None and document.status != "deleted"
            job.status = "pending" if retry else "failed"
            job.error_code = type(exc).__name__[:100]
            job.error_message = str(exc)[:4000]
            job.finished_at = None if retry else datetime.now(timezone.utc)
            job.worker_id = None
            job.heartbeat_at = None
            job.lease_expires_at = None
        if document and document.status != "deleted":
            document.status = "pending" if job and job.status == "pending" else "failed"
        db.commit()
        raise
