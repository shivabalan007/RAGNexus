from datetime import datetime, timezone, timedelta
from pathlib import Path
from sqlalchemy import select
from sqlalchemy.orm import Session
from ragnexus.config import get_settings
from ragnexus.infrastructure.database.models.document import Document, DocumentChunk
from ragnexus.infrastructure.database.models.ingestion import IngestionJob
from ragnexus.ingestion.loaders.registry import load_file
from ragnexus.ingestion.chunking.factory import build_chunker
from ragnexus.embeddings.embedder import get_embedder
from ragnexus.infrastructure.storage.factory import build_document_storage
import hashlib



def recover_stale_jobs(db: Session, *, stale_after_minutes: int | None = None, max_attempts: int | None = None) -> int:
    """Return abandoned worker leases to the queue, or fail exhausted jobs."""
    settings = get_settings()
    lease_seconds = settings.INGESTION_LEASE_SECONDS
    if stale_after_minutes is not None:
        lease_seconds = max(60, stale_after_minutes * 60)
    max_attempts = max_attempts or settings.INGESTION_MAX_ATTEMPTS
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=lease_seconds)
    candidates = db.execute(
        select(IngestionJob.id, IngestionJob.document_id).where(
            IngestionJob.status == "running",
            IngestionJob.started_at < cutoff,
        ).order_by(IngestionJob.created_at)
    ).all()
    recovered = 0
    for job_id, document_id in candidates:
        # Use the same lock order as document deletion: document, then job.
        document = db.scalar(
            select(Document).where(Document.id == document_id).with_for_update(skip_locked=True)
        )
        if document is None:
            continue
        job = db.scalar(
            select(IngestionJob).where(
                IngestionJob.id == job_id,
                IngestionJob.status == "running",
            ).with_for_update(skip_locked=True)
        )
        if job is None:
            continue
        now = datetime.now(timezone.utc)
        if document.status == "deleted":
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
            job.started_at = None
            job.error_code = "WORKER_REQUEUED"
            job.error_message = "Previous worker lease expired; job returned to queue."
            job.finished_at = None
            document.status = "pending"
            recovered += 1
    db.commit()
    return recovered


def claim_job(db: Session) -> IngestionJob | None:
    candidate = db.execute(
        select(IngestionJob.id, IngestionJob.document_id)
        .where(IngestionJob.status == "pending")
        .order_by(IngestionJob.created_at)
        .limit(1)
    ).first()
    if candidate is None:
        return None
    job_id, document_id = candidate

    # Match the deletion lock order (document -> job) to avoid deadlocks.
    document = db.scalar(
        select(Document).where(Document.id == document_id).with_for_update(skip_locked=True)
    )
    if document is None:
        return None
    job = db.scalar(
        select(IngestionJob).where(
            IngestionJob.id == job_id,
            IngestionJob.status == "pending",
        ).with_for_update(skip_locked=True)
    )
    if job is None:
        return None
    if document.status == "deleted":
        job.status = "failed"
        job.error_code = "DOCUMENT_DELETED"
        job.error_message = "Document is unavailable for ingestion."
        job.finished_at = datetime.now(timezone.utc)
        db.commit()
        return None

    job.status = "running"
    job.attempts += 1
    job.started_at = datetime.now(timezone.utc)
    job.finished_at = None
    job.error_code = None
    job.error_message = None
    document.status = "processing"
    db.commit()
    db.refresh(job)
    return job


def claim_job_by_id(db: Session, job_id) -> IngestionJob | None:
    """Atomically claim one specific pending job for a Celery delivery."""
    candidate = db.execute(
        select(IngestionJob.document_id).where(
            IngestionJob.id == job_id,
            IngestionJob.status == "pending",
        )
    ).first()
    if candidate is None:
        return None
    document_id = candidate[0]

    # Deletion also locks document first; keep the order consistent.
    document = db.scalar(
        select(Document).where(Document.id == document_id).with_for_update(skip_locked=True)
    )
    if document is None:
        return None
    job = db.scalar(
        select(IngestionJob).where(
            IngestionJob.id == job_id,
            IngestionJob.status == "pending",
        ).with_for_update(skip_locked=True)
    )
    if job is None:
        return None
    if document.status == "deleted":
        job.status = "failed"
        job.error_code = "DOCUMENT_DELETED"
        job.error_message = "Document is unavailable for ingestion."
        job.finished_at = datetime.now(timezone.utc)
        db.commit()
        return None

    job.status = "running"
    job.attempts += 1
    job.started_at = datetime.now(timezone.utc)
    job.finished_at = None
    job.error_code = None
    job.error_message = None
    document.status = "processing"
    db.commit()
    db.refresh(job)
    return job


def process_job(db: Session, job_id) -> None:
    settings = get_settings()
    storage = build_document_storage(settings)
    job = db.get(IngestionJob, job_id)
    if job is None:
        return
    if job.status != "running":
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
            # S3 resolve() materializes a local temporary copy. Always remove it,
            # including when parsing raises; local storage paths are permanent.
            if settings.STORAGE_BACKEND == "s3":
                path.unlink(missing_ok=True)
        embedder = get_embedder()
        chunker = build_chunker(document.chunking_strategy or "fixed", embedder=embedder)
        chunks = chunker.split(loaded)
        if not chunks:
            raise ValueError("Document produced no chunks")
        texts = [chunk.text for chunk in chunks]
        vectors = embedder.embed_documents(texts)
        if len(vectors) != len(chunks):
            raise RuntimeError("Embedding count does not match chunk count")

        # Re-check under a row lock: a concurrent delete must never be undone by a late worker.
        document = db.scalar(select(Document).where(Document.id == document.id).with_for_update().execution_options(populate_existing=True))
        if document is None or document.status == "deleted":
            db.rollback()
            return

        # Idempotency: remove any partial chunks from an earlier attempt.
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
        job.error_code = None
        job.error_message = None
        db.commit()
    except Exception as exc:
        db.rollback()
        job = db.get(IngestionJob, job_id)
        document = db.get(Document, job.document_id) if job else None
        if job:
            retry = job.attempts < settings.INGESTION_MAX_ATTEMPTS and document is not None and document.status != "deleted"
            job.status = "pending" if retry else "failed"
            job.error_code = type(exc).__name__[:100]
            job.error_message = str(exc)[:4000]
            job.finished_at = None if retry else datetime.now(timezone.utc)
        if document and document.status != "deleted":
            document.status = "pending" if job and job.status == "pending" else "failed"
        db.commit()
        raise
