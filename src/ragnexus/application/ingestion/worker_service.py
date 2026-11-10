from datetime import datetime, timezone
from pathlib import Path
from sqlalchemy import select
from sqlalchemy.orm import Session
from ragnexus.config import get_settings
from ragnexus.infrastructure.database.models.document import Document, DocumentChunk
from ragnexus.infrastructure.database.models.ingestion import IngestionJob
from ragnexus.ingestion.loaders.registry import load_file
from ragnexus.ingestion.chunking.factory import build_chunker
from ragnexus.embeddings.embedder import get_embedder
from ragnexus.infrastructure.storage.local_storage import LocalDocumentStorage
import hashlib


def claim_job(db: Session) -> IngestionJob | None:
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
    job.started_at = datetime.now(timezone.utc)
    job.error_code = None
    job.error_message = None
    document = db.get(Document, job.document_id)
    if document:
        document.status = "processing"
    db.commit()
    db.refresh(job)
    return job


def process_job(db: Session, job_id) -> None:
    settings = get_settings()
    storage = LocalDocumentStorage(settings.DOCUMENT_STORAGE_DIR)
    job = db.get(IngestionJob, job_id)
    if job is None:
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
        loaded = load_file(path, original_filename=document.filename)
        embedder = get_embedder()
        chunker = build_chunker(document.chunking_strategy or "fixed", embedder=embedder)
        chunks = chunker.split(loaded)
        if not chunks:
            raise ValueError("Document produced no chunks")
        texts = [chunk.text for chunk in chunks]
        vectors = embedder.embed_documents(texts)
        if len(vectors) != len(chunks):
            raise RuntimeError("Embedding count does not match chunk count")

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
            job.status = "failed"
            job.error_code = type(exc).__name__[:100]
            job.error_message = str(exc)[:4000]
            job.finished_at = datetime.now(timezone.utc)
        if document:
            document.status = "failed"
        db.commit()
        raise
