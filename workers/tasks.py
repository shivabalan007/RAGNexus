"""Celery tasks for RAGNexus document ingestion."""
from __future__ import annotations

import logging
from uuid import UUID

from sqlalchemy import select

from ragnexus.config import get_settings
from ragnexus.infrastructure.database.connection import (
    create_database_engine,
    create_session_factory,
)
from ragnexus.infrastructure.database.models.ingestion import IngestionJob
from ragnexus.application.ingestion.worker_service import (
    claim_job_by_id,
    process_job,
    recover_stale_jobs,
)
from workers.celery_app import celery_app

log = logging.getLogger("ragnexus.ingestion.tasks")


@celery_app.task(
    bind=True,
    name="ragnexus.ingestion.process_document",
    acks_late=True,
    reject_on_worker_lost=True,
    max_retries=9,
)
def process_ingestion_job(self, job_id: str):
    """Claim and process one durable DB job; retries reuse the same job row."""
    settings = get_settings()
    engine = create_database_engine(settings)
    factory = create_session_factory(engine)
    try:
        with factory() as db:
            claimed = claim_job_by_id(db, UUID(job_id))
        if claimed is None:
            return {"job_id": job_id, "status": "skipped"}

        try:
            with factory() as db:
                process_job(db, UUID(job_id))
        except Exception as exc:
            with factory() as db:
                current = db.get(IngestionJob, UUID(job_id))
                should_retry = bool(current and current.status == "pending")
            if should_retry and self.request.retries < settings.INGESTION_MAX_ATTEMPTS - 1:
                raise self.retry(
                    exc=exc,
                    countdown=min(10 * (2 ** self.request.retries), 60),
                )
            raise

        return {"job_id": job_id, "status": "completed"}
    finally:
        engine.dispose()


@celery_app.task(name="ragnexus.ingestion.recover_stale_jobs")
def recover_stale_ingestion_jobs():
    """Recover expired DB leases and re-publish all pending jobs.

    This periodic reconciliation closes the gap between the database's durable
    job state and broker delivery after worker termination or broker outages.
    """
    settings = get_settings()
    engine = create_database_engine(settings)
    factory = create_session_factory(engine)
    try:
        with factory() as db:
            recovered = recover_stale_jobs(db)
            # Only jobs just recovered from an expired running lease need a
            # fresh broker delivery. New pending jobs are published by the API.
            pending_ids = list(db.scalars(
                select(IngestionJob.id)
                .where(
                    IngestionJob.status == "pending",
                    IngestionJob.error_code == "WORKER_REQUEUED",
                )
                .order_by(IngestionJob.created_at)
            ).all())
        published = 0
        for pending_id in pending_ids:
            process_ingestion_job.delay(str(pending_id))
            with factory() as db:
                pending = db.get(IngestionJob, pending_id)
                if pending and pending.status == "pending" and pending.error_code == "WORKER_REQUEUED":
                    pending.error_code = "WORKER_REQUEUED_PUBLISHED"
                    pending.error_message = "Recovered ingestion job re-published to Celery."
                    db.commit()
            published += 1
        if recovered or published:
            log.info(
                "Ingestion reconciliation recovered=%s published=%s",
                recovered, published,
            )
        return {"recovered": recovered, "published": published}
    finally:
        engine.dispose()


def enqueue_ingestion_job(job_id) -> None:
    """Publish a committed ingestion job to Celery."""
    process_ingestion_job.delay(str(job_id))
