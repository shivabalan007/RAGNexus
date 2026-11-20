"""Celery application for durable asynchronous document ingestion.

Start from the project root with:
  PYTHONPATH=src celery -A workers.celery_app:celery_app worker -B --loglevel=INFO
"""
from ragnexus.config import get_settings

settings = get_settings()
broker_url = settings.CELERY_BROKER_URL or settings.REDIS_URL
if not broker_url:
    raise RuntimeError(
        "Celery ingestion requires CELERY_BROKER_URL or REDIS_URL in .env"
    )

from celery import Celery

celery_app = Celery(
    "ragnexus",
    broker=broker_url,
    backend=settings.CELERY_RESULT_BACKEND or broker_url,
    include=["workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
    result_expires=3600,
    beat_schedule={
        "recover-stale-ragnexus-ingestion-jobs": {
            "task": "ragnexus.ingestion.recover_stale_jobs",
            "schedule": float(settings.INGESTION_RECOVERY_INTERVAL_SECONDS),
        }
    },
)
