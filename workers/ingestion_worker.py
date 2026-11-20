import argparse
import logging
import time

from sqlalchemy import select

from ragnexus.config import get_settings
from ragnexus.infrastructure.database.connection import create_database_engine, create_session_factory
from ragnexus.infrastructure.database.models.ingestion import IngestionJob
from ragnexus.application.ingestion.worker_service import (
    claim_job, process_job, recover_stale_jobs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ragnexus.ingestion.worker")


def run(poll_seconds: float = 2.0):
    settings = get_settings()
    engine = create_database_engine(settings)
    factory = create_session_factory(engine)
    log.info("Database-polling ingestion fallback started")
    last_recovery = 0.0
    try:
        while True:
            if time.monotonic() - last_recovery >= 60:
                with factory() as db:
                    recovered = recover_stale_jobs(db)
                if recovered:
                    log.warning("Re-queued %s stale ingestion job(s)", recovered)
                last_recovery = time.monotonic()

            with factory() as db:
                job = claim_job(db)
                job_id = job.id if job else None
            if job_id is None:
                time.sleep(max(poll_seconds, 0.2))
                continue
            try:
                with factory() as db:
                    process_job(db, job_id)
                log.info("Completed ingestion job %s", job_id)
            except Exception:
                log.exception("Ingestion job %s failed", job_id)
    except KeyboardInterrupt:
        log.info("Worker stopping")
    finally:
        engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    run(parser.parse_args().poll_seconds)
