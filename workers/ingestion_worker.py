import argparse
import logging
import time
from ragnexus.config import get_settings
from ragnexus.infrastructure.database.connection import create_database_engine, create_session_factory
from ragnexus.application.ingestion.worker_service import claim_job, process_job

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ragnexus.ingestion.worker")

def run(poll_seconds: float = 2.0):
    settings = get_settings()
    engine = create_database_engine(settings)
    factory = create_session_factory(engine)
    log.info("Ingestion worker started")
    try:
        while True:
            with factory() as db:
                job = claim_job(db)
                job_id = job.id if job else None
            if job_id is None:
                time.sleep(poll_seconds)
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
