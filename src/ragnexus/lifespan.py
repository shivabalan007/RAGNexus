from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text

from ragnexus.config import settings
from ragnexus.infrastructure.database.connection import (
    create_database_engine,
    create_session_factory,
)
from ragnexus.observability.logging import (
    configure_logging,
    get_logger,
)

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    logger.info("Starting RAGNexus")

    engine = create_database_engine(settings)

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        app.state.db_engine = engine
        app.state.db_session_factory = create_session_factory(engine)

        logger.info("PostgreSQL connection established")

        yield

    except Exception:
        logger.exception("RAGNexus startup or runtime failure")
        raise

    finally:
        engine.dispose()
        logger.info("RAGNexus shutdown complete")