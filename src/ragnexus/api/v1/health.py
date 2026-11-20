from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import text

from ragnexus.config import get_settings

router = APIRouter(tags=["Health"])


@router.get("/health")
def health_check(request: Request):
    """Readiness check: report healthy only while the configured database is reachable."""
    engine = getattr(request.app.state, "db_engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Database connection is not initialized")
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database health check failed: {type(exc).__name__}") from exc
    dependencies = {"postgresql": "healthy"}
    settings = get_settings()
    if settings.INGESTION_QUEUE_BACKEND == "celery":
        broker_url = settings.CELERY_BROKER_URL or settings.REDIS_URL
        if not broker_url:
            raise HTTPException(status_code=503, detail="Celery is selected but no Redis broker URL is configured")
        try:
            from redis import Redis
            Redis.from_url(
                broker_url,
                socket_connect_timeout=1,
                socket_timeout=1,
            ).ping()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Redis health check failed: {type(exc).__name__}") from exc
        dependencies["redis"] = "healthy"
    return {
        "status": "healthy",
        "service": "RAGNexus",
        "dependencies": dependencies,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
