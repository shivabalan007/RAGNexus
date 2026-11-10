from fastapi import FastAPI

from ragnexus.api.v1.router import api_router
from ragnexus.api.exception_handlers import register_exception_handlers
from ragnexus.config import settings
from ragnexus.lifespan import lifespan


app = FastAPI(
    title=settings.APP_NAME,
    description="Production-oriented Retrieval Augmented Generation API",
    version=settings.APP_VERSION,
    lifespan=lifespan,
    debug=settings.DEBUG,
)

app.include_router(
    api_router,
    prefix=settings.API_V1_PREFIX,
)

register_exception_handlers(app)


@app.get("/")
async def root():
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
    }