from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ragnexus.domain.exceptions import RAGNexusException
from ragnexus.observability.logging import get_logger

logger = get_logger(__name__)


async def ragnexus_exception_handler(
    request: Request,
    exc: RAGNexusException,
) -> JSONResponse:
    logger.warning(
        "Application error | code=%s | path=%s",
        exc.code,
        request.url.path,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
            }
        },
    )


async def unexpected_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    logger.exception(
        "Unhandled exception | path=%s",
        request.url.path,
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred.",
            }
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(
        RAGNexusException,
        ragnexus_exception_handler,
    )
    app.add_exception_handler(
        Exception,
        unexpected_exception_handler,
    )