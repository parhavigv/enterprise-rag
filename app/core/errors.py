"""Domain errors and FastAPI exception handlers.

A small, typed error hierarchy lets services raise intent-revealing
exceptions and the API layer translate them into consistent RFC-7807 style
problem responses without leaking stack traces to clients.
"""

from __future__ import annotations

import traceback

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.logging import get_logger

logger = get_logger(__name__)


class AppError(Exception):
    """Base class for all application errors."""

    status_code = 500
    code = "internal_error"
    message = "An unexpected error occurred."

    def __init__(self, message: str | None = None, *, hint: str | None = None) -> None:
        self.message = message or self.message
        self.hint = hint
        super().__init__(self.message)


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"
    message = "Resource not found."


class InvalidInputError(AppError):
    status_code = 422
    code = "invalid_input"
    message = "The request payload is invalid."


class UpstreamUnavailableError(AppError):
    status_code = 503
    code = "upstream_unavailable"
    message = "A required dependency (Ollama / LLM / vector store) is unavailable."


class IngestionError(AppError):
    status_code = 422
    code = "ingestion_failed"
    message = "Ingestion failed."


class RetrieverNotReadyError(AppError):
    status_code = 503
    code = "retriever_not_ready"
    message = "The index has not been built. Run ingestion first."


def _build_body(exc: AppError) -> dict:
    body = {
        "error": {
            "code": exc.code,
            "message": exc.message,
            "status": exc.status_code,
        }
    }
    if exc.hint:
        body["error"]["hint"] = exc.hint
    return body


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        if exc.status_code >= 500:
            logger.error("App error | code={} | msg={}", exc.code, exc.message)
        else:
            logger.warning("App error | code={} | msg={}", exc.code, exc.message)
        return JSONResponse(status_code=exc.status_code, content=_build_body(exc))

    @app.exception_handler(Exception)
    async def _unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled exception | path={} | exc={}", request.url.path, exc)
        logger.opt(exception=exc).error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "An unexpected error occurred.",
                    "status": 500,
                }
            },
        )
