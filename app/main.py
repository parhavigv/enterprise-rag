"""FastAPI application factory for the Enterprise RAG service."""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from importlib.metadata import version as pkg_version

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health, ingest, query, speech, ui
from app.api.routes import settings as settings_routes
from app.core.config import get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import get_logger, setup_logging
from app.services.container import Container

logger = get_logger(__name__)


def _app_version() -> str:
    try:
        return pkg_version("enterprise-rag")
    except Exception:  # noqa: BLE001
        return "0.1.0"


def create_app(settings=None, container: Container | None = None) -> FastAPI:
    """Build and configure the FastAPI application.

    Args:
        settings: optional ``Settings`` override (tests inject here).
        container: optional pre-built service container (tests inject here).
    """
    settings = settings or get_settings()
    setup_logging(level=settings.log_level, fmt=settings.log_format)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.container = container or Container(settings=settings)
        app.state.started_at = time.monotonic()
        logger.info(
            "{} starting | env={} | chroma={} | bm25={}",
            settings.app_name,
            settings.environment,
            settings.chroma_collection,
            settings.bm25_index_path,
        )
        yield
        logger.info("{} shutting down", settings.app_name)

    app = FastAPI(
        title="Enterprise RAG API",
        description=(
            "Hybrid retrieval (ChromaDB dense + BM25 sparse, RRF fusion), "
            "cross-encoder re-ranking, and grounded LLM answers."
        ),
        version=_app_version(),
        lifespan=lifespan,
        docs_url="/docs" if not settings.debug else "/docs",
        redoc_url=None,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Observability middleware
    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next):
        request_id = request.headers.get(settings.request_id_header) or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "{} {} -> {} | {:.1f}ms | rid={}",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            request_id,
        )
        return response

    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(health.api_router, prefix=settings.api_prefix)
    app.include_router(query.router, prefix=settings.api_prefix)
    app.include_router(ingest.router, prefix=settings.api_prefix)
    app.include_router(speech.router, prefix=settings.api_prefix)
    app.include_router(settings_routes.router, prefix=settings.api_prefix)
    app.include_router(ui.router)

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service": "enterprise-rag",
            "version": _app_version(),
            "ui": "/ui",
            "docs": "/docs",
            "health": "/health",
            "readiness": "/health/ready",
        }

    return app


app = create_app()
