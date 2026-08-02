"""Health and readiness endpoints."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request

from app.api.deps import get_container
from app.api.models import HealthCheck
from app.core.logging import get_logger
from app.services.container import Container

logger = get_logger(__name__)

# Mounted at the app root (used by container orchestrators / load balancers).
router = APIRouter(tags=["health"])
# Mounted under the API prefix for versioned consumers.
api_router = APIRouter(tags=["health"])


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("enterprise-rag")
    except Exception:  # noqa: BLE001
        return "0.1.0"


@router.get("/health", response_model=HealthCheck, summary="Liveness probe")
def liveness(request: Request) -> HealthCheck:
    state = request.app.state
    return HealthCheck(
        status="ok",
        version=_version(),
        uptime_seconds=round(time.monotonic() - state.started_at, 1),
        checks={"app": "ok"},
    )


@router.get("/health/ready", response_model=HealthCheck, summary="Readiness probe")
def readiness(
    container: Container = Depends(get_container), request: Request = None
) -> HealthCheck:
    state = request.app.state

    checks: dict = {}
    try:
        checks["chroma"] = container.adapter().health()
    except Exception as e:  # noqa: BLE001
        checks["chroma"] = False
        logger.error("Readiness chroma check failed: {}", e)

    try:
        checks["bm25"] = container.bm25_indexer().count() > 0
    except Exception as e:  # noqa: BLE001
        checks["bm25"] = False
        logger.error("Readiness bm25 check failed: {}", e)

    try:
        checks["embedder"] = container.embedder().health()
    except Exception as e:  # noqa: BLE001
        checks["embedder"] = False
        logger.error("Readiness embedder check failed: {}", e)

    all_ok = all(checks.values())
    return HealthCheck(
        status="ok" if all_ok else "degraded",
        version=_version(),
        uptime_seconds=round(time.monotonic() - state.started_at, 1),
        checks=checks,
    )


@api_router.get("/stats", summary="Index statistics")
def stats(container: Container = Depends(get_container)) -> dict:
    return container.stats()
