"""Ingestion service - thin async-safe wrapper around the ingestion
pipeline so the API can schedule ingestion in background tasks."""

from __future__ import annotations

from collections.abc import Callable

from app.core.logging import get_logger
from ingestion.ingest import run_ingestion

logger = get_logger(__name__)


class IngestService:
    """Runs the ingestion pipeline and records metrics + failures."""

    def __init__(self) -> None:
        self._last_metrics: dict | None = None
        self._last_error: str | None = None

    @property
    def last_run(self) -> dict:
        return {
            "metrics": self._last_metrics,
            "error": self._last_error,
        }

    def ingest(
        self,
        path: str,
        fmt: str,
        chunk_size: str = "512T",
        collection_name: str = "enterprise_rag",
        rebuild_bm25: bool = False,
        source_name: str | None = None,
        department: str | None = None,
        clearance_level: str | None = None,
        owner: str | None = None,
    ) -> dict:
        self._last_error = None
        metrics = run_ingestion(
            path=path,
            fmt=fmt,
            chunk_size=chunk_size,
            collection_name=collection_name,
            rebuild_bm25=rebuild_bm25,
            source_name=source_name,
            department=department,
            clearance_level=clearance_level,
            owner=owner,
        )
        self._last_metrics = metrics
        return metrics

    def ingest_background(
        self,
        path: str,
        fmt: str,
        chunk_size: str = "512T",
        collection_name: str = "enterprise_rag",
        rebuild_bm25: bool = False,
        source_name: str | None = None,
        department: str | None = None,
        clearance_level: str | None = None,
        owner: str | None = None,
    ) -> Callable[[], None]:
        """Return a zero-arg callable for FastAPI ``BackgroundTasks``."""

        def _run() -> None:
            try:
                self.ingest(
                    path=path,
                    fmt=fmt,
                    chunk_size=chunk_size,
                    collection_name=collection_name,
                    rebuild_bm25=rebuild_bm25,
                    source_name=source_name,
                    department=department,
                    clearance_level=clearance_level,
                    owner=owner,
                )
                logger.info("Background ingestion finished | path={}", path)
            except Exception as e:  # noqa: BLE001 - background jobs must not crash the worker
                self._last_error = str(e)
                logger.error("Background ingestion failed | path={} | err={}", path, e)

        return _run
