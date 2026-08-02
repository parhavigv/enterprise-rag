"""Ingestion endpoints."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends

from app.api.deps import get_ingest_service
from app.api.models import IngestRequest, IngestResponse
from app.core.errors import IngestionError
from app.core.logging import get_logger
from app.services.ingest_service import IngestService

logger = get_logger(__name__)

router = APIRouter(tags=["ingest"])


@router.post("/ingest", response_model=IngestResponse, summary="Ingest a document")
def ingest(
    req: IngestRequest,
    background_tasks: BackgroundTasks,
    service: IngestService = Depends(get_ingest_service),
) -> IngestResponse:
    if req.background:
        background_tasks.add_task(
            service.ingest_background(
                path=req.path,
                fmt=req.format,
                chunk_size=req.chunk_size,
                collection_name=req.collection_name,
                rebuild_bm25=req.rebuild_bm25,
            )
        )
        return IngestResponse(status="started", background=True)

    try:
        metrics = service.ingest(
            path=req.path,
            fmt=req.format,
            chunk_size=req.chunk_size,
            collection_name=req.collection_name,
            rebuild_bm25=req.rebuild_bm25,
        )
    except (FileNotFoundError, ValueError) as e:
        raise IngestionError(str(e)) from e
    return IngestResponse(status="ok", background=False, metrics=metrics)


@router.get("/ingest/last", summary="Last ingestion result")
def last_ingest(service: IngestService = Depends(get_ingest_service)) -> dict:
    return service.last_run
