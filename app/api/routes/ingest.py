"""Ingestion endpoints.

Ingestion now requires a valid JWT and tags every chunk with ACL metadata
(department, clearance_level, owner) so RBAC filtering can enforce access
at retrieval time.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile

from app.api.deps import get_container, get_ingest_service
from app.api.models import IngestRequest, IngestResponse
from app.auth.dependencies import get_current_user
from app.auth.models import AuthUser
from app.core.errors import IngestionError, InvalidInputError
from app.core.logging import get_logger
from app.services.container import Container
from app.services.ingest_service import IngestService

logger = get_logger(__name__)

router = APIRouter(tags=["ingest"])

EXTENSION_TO_FORMAT = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".txt": "txt",
    ".md": "txt",
    ".markdown": "txt",
    ".csv": "txt",
    ".json": "txt",
    ".xlsx": "xlsx",
    ".xls": "xlsx",
}

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB


@router.post("/ingest", response_model=IngestResponse, summary="Ingest a document")
def ingest(
    req: IngestRequest,
    background_tasks: BackgroundTasks,
    user: AuthUser = Depends(get_current_user),
    container: Container = Depends(get_container),
    service: IngestService = Depends(get_ingest_service),
) -> IngestResponse:
    acl_defaults: dict = {}
    if req.department:
        acl_defaults["department"] = req.department
    if req.clearance_level:
        acl_defaults["clearance_level"] = req.clearance_level
    if req.owner:
        acl_defaults["owner"] = req.owner

    if req.background:
        task = service.ingest_background(
            path=req.path,
            fmt=req.format,
            chunk_size=req.chunk_size,
            collection_name=req.collection_name,
            rebuild_bm25=req.rebuild_bm25,
            source_name=Path(req.path).name,
            **acl_defaults,
        )

        def _run_and_refresh() -> None:
            task()
            container.refresh_index()

        background_tasks.add_task(_run_and_refresh)
        return IngestResponse(status="started", background=True)

    try:
        metrics = service.ingest(
            path=req.path,
            fmt=req.format,
            chunk_size=req.chunk_size,
            collection_name=req.collection_name,
            rebuild_bm25=req.rebuild_bm25,
            source_name=Path(req.path).name,
            **acl_defaults,
        )
        container.refresh_index()
    except (FileNotFoundError, ValueError) as e:
        raise IngestionError(str(e)) from e
    return IngestResponse(status="ok", background=False, metrics=metrics)


@router.get("/ingest/last", summary="Last ingestion result")
def last_ingest(service: IngestService = Depends(get_ingest_service)) -> dict:
    return service.last_run


@router.post(
    "/upload",
    response_model=IngestResponse,
    summary="Upload and ingest a document",
    description="Multipart file upload (.pdf, .docx, .txt, .md, .csv, .json). "
    "The file is parsed, chunked, embedded and indexed, then queryable via /api/v1/query.",
)
async def upload(
    file: UploadFile = File(..., description="Document to index"),
    chunk_size: str = Form("512T", description="One of: 256T, 512T, 1024T"),
    rebuild_bm25: bool = Form(False, description="Force BM25 index rebuild"),
    department: str | None = Form(None, description="Owning department (ACL metadata)"),
    clearance_level: str | None = Form(None, description="public|internal|confidential|secret"),
    owner: str | None = Form(None, description="Document owner (ACL metadata)"),
    user: AuthUser = Depends(get_current_user),
    container: Container = Depends(get_container),
    service: IngestService = Depends(get_ingest_service),
) -> IngestResponse:
    settings = container.settings
    ext = Path(file.filename or "").suffix.lower()
    fmt = EXTENSION_TO_FORMAT.get(ext)
    if fmt is None:
        raise InvalidInputError(
            f"Unsupported file type '{ext or 'unknown'}'. Allowed: "
            f"{', '.join(sorted(EXTENSION_TO_FORMAT))}"
        )

    body = await file.read()
    if not body:
        raise InvalidInputError(f"Uploaded file '{file.filename}' is empty.")
    if len(body) > MAX_UPLOAD_BYTES:
        raise InvalidInputError(f"File too large: {len(body)} bytes (limit {MAX_UPLOAD_BYTES}).")

    uploads_dir = Path(settings.uploads_dir)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    dest = uploads_dir / f"{uuid.uuid4().hex}{ext}"

    try:
        dest.write_bytes(body)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Could not persist upload: {e}") from e

    logger.info("Upload accepted | file={} | fmt={} | size={}B", file.filename, fmt, len(body))
    try:
        metrics = service.ingest(
            path=str(dest),
            fmt=fmt,
            chunk_size=chunk_size,
            collection_name=settings.chroma_collection,
            rebuild_bm25=rebuild_bm25,
            source_name=file.filename,
            department=department,
            clearance_level=clearance_level,
            owner=owner,
        )
        container.refresh_index()
    except (FileNotFoundError, ValueError, InvalidInputError) as e:
        raise IngestionError(str(e)) from e
    finally:
        # Cleanup the staged file once parsed (content now lives in the index).
        dest.unlink(missing_ok=True)

    metrics["filename"] = file.filename
    return IngestResponse(status="ok", background=False, metrics=metrics)
