"""Shared FastAPI dependencies."""

from __future__ import annotations

from fastapi import Depends, Request

from app.services.container import Container
from app.services.ingest_service import IngestService
from app.services.query_service import QueryService


def get_container(request: Request) -> Container:
    return request.app.state.container


def get_query_service(container: Container = Depends(get_container)) -> QueryService:
    return container.query_service()


def get_ingest_service(container: Container = Depends(get_container)) -> IngestService:
    return container.ingest_service()
