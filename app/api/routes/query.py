"""Query endpoints - hybrid retrieval + re-ranking + grounded answers.

Every query endpoint now requires a valid JWT and threads the caller's
:class:`AuthUser` through to :class:`QueryService` so RBAC filtering can
drop chunks the user is not cleared to read *before* fusion / re-ranking.
"""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends

from app.api.deps import get_container, get_query_service
from app.api.models import QueryRequest, QueryResponse, SourceInfo
from app.auth.dependencies import get_current_user
from app.auth.models import AuthUser
from app.services.container import Container
from app.services.query_service import QueryService

router = APIRouter(tags=["query"])


@router.get("/sources", response_model=list[SourceInfo], summary="List indexed document sources")
def sources(container: Container = Depends(get_container)) -> list[SourceInfo]:
    return [SourceInfo(**item) for item in container.list_sources()]


@router.post("/query", response_model=QueryResponse, summary="Retrieve and answer")
async def query(
    req: QueryRequest,
    user: AuthUser = Depends(get_current_user),
    service: QueryService = Depends(get_query_service),
    container: Container = Depends(get_container),
) -> QueryResponse:
    researcher = None
    if req.provider or req.model:
        researcher = container.researcher_for(provider=req.provider, model=req.model)
    result = await service.answer(
        query=req.query,
        top_k=req.top_k,
        rerank=req.rerank,
        generate=req.generate,
        researcher=researcher,
        images=req.images,
        provider=req.provider,
        model=req.model,
        source=req.source,
        user=user,
    )
    return QueryResponse(**asdict(result))


@router.post("/search", response_model=QueryResponse, summary="Retrieve documents only")
def search(
    req: QueryRequest,
    user: AuthUser = Depends(get_current_user),
    service: QueryService = Depends(get_query_service),
) -> QueryResponse:
    result = service.search(
        query=req.query, top_k=req.top_k, rerank=req.rerank, source=req.source, user=user
    )
    return QueryResponse(**asdict(result))
