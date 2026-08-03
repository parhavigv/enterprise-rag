"""Query endpoints - hybrid retrieval + re-ranking + grounded answers."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends

from app.api.deps import get_container, get_query_service
from app.api.models import QueryRequest, QueryResponse
from app.services.container import Container
from app.services.query_service import QueryService

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse, summary="Retrieve and answer")
async def query(
    req: QueryRequest,
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
    )
    return QueryResponse(**asdict(result))


@router.post("/search", response_model=QueryResponse, summary="Retrieve documents only")
def search(req: QueryRequest, service: QueryService = Depends(get_query_service)) -> QueryResponse:
    result = service.search(query=req.query, top_k=req.top_k, rerank=req.rerank, source=req.source)
    return QueryResponse(**asdict(result))
