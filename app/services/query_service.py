"""Query orchestration: hybrid retrieval -> (re-rank) -> LLM answer.

Layers:
    HybridRetriever (dense + sparse / RRF)
        -> CrossEncoderReranker (optional)
        -> ResearcherAgent (LLM answer, extractive fallback)
        -> TTL response cache
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from app.agents.researcher import ResearcherAgent, ResearchResponse
from app.core.cache import TTLCache
from app.core.errors import RetrieverNotReadyError
from app.core.logging import get_logger
from retrieval.hybrid import HybridRetriever
from retrieval.reranker import CrossEncoderReranker
from retrieval.types import RetrievedDocument

logger = get_logger(__name__)


@dataclass(slots=True)
class QueryResult:
    query: str
    search: list[dict] = field(default_factory=list)
    answer: str | None = None
    model: str | None = None
    generated: bool = False
    latency_ms: float = 0.0
    error: str | None = None
    cache_hit: bool = False


class QueryService:
    def __init__(
        self,
        hybrid: HybridRetriever,
        researcher: ResearcherAgent,
        reranker: CrossEncoderReranker | None = None,
        *,
        hybrid_top_k: int = 20,
        dense_top_k: int = 50,
        sparse_top_k: int = 50,
        default_final_top_k: int = 5,
        cache: TTLCache | None = None,
    ) -> None:
        self._hybrid = hybrid
        self._researcher = researcher
        self._reranker = reranker
        self._hybrid_top_k = hybrid_top_k
        self._dense_top_k = dense_top_k
        self._sparse_top_k = sparse_top_k
        self._default_final_top_k = default_final_top_k
        self._cache = cache

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def is_ready(self) -> bool:
        """Cheap readiness check - the retriever must have an index."""
        return self._hybrid.is_ready()

    def search(self, query: str, top_k: int | None = None, rerank: bool = True) -> QueryResult:
        """Retrieve (and optionally re-rank) top-k documents."""
        t0 = time.perf_counter()
        top_k = top_k or self._default_final_top_k
        docs = self._retrieve(query, top_k, rerank)
        return QueryResult(
            query=query,
            search=[d.to_dict() for d in docs],
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    async def answer(
        self,
        query: str,
        top_k: int | None = None,
        rerank: bool = True,
        generate: bool = True,
        *,
        researcher: ResearcherAgent | None = None,
        images: list[str] | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> QueryResult:
        """Full RAG: retrieve, re-rank, and generate a grounded answer.

        ``researcher`` overrides the memoised agent (used for per-request
        provider/model overrides); ``images`` are attached to the LLM call as
        vision content blocks; ``provider`` / ``model`` only affect the cache
        key so switching models never reuses a stale cached answer.
        """
        t0 = time.perf_counter()
        top_k = top_k or self._default_final_top_k

        cache_key = self._cache_key(query, top_k, rerank, generate, provider, model, images)
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                cached["cache_hit"] = True
                logger.info("Cache hit for query | key={}", cache_key)
                return QueryResult(**cached)

        docs = self._retrieve(query, top_k, rerank)

        if generate:
            agent = researcher or self._researcher
            research: ResearchResponse = await agent.generate(query, docs, images=images)
            result = QueryResult(
                query=query,
                search=[d.to_dict() for d in docs],
                answer=research.answer,
                model=research.model,
                generated=research.generated,
                error=research.error,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        else:
            result = QueryResult(
                query=query,
                search=[d.to_dict() for d in docs],
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        if self._cache is not None and not result.error:
            self._cache.set(cache_key, asdict(result))
        return result

    def clear_cache(self) -> None:
        if self._cache is not None:
            self._cache.clear()

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #
    def _retrieve(self, query: str, top_k: int, rerank: bool) -> list[RetrievedDocument]:
        if not self.is_ready():
            raise RetrieverNotReadyError()
        docs = self._hybrid.retrieve(
            query,
            top_k=self._hybrid_top_k,
            dense_k=self._dense_top_k,
            sparse_k=self._sparse_top_k,
        )
        if rerank and self._reranker is not None and docs:
            docs = self._reranker.rerank_documents(query, docs, top_k=top_k)
        else:
            docs = docs[:top_k]
        return docs

    @staticmethod
    def _cache_key(
        query: str,
        top_k: int,
        rerank: bool,
        generate: bool,
        provider: str | None = None,
        model: str | None = None,
        images: list[str] | None = None,
    ) -> str:
        image_count = len(images) if images else 0
        model_scope = f"{provider}|{model}|img{image_count}"
        return f"{query.strip().lower()}|{top_k}|{rerank}|{generate}|{model_scope}"
