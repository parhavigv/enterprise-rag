"""Service container - lazily constructs and memoises application services.

A single container instance is shared per application via the lifespan;
tests can swap any dependency through FastAPI's ``app.dependency_overrides``.
"""

from __future__ import annotations

from app.agents.llm_client import AsyncLLMClient
from app.agents.researcher import ResearcherAgent
from app.core.cache import TTLCache
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.services.ingest_service import IngestService
from app.services.query_service import QueryService
from ingestion.embedders.ollama_embedder import OllamaEmbedder
from retrieval.bm25 import BM25Indexer
from retrieval.dense import DenseRetriever
from retrieval.hybrid import HybridRetriever
from retrieval.reranker import CrossEncoderReranker
from retrieval.vector_store import ChromaAdapter

logger = get_logger(__name__)


class Container:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._cache: dict[str, object] = {}

    @property
    def settings(self) -> Settings:
        return self._settings

    # ------------------------------------------------------------------ #
    # Infrastructure
    # ------------------------------------------------------------------ #
    def embedder(self) -> OllamaEmbedder:
        return self._memo("embedder", OllamaEmbedder.from_env, self._settings)

    def adapter(self) -> ChromaAdapter:
        return self._memo("adapter", ChromaAdapter.from_env, self._settings)

    def cache(self) -> TTLCache:
        return self._memo(
            "cache",
            lambda: TTLCache(
                ttl_seconds=self._settings.cache_ttl_seconds,
                max_entries=self._settings.cache_max_entries,
            ),
        )

    # ------------------------------------------------------------------ #
    # Retrieval stack
    # ------------------------------------------------------------------ #
    def bm25_indexer(self) -> BM25Indexer:
        def _build() -> BM25Indexer:
            idx_path = self._settings.bm25_index_path
            indexer = BM25Indexer(index_path=idx_path)
            try:
                indexer = BM25Indexer.load(index_path=idx_path)
            except FileNotFoundError:
                logger.warning("BM25 index not found at '{}' - run ingestion first.", idx_path)
            except Exception as e:  # noqa: BLE001 - corrupt/outdated index
                logger.warning("Could not load BM25 index: {}", e)
            return indexer

        return self._memo("bm25", _build)

    def dense_retriever(self) -> DenseRetriever:
        return self._memo(
            "dense",
            lambda: DenseRetriever(adapter=self.adapter(), embedder=self.embedder()),
        )

    def hybrid_retriever(self) -> HybridRetriever:
        return self._memo(
            "hybrid",
            lambda: HybridRetriever(
                dense=self.dense_retriever(),
                sparse=self.bm25_indexer(),
                rrf_k=self._settings.rrf_k,
            ),
        )

    def reranker(self) -> CrossEncoderReranker | None:
        if not self._settings.reranker_enabled:
            return None
        if "reranker" not in self._cache:
            try:
                self._cache["reranker"] = CrossEncoderReranker(
                    model_name=self._settings.reranker_model, lazy_load=True
                )
            except Exception as e:  # noqa: BLE001 - degrade gracefully without the model
                logger.error("Failed to initialise reranker: {}", e)
                self._cache["reranker"] = None
        return self._cache["reranker"]  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def llm_client(self) -> AsyncLLMClient:
        s = self._settings
        return self._memo(
            "llm",
            lambda: AsyncLLMClient(
                provider=s.llm_provider,
                model=s.llm_model,
                base_url=s.llm_base_url,
                api_key=s.llm_api_key,
                temperature=s.llm_temperature,
                max_tokens=s.llm_max_tokens,
                timeout=s.llm_timeout_seconds,
                connect_timeout=s.llm_request_timeout_seconds,
            ),
        )

    def researcher(self) -> ResearcherAgent:
        return self._memo(
            "researcher",
            lambda: ResearcherAgent(
                llm=self.llm_client(), max_context_docs=self._settings.final_top_k
            ),
        )

    # ------------------------------------------------------------------ #
    # Services
    # ------------------------------------------------------------------ #
    def query_service(self) -> QueryService:
        return self._memo(
            "query_service",
            lambda: QueryService(
                hybrid=self.hybrid_retriever(),
                researcher=self.researcher(),
                reranker=self.reranker(),
                hybrid_top_k=self._settings.hybrid_top_k,
                dense_top_k=self._settings.dense_top_k,
                sparse_top_k=self._settings.sparse_top_k,
                default_final_top_k=self._settings.final_top_k,
                cache=self.cache() if self._settings.cache_enabled else None,
            ),
        )

    def ingest_service(self) -> IngestService:
        return self._memo("ingest_service", IngestService)

    def refresh_index(self) -> None:
        """Reload indexes from disk after ingestion so new docs are queryable.

        The retriever stack (dense / hybrid / query service) is re-memoised;
        heavier components (embedder, adapter, cache, reranker, researcher,
        LLM client) are left untouched.
        """
        for key in ("dense", "hybrid", "bm25", "query_service"):
            self._cache.pop(key, None)
        logger.info("Index refreshed after ingestion.")

    def stats(self) -> dict:
        """Aggregate index statistics for observability endpoints."""
        try:
            bm25_count = self.bm25_indexer().count()
        except Exception:  # noqa: BLE001
            bm25_count = 0
        try:
            chroma_count = self.adapter().count()
        except Exception:  # noqa: BLE001
            chroma_count = 0
        return {
            "chroma_collection": self._settings.chroma_collection,
            "chroma_count": chroma_count,
            "bm25_index_path": self._settings.bm25_index_path,
            "bm25_count": bm25_count,
            "embed_model": self._settings.embed_model,
            "llm_model": self._settings.llm_model,
        }

    def is_ready(self) -> bool:
        """Readiness gate: the retriever must have a usable BM25 index."""
        return self.bm25_indexer().count() > 0

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _memo(self, key: str, factory, *args) -> object:
        if key not in self._cache:
            self._cache[key] = factory(*args) if args else factory()
        return self._cache[key]
