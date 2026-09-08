"""Service container - lazily constructs and memoises application services.

A single container instance is shared per application via the lifespan;
tests can swap any dependency through FastAPI's ``app.dependency_overrides``.
"""

from __future__ import annotations

from app.agents.llm_client import AsyncLLMClient
from app.agents.researcher import ResearcherAgent
from app.auth.audit import AuditLogger
from app.auth.service import AuthService
from app.auth.users import UserStore, build_user_store, seed_demo_users
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
    # RBAC / Auth
    # ------------------------------------------------------------------ #
    def auth(self) -> AuthService:
        return self._memo(
            "auth",
            lambda: AuthService(
                secret=self._settings.auth_jwt_secret or None,
                algorithm=self._settings.auth_jwt_algorithm,
                expiry_seconds=self._settings.auth_jwt_expiry_seconds,
                issuer=self._settings.auth_issuer,
                audience=self._settings.auth_audience,
            ),
        )

    def user_store(self) -> UserStore:
        store = self._memo("user_store", build_user_store, self._settings)
        # Auto-provision demo identities on an empty store - development only.
        # Production deployments must provision accounts explicitly (see
        # scripts/manage_users.py) or wire an IdP behind the UserStore seam.
        if (
            self._settings.environment != "production"
            and getattr(self._settings, "auth_seed_demo_users", True)
            and callable(getattr(store, "create_user", None))
            and store.is_empty()
        ):
            seed_demo_users(store)
        return store

    def audit(self) -> AuditLogger:
        return self._memo("audit", lambda: AuditLogger(enabled=self._settings.audit_enabled))

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
    def llm_client(self, provider: str | None = None, model: str | None = None) -> AsyncLLMClient:
        """Build an LLM client, memoising the defaults.

        Pass ``provider`` / ``model`` to build a one-off client (per-request
        override in the query API). OpenAI endpoints are resolved through
        ``openai_base_url`` / ``openai_api_key``.
        """
        if provider is None and model is None:
            return self._memo("llm", self._build_llm_client)
        return self._build_llm_client(provider, model)

    def _build_llm_client(
        self, provider: str | None = None, model: str | None = None
    ) -> AsyncLLMClient:
        s = self._settings
        prov = provider or s.llm_provider
        base_url, api_key = s.llm_base_url, s.llm_api_key
        if prov == "openai":
            base_url = s.openai_base_url or s.llm_base_url
            api_key = s.openai_api_key or s.llm_api_key
        return AsyncLLMClient(
            provider=prov,
            model=model or s.llm_model,
            base_url=base_url,
            api_key=api_key,
            temperature=s.llm_temperature,
            max_tokens=s.llm_max_tokens,
            timeout=s.llm_timeout_seconds,
            connect_timeout=s.llm_request_timeout_seconds,
        )

    def researcher(self) -> ResearcherAgent:
        return self._memo(
            "researcher",
            lambda: ResearcherAgent(
                llm=self.llm_client(), max_context_docs=self._settings.final_top_k
            ),
        )

    def researcher_for(
        self, provider: str | None = None, model: str | None = None
    ) -> ResearcherAgent:
        """A researcher bound to a specific provider/model (per-request override)."""
        return ResearcherAgent(
            llm=self.llm_client(provider, model), max_context_docs=self._settings.final_top_k
        )

    def update_llm_settings(
        self, *, provider: str | None = None, model: str | None = None, api_key: str | None = None
    ) -> None:
        """Apply runtime LLM settings changes (no restart needed).

        Mutates the shared ``Settings`` instance and drops the memoised LLM
        client / researcher so the next query picks up the new defaults.
        """
        s = self._settings
        if provider:
            s.llm_provider = provider
        if model:
            s.llm_model = model
        if api_key:
            if s.llm_provider == "openai":
                s.openai_api_key = api_key
            else:
                s.llm_api_key = api_key
        self._cache.pop("llm", None)
        self._cache.pop("researcher", None)
        logger.info("LLM settings updated | provider={} | model={}", s.llm_provider, s.llm_model)

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
                audit=self.audit(),
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

    def list_sources(self) -> list[dict]:
        """Distinct metadata sources in the index, most chunks first."""
        try:
            counts = self.bm25_indexer().sources()
        except Exception:  # noqa: BLE001 - an empty index is a valid answer
            return []
        return [
            {"source": src, "count": cnt}
            for src, cnt in sorted(counts.items(), key=lambda item: -item[1])
        ]

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
