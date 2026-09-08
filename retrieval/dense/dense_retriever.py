"""Dense (vector) retriever over ChromaDB.

Implements the Week 2 interface contract:
    dense_retriever.retrieve(query, top_k) -> list[RetrievedDocument]

RBAC extension: when ``user`` is passed, retrieval applies an additional
ChromaDB metadata ``where`` clause (clearance_level + department) so
inaccessible chunks are excluded at the vector-store level, before they are
even returned to the fusion layer.
"""

from __future__ import annotations

from app.auth.acl import visibility_filter
from app.auth.models import AuthUser
from ingestion.embedders.ollama_embedder import OllamaEmbedder
from retrieval.types import RetrievedDocument
from retrieval.vector_store import ChromaAdapter


class DenseRetriever:
    """Embeds a query and searches the ChromaDB collection."""

    def __init__(self, adapter: ChromaAdapter, embedder: OllamaEmbedder) -> None:
        self._adapter = adapter
        self._embedder = embedder

    @property
    def adapter(self) -> ChromaAdapter:
        return self._adapter

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        where: dict | None = None,
        user: AuthUser | None = None,
    ) -> list[RetrievedDocument]:
        if not query or not query.strip():
            raise ValueError("Query text must not be empty.")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        embedding = self._embedder.embed_query(query)
        # Only apply the clearance/department gate when a principal is
        # provided; otherwise defer to the retriever-level filter sweep so
        # the no-auth path keeps its original unfiltered behaviour.
        effective_where = visibility_filter(where, user) if user is not None else where
        return self._adapter.query(embedding, top_k=top_k, where=effective_where)
