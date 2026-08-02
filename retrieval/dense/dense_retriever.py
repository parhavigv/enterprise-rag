"""Dense (vector) retriever over ChromaDB.

Implements the Week 2 interface contract:
    dense_retriever.retrieve(query, top_k) -> list[RetrievedDocument]
"""

from __future__ import annotations

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

    def retrieve(self, query: str, top_k: int = 20) -> list[RetrievedDocument]:
        if not query or not query.strip():
            raise ValueError("Query text must not be empty.")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        embedding = self._embedder.embed_query(query)
        return self._adapter.query(embedding, top_k=top_k)
