"""Persistent ChromaDB adapter.

Wraps upsert / query / delete / count and exposes a ``health()`` probe used
by the service readiness endpoint. Supports both the embedded persistent
client (default) and a remote Chroma server via ``CHROMA_SERVER_ENABLED``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import chromadb
from chromadb.config import Settings
from llama_index.core.schema import TextNode

from app.core.logging import get_logger
from retrieval.types import RetrievedDocument

logger = get_logger(__name__)


class ChromaAdapter:
    """Thin, defensive wrapper around a ChromaDB collection."""

    def __init__(
        self,
        path: str = "./data/chroma",
        collection: str = "enterprise_rag",
        server_enabled: bool = False,
        server_host: str = "localhost",
        server_port: int = 8000,
    ) -> None:
        try:
            if server_enabled:
                self.client = chromadb.HttpClient(
                    host=server_host,
                    port=server_port,
                    settings=Settings(anonymized_telemetry=False),
                )
            else:
                self.client = chromadb.PersistentClient(
                    path=path,
                    settings=Settings(anonymized_telemetry=False),
                )
            self.col = self.client.get_or_create_collection(
                name=collection,
                metadata={"hnsw:space": "cosine"},
            )
            self._dimension: int | None = None
            logger.info(
                "ChromaAdapter ready | collection={} | mode={}",
                collection,
                "server" if server_enabled else "embedded",
            )
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to initialise ChromaDB: {}", e)
            raise RuntimeError(f"ChromaDB init failed: {e}") from e

    # ------------------------------------------------------------------ #
    # Write path
    # ------------------------------------------------------------------ #
    def upsert(self, nodes: list[TextNode]) -> None:
        """Idempotent upsert of ``TextNode`` objects."""
        self._validate_nodes(nodes)
        try:
            ids = [n.node_id for n in nodes]
            embeddings = [n.embedding for n in nodes]
            documents = [n.text for n in nodes]
            metadatas = [self._sanitise_metadata(dict(n.metadata)) for n in nodes]

            self.col.upsert(
                ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
            )
            self._dimension = len(embeddings[0])
            logger.info(
                "Upserted {} nodes | collection total: {}",
                len(nodes),
                self.col.count(),
            )
        except ValueError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.error("ChromaDB upsert failed: {}", e)
            raise RuntimeError(f"Upsert failed: {e}") from e

    def delete(self, ids: Iterable[str]) -> None:
        """Delete nodes by id (no-op for unknown ids)."""
        id_list = list(ids)
        if not id_list:
            return
        try:
            self.col.delete(ids=id_list)
            logger.info("Deleted {} nodes", len(id_list))
        except Exception as e:  # noqa: BLE001
            logger.error("ChromaDB delete failed: {}", e)
            raise RuntimeError(f"Delete failed: {e}") from e

    # ------------------------------------------------------------------ #
    # Read path
    # ------------------------------------------------------------------ #
    def query(
        self,
        embedding: list[float],
        top_k: int = 20,
        where: dict | None = None,
    ) -> list[RetrievedDocument]:
        """Top-k cosine similarity search, returning typed hits.

        ``where`` is a ChromaDB metadata filter (e.g. ``{"source": "x.pdf"}``)
        used to scope retrieval to a single document.
        """
        if not embedding:
            raise ValueError("Query embedding is empty")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        try:
            kwargs: dict = {
                "query_embeddings": [embedding],
                "n_results": top_k,
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                kwargs["where"] = where
            results = self.col.query(**kwargs)
        except Exception as e:  # noqa: BLE001
            logger.error("ChromaDB query failed: {}", e)
            raise RuntimeError(f"Query failed: {e}") from e

        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        # Chroma distances are cosine *distances*: lower is better.
        return [
            RetrievedDocument(
                node_id=i,
                text=d,
                score=1.0 - dist,  # convert distance -> similarity in [0,1]
                metadata=m or {},
                source="dense",
            )
            for i, d, m, dist in zip(ids, docs, metas, dists, strict=False)
        ]

    def get(self, node_id: str) -> RetrievedDocument | None:
        """Fetch a single document by id."""
        try:
            result = self.col.get(ids=[node_id], include=["documents", "metadatas"])
        except Exception as e:  # noqa: BLE001
            logger.error("ChromaDB get failed: {}", e)
            raise RuntimeError(f"Get failed: {e}") from e
        ids = result.get("ids", [])
        if not ids:
            return None
        return RetrievedDocument(
            node_id=ids[0],
            text=result["documents"][0],
            score=1.0,
            metadata=result["metadatas"][0] or {},
            source="dense",
        )

    def count(self) -> int:
        try:
            return int(self.col.count())
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to get collection count: {}", e)
            raise RuntimeError(f"Count failed: {e}") from e

    def reset(self) -> None:
        """Drop and recreate the collection (fresh ingestion)."""
        try:
            self.client.delete_collection(self.col.name)
            self.col = self.client.get_or_create_collection(
                name=self.col.name,
                metadata={"hnsw:space": "cosine"},
            )
            self._dimension = None
            logger.warning("Collection reset - all embeddings cleared")
        except Exception as e:  # noqa: BLE001
            logger.error("Collection reset failed: {}", e)
            raise RuntimeError(f"Reset failed: {e}") from e

    # ------------------------------------------------------------------ #
    # Factories
    # ------------------------------------------------------------------ #
    @staticmethod
    def from_env(settings) -> ChromaAdapter:
        return ChromaAdapter(
            path=settings.chroma_path,
            collection=settings.chroma_collection,
            server_enabled=settings.chroma_server_enabled,
            server_host=settings.chroma_server_host,
            server_port=settings.chroma_server_port,
        )

    # ------------------------------------------------------------------ #
    # Health / introspection
    # ------------------------------------------------------------------ #
    def health(self) -> bool:
        try:
            self.col.count()
            return True
        except Exception:  # noqa: BLE001
            return False

    def dimension(self) -> int | None:
        if self._dimension is not None:
            return self._dimension
        try:
            count = self.col.count()
            if count == 0:
                return None
            sample = self.col.get(limit=1, include=["embeddings"])
            if sample.get("embeddings"):
                self._dimension = len(sample["embeddings"][0])
            return self._dimension
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sanitise_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """Keep only ChromaDB-safe value types (str/int/float/bool)."""
        sanitised: dict[str, Any] = {}
        for k, v in metadata.items():
            if v is None:
                continue
            if isinstance(v, str | int | float | bool):
                sanitised[k] = v
            else:
                sanitised[k] = str(v)
        return sanitised

    @staticmethod
    def _validate_nodes(nodes: list[TextNode]) -> None:
        if not nodes:
            raise ValueError("upsert called with empty nodes list")
        missing = [n.node_id for n in nodes if getattr(n, "embedding", None) is None]
        if missing:
            raise ValueError(
                f"{len(missing)} nodes have no embedding. First missing node_id: {missing[0]}"
            )
        dims = {len(n.embedding) for n in nodes}
        if len(dims) > 1:
            raise ValueError(f"Inconsistent embedding dimensions found: {dims}")
