import logging
import chromadb
from chromadb.config import Settings
from llama_index.core.schema import TextNode
from typing import List

# Module level logger — replaces all print() statements
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)


class ChromaAdapter:
    """
    Persistent ChromaDB adapter — wraps upsert, dedup, and query.
    Week 2 DenseRetriever imports this class directly.
    """

    def __init__(
        self,
        path: str = './chroma_data',
        collection: str = 'enterprise_rag'
    ):
        try:
            self.client = chromadb.PersistentClient(
                path=path,
                settings=Settings(anonymized_telemetry=False)
            )
            self.col = self.client.get_or_create_collection(
                name=collection,
                metadata={'hnsw:space': 'cosine'}
            )
            logger.info(f"ChromaAdapter ready | collection: {collection} | path: {path}")
        except Exception as e:
            logger.error(f"Failed to initialise ChromaDB: {e}")
            raise RuntimeError(f"ChromaDB init failed: {e}") from e

    def _sanitise_metadata(self, metadata: dict) -> dict:
        """
        ChromaDB only accepts str, int, float, bool values.
        Converts everything else to string and drops None values.
        """
        sanitised = {}
        for k, v in metadata.items():
            if v is None:
                continue  # drop None values
            elif isinstance(v, (str, int, float, bool)):
                sanitised[k] = v
            else:
                sanitised[k] = str(v)  # convert lists, dicts, etc to string
        return sanitised

    def _validate_nodes(self, nodes: List[TextNode]) -> None:
        """
        Validate nodes before upsert.
        Raises ValueError if nodes are empty or missing embeddings.
        """
        if not nodes:
            raise ValueError("upsert called with empty nodes list")

        missing = [n.node_id for n in nodes if n.embedding is None]
        if missing:
            raise ValueError(
                f"{len(missing)} nodes have no embedding. "
                f"First missing node_id: {missing[0]}"
            )

        dims = set(len(n.embedding) for n in nodes)
        if len(dims) > 1:
            raise ValueError(
                f"Inconsistent embedding dimensions found: {dims}. "
                f"All embeddings must have the same dimension."
            )

    def upsert(self, nodes: List[TextNode]) -> None:
        """
        Upsert TextNodes into ChromaDB.
        Idempotent — re-ingesting same node_ids overwrites, never duplicates.
        """
        try:
            self._validate_nodes(nodes)

            ids        = [n.node_id for n in nodes]
            embeddings = [n.embedding for n in nodes]
            documents  = [n.text for n in nodes]
            metadatas  = [self._sanitise_metadata(n.metadata) for n in nodes]

            self.col.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"Upserted {len(nodes)} nodes | collection total: {self.col.count()}")

        except ValueError as e:
            logger.error(f"Validation error before upsert: {e}")
            raise
        except Exception as e:
            logger.error(f"ChromaDB upsert failed: {e}")
            raise RuntimeError(f"Upsert failed: {e}") from e

    def query(
        self,
        embedding: List[float],
        top_k: int = 20
    ) -> dict:
        """
        Top-k cosine similarity query.
        Returns ChromaDB result dict with ids, documents, metadatas, distances.
        Week 2 DenseRetriever calls this method directly.
        """
        try:
            if not embedding:
                raise ValueError("Query embedding is empty")

            results = self.col.query(
                query_embeddings=[embedding],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            logger.info(f"Query returned {len(results['documents'][0])} results | top_k={top_k}")
            return results

        except ValueError as e:
            logger.error(f"Validation error before query: {e}")
            raise
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            raise RuntimeError(f"Query failed: {e}") from e

    def count(self) -> int:
        """Return total documents in collection."""
        try:
            return self.col.count()
        except Exception as e:
            logger.error(f"Failed to get collection count: {e}")
            raise RuntimeError(f"Count failed: {e}") from e

    def reset(self) -> None:
        """Delete and recreate the collection — use for fresh ingestion only."""
        try:
            self.client.delete_collection(self.col.name)
            self.col = self.client.get_or_create_collection(
                name=self.col.name,
                metadata={'hnsw:space': 'cosine'}
            )
            logger.warning("Collection reset — all embeddings cleared")
        except Exception as e:
            logger.error(f"Collection reset failed: {e}")
            raise RuntimeError(f"Reset failed: {e}") from e