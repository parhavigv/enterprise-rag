import logging
from typing import List, Dict, Any
from ingestion.embedders.ollama_embedder import get_embeddings
from ingestion.store import _get_collection

logger = logging.getLogger(__name__)


def vector_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Embed the query and find the most similar chunks in ChromaDB.

    Args:
        query: user's question
        top_k: number of results to return

    Returns:
        list of dicts with keys: text, source, doc_type, score
    """
    query_vector = get_embeddings([query])[0]
    collection = _get_collection()

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text":     doc,
            "source":   meta.get("source", ""),
            "doc_type": meta.get("doc_type", ""),
            "score":    round(1 - dist, 4),  # cosine similarity
        })

    logger.info(f"Vector search returned {len(hits)} results for query: '{query}'")
    return hits