import logging
import pickle
import os
from typing import List, Dict, Any
from ingestion.config import BM25_INDEX_PATH

logger = logging.getLogger(__name__)

_bm25_data = None


def _load_index():
    """Lazy-load the BM25 index from disk."""
    global _bm25_data
    if _bm25_data is None:
        if not os.path.exists(BM25_INDEX_PATH):
            logger.warning(f"BM25 index not found at {BM25_INDEX_PATH}")
            return None
        with open(BM25_INDEX_PATH, "rb") as f:
            _bm25_data = pickle.load(f)
        logger.info(f"BM25 index loaded from {BM25_INDEX_PATH}")
    return _bm25_data


def build_bm25_index(chunks: List[Dict[str, Any]]) -> None:
    """
    Build and save a BM25 index from chunks.

    Args:
        chunks: list of dicts with 'text', 'source', 'doc_type' keys
    """
    from rank_bm25 import BM25Okapi

    corpus = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(corpus)

    data = {"bm25": bm25, "chunks": chunks}
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(data, f)

    global _bm25_data
    _bm25_data = data
    logger.info(f"BM25 index built with {len(chunks)} chunks and saved.")


def bm25_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Keyword search using BM25.

    Args:
        query: user's question
        top_k: number of results to return

    Returns:
        list of dicts with keys: text, source, doc_type, score
    """
    data = _load_index()
    if data is None:
        logger.warning("BM25 index unavailable, returning empty results.")
        return []

    bm25 = data["bm25"]
    chunks = data["chunks"]

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    hits = []
    for i in top_indices:
        if scores[i] > 0:
            hits.append({
                "text":     chunks[i]["text"],
                "source":   chunks[i].get("source", ""),
                "doc_type": chunks[i].get("doc_type", ""),
                "score":    round(float(scores[i]), 4),
            })

    logger.info(f"BM25 search returned {len(hits)} results for query: '{query}'")
    return hits