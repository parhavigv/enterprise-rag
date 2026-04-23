import logging
from typing import List, Dict, Any

from retrieval.vector_store.chroma_retriever import vector_search
from retrieval.bm25.bm25_retriever import bm25_search
from retrieval.reranker.reranker import reciprocal_rank_fusion, deduplicate

logger = logging.getLogger(__name__)


def retrieve(
    query: str,
    top_k: int = 5,
    use_vector: bool = True,
    use_bm25: bool = True,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> List[Dict[str, Any]]:
    """
    Main retrieval entry point. Runs vector + BM25 search in parallel,
    fuses results with RRF, deduplicates, and returns top_k chunks.

    Args:
        query        : user's question
        top_k        : final number of chunks to return
        use_vector   : whether to run vector search
        use_bm25     : whether to run BM25 keyword search
        vector_weight: RRF weight for vector results
        bm25_weight  : RRF weight for BM25 results

    Returns:
        list of dicts — each with keys: text, source, doc_type, score
    """
    vector_hits: List[Dict[str, Any]] = []
    bm25_hits:   List[Dict[str, Any]] = []

    if use_vector:
        try:
            vector_hits = vector_search(query, top_k=top_k * 2)
        except Exception as e:
            logger.error(f"Vector search failed: {e}")

    if use_bm25:
        try:
            bm25_hits = bm25_search(query, top_k=top_k * 2)
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")

    # If only one source is available, fall back gracefully
    if not vector_hits and not bm25_hits:
        logger.warning("Both vector and BM25 search returned no results.")
        return []

    if vector_hits and not bm25_hits:
        results = deduplicate(vector_hits)
    elif bm25_hits and not vector_hits:
        results = deduplicate(bm25_hits)
    else:
        results = reciprocal_rank_fusion(
            vector_hits,
            bm25_hits,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
        )

    final = results[:top_k]
    logger.info(f"retrieve() returning {len(final)} chunks for query: '{query}'")
    return final