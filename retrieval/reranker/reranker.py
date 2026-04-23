import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    vector_hits: List[Dict[str, Any]],
    bm25_hits: List[Dict[str, Any]],
    k: int = 60,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> List[Dict[str, Any]]:
    """
    Combine vector + BM25 results using Reciprocal Rank Fusion (RRF).

    Args:
        vector_hits : results from chroma_retriever.vector_search()
        bm25_hits   : results from bm25_retriever.bm25_search()
        k           : RRF smoothing constant (default 60)
        vector_weight: weight applied to vector RRF scores
        bm25_weight  : weight applied to BM25 RRF scores

    Returns:
        Deduplicated, reranked list of hits sorted by fused score (desc)
    """
    scores: Dict[str, float] = {}
    meta_map: Dict[str, Dict[str, Any]] = {}

    def _add(hits: List[Dict[str, Any]], weight: float):
        for rank, hit in enumerate(hits):
            key = hit["text"]  # deduplicate by exact text
            rrf_score = weight * (1.0 / (k + rank + 1))
            scores[key] = scores.get(key, 0.0) + rrf_score
            if key not in meta_map:
                meta_map[key] = hit

    _add(vector_hits, vector_weight)
    _add(bm25_hits, bm25_weight)

    fused = []
    for text, fused_score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        entry = dict(meta_map[text])
        entry["score"] = round(fused_score, 6)
        fused.append(entry)

    logger.info(f"RRF reranker produced {len(fused)} fused results.")
    return fused


def deduplicate(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate chunks by text content (case-insensitive strip).
    Keeps the highest-scored version.
    """
    seen = {}
    for hit in hits:
        key = hit["text"].strip().lower()
        if key not in seen or hit["score"] > seen[key]["score"]:
            seen[key] = hit
    return list(seen.values())