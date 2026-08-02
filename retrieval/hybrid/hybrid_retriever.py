"""Hybrid retriever fusing dense + sparse results with Reciprocal Rank
Fusion (RRF).

RRF is parameter-free-ish, robust to score-scale mismatch between dense
(similarity in [0,1]) and sparse (BM25 term-frequency scores), and is the
documented strategy for this project ("fuses BM25 sparse scores with
ChromaDB cosine distances using RRF").

Fusion formula::

    score(doc) = SUM_rankers  w_ranker / (rrf_k + rank(doc))
"""

from __future__ import annotations

from collections import defaultdict

from retrieval.bm25 import BM25Indexer
from retrieval.dense import DenseRetriever
from retrieval.types import RetrievedDocument


class HybridRetriever:
    def __init__(
        self,
        dense: DenseRetriever,
        sparse: BM25Indexer,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> None:
        if rrf_k < 1:
            raise ValueError("rrf_k must be >= 1")
        self._dense = dense
        self._sparse = sparse
        self._rrf_k = rrf_k
        self._dense_weight = dense_weight
        self._sparse_weight = sparse_weight

    def is_ready(self) -> bool:
        """True once the sparse index is loaded/built (dense is always ready)."""
        try:
            return self._sparse.count() > 0
        except Exception:  # noqa: BLE001
            return False

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        dense_k: int = 50,
        sparse_k: int = 50,
    ) -> list[RetrievedDocument]:
        if not query or not query.strip():
            raise ValueError("Query text must not be empty.")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        dense_hits = self._dense.retrieve(query, top_k=dense_k)
        sparse_hits = self._sparse.query_documents(query, top_k=sparse_k)

        fused = self.fuse(dense_hits, sparse_hits, top_k=top_k)
        for doc in fused:
            doc.source = "hybrid"
        return fused

    def fuse(
        self,
        dense_hits: list[RetrievedDocument],
        sparse_hits: list[RetrievedDocument],
        top_k: int = 20,
    ) -> list[RetrievedDocument]:
        """Merge two ranked lists by Reciprocal Rank Fusion."""
        scores: dict[str, float] = defaultdict(float)
        doc_by_id: dict[str, RetrievedDocument] = {}

        for hits, weight in ((dense_hits, self._dense_weight), (sparse_hits, self._sparse_weight)):
            for rank, hit in enumerate(hits):
                scores[hit.node_id] += weight / (self._rrf_k + rank + 1)
                # Prefer dense metadata/text when the same id appears in both.
                doc_by_id.setdefault(hit.node_id, hit)

        ranked_ids = sorted(scores, key=lambda i: scores[i], reverse=True)[:top_k]
        fused = [
            RetrievedDocument(
                node_id=doc_id,
                text=doc_by_id[doc_id].text,
                score=scores[doc_id],
                metadata=dict(doc_by_id[doc_id].metadata),
                source="hybrid",
            )
            for doc_id in ranked_ids
        ]
        return fused

    @staticmethod
    def normalized_dense_score(score: float) -> float:
        """Map cosine distance -> similarity in [0, 1]."""
        return max(0.0, min(1.0, score))
