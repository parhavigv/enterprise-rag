from __future__ import annotations
import logging
import time
from typing import List, Tuple
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class CrossEncoderReranker:
    """
    Cross-encoder re-ranker for Enterprise RAG pipeline.
    Takes top-20 candidates from HybridRetriever and re-ranks to top-5.
    Week 2 ResearcherAgent imports this class directly.

    Interface contract (frozen after Day 7):
        reranker.rerank(query, docs, top_k=5) -> List[Tuple[str, float]]
        reranker.rerank_with_ids(query, pairs, top_k=5) -> List[Tuple[str, str, float]]
    """

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        print(f"Loading cross-encoder: {model_name} ...")
        t0 = time.perf_counter()
        self._model = CrossEncoder(model_name)
        self._model_name = model_name
        print(f"Cross-encoder loaded in {time.perf_counter()-t0:.2f}s")

    def rerank(self, query: str, docs: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        if not docs:
            return []
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        top_k = min(top_k, len(docs))
        t0 = time.perf_counter()
        pairs = [(query, doc) for doc in docs]
        scores = self._model.predict(pairs).tolist()
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        elapsed = time.perf_counter() - t0
        print(f"Reranked {len(docs)} docs -> top-{top_k}  |  latency={elapsed:.3f}s  |  top_score={ranked[0][1]:.4f}")
        return ranked[:top_k]

    def rerank_with_ids(self, query: str, doc_id_pairs: List[Tuple[str, str]], top_k: int = 5) -> List[Tuple[str, str, float]]:
        if not doc_id_pairs:
            return []
        doc_ids   = [p[0] for p in doc_id_pairs]
        doc_texts = [p[1] for p in doc_id_pairs]
        reranked  = self.rerank(query, doc_texts, top_k=top_k)
        text_to_id = {text: did for did, text in zip(doc_ids, doc_texts)}
        return [(text_to_id.get(text, "unknown"), text, score) for text, score in reranked]

    @property
    def model_name(self) -> str:
        return self._model_name
