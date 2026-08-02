"""Cross-encoder re-ranker for the hybrid retrieval pipeline.

Implements the frozen Week 2 interface contract:

    reranker.rerank(query, docs, top_k=5) -> list[Tuple[str, float]]
    reranker.rerank_with_ids(query, pairs, top_k=5) -> list[Tuple[str, str, float]]

Production extras over the Week 1 version:
  - lazy model loading (the ~430MB model is only pulled on first use),
  - deterministic scoring, no numpy dependency in the public API,
  - ``rerank_documents`` operating on :class:`RetrievedDocument`.
"""

from __future__ import annotations

import logging
import time

from sentence_transformers import CrossEncoder

from retrieval.types import RetrievedDocument

logger = logging.getLogger(__name__)
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    def __init__(self, model_name: str = MODEL_NAME, lazy_load: bool = True) -> None:
        self._model_name = model_name
        self._model: CrossEncoder | None = None
        if not lazy_load:
            self._ensure_model()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def rerank(self, query: str, docs: list[str], top_k: int = 5) -> list[tuple[str, float]]:
        if not docs:
            return []
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        top_k = min(top_k, len(docs))
        pairs = [(query, doc) for doc in docs]
        scores = self._predict(pairs)
        ranked = sorted(zip(docs, scores, strict=False), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def rerank_with_ids(
        self, query: str, doc_id_pairs: list[tuple[str, str]], top_k: int = 5
    ) -> list[tuple[str, str, float]]:
        if not doc_id_pairs:
            return []
        doc_ids = [p[0] for p in doc_id_pairs]
        doc_texts = [p[1] for p in doc_id_pairs]
        reranked = self.rerank(query, doc_texts, top_k=top_k)
        text_to_id = {text: did for did, text in zip(doc_ids, doc_texts, strict=False)}
        return [(text_to_id.get(text, "unknown"), text, score) for text, score in reranked]

    def rerank_documents(
        self, query: str, docs: list[RetrievedDocument], top_k: int = 5
    ) -> list[RetrievedDocument]:
        """Re-rank typed retrieved documents, preserving metadata."""
        if not docs:
            return []
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        pairs = [(query, doc.text) for doc in docs]
        scores = self._predict(pairs)
        ranked = sorted(zip(docs, scores, strict=False), key=lambda x: x[1], reverse=True)
        out: list[RetrievedDocument] = []
        for doc, score in ranked[: min(top_k, len(docs))]:
            doc.score = float(score)
            doc.source = "reranked"
            out.append(doc)
        return out

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def loaded(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #
    def _ensure_model(self) -> None:
        if self._model is None:
            logger.info("Loading cross-encoder: %s ...", self._model_name)
            t0 = time.perf_counter()
            self._model = CrossEncoder(self._model_name)
            logger.info("Cross-encoder loaded in %.2fs", time.perf_counter() - t0)

    def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self._ensure_model()
        batch_size = min(64, max(8, len(pairs)))
        t0 = time.perf_counter()
        scores = self._model.predict(pairs, batch_size=batch_size)
        logger.info("Reranked %d pairs in %.3fs", len(pairs), time.perf_counter() - t0)
        return [float(s) for s in scores.tolist()]
