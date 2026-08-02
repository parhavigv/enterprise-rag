from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from retrieval.reranker import CrossEncoderReranker

SAMPLE_DOCS = [
    "The ingestion pipeline parses PDF and DOCX files using LlamaIndex.",
    "ChromaDB stores dense vector embeddings with cosine similarity.",
    "BM25Okapi is a sparse retrieval algorithm based on term frequency.",
    "The semantic chunker splits documents into overlapping 512-token chunks.",
    "nomic-embed-text produces 768-dimensional embedding vectors.",
    "The HybridRetriever fuses BM25 scores with ChromaDB cosine distances.",
    "Cross-encoder re-ranking reorders top-k results for precision.",
    "FastAPI exposes the ingestion endpoint on Day 7 of the sprint.",
    "Pytest evaluates Recall@5 and Precision@5 on the gold-set queries.",
    "The .env file stores API keys and must never be committed to Git.",
]

QUERY = "How does the hybrid retriever combine BM25 and ChromaDB?"


@pytest.fixture(scope="module")
def mock_reranker():
    with patch("retrieval.reranker.cross_encoder.CrossEncoder") as MockCE:
        instance = MagicMock()

        def fake_predict(pairs, batch_size=None):
            scores = []
            for _, doc in pairs:
                q_words = set(QUERY.lower().split())
                d_words = set(doc.lower().split())
                scores.append(float(len(q_words & d_words)))
            return np.array(scores)

        instance.predict.side_effect = fake_predict
        MockCE.return_value = instance
        reranker = CrossEncoderReranker(model_name="mock-model", lazy_load=True)
        yield reranker


class TestRerank:
    def test_returns_list_of_tuples(self, mock_reranker):
        results = mock_reranker.rerank(QUERY, SAMPLE_DOCS, top_k=5)
        assert isinstance(results, list)
        assert len(results) == 5
        for text, score in results:
            assert isinstance(text, str)
            assert isinstance(score, float)

    def test_respects_top_k(self, mock_reranker):
        for k in [1, 3, 5]:
            results = mock_reranker.rerank(QUERY, SAMPLE_DOCS, top_k=k)
            assert len(results) == k

    def test_sorted_descending(self, mock_reranker):
        results = mock_reranker.rerank(QUERY, SAMPLE_DOCS, top_k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_relevant_doc_ranks_top(self, mock_reranker):
        results = mock_reranker.rerank(QUERY, SAMPLE_DOCS, top_k=5)
        top_texts = [t for t, _ in results]
        hybrid_doc = "The HybridRetriever fuses BM25 scores with ChromaDB cosine distances."
        assert hybrid_doc in top_texts

    def test_empty_docs_returns_empty(self, mock_reranker):
        results = mock_reranker.rerank(QUERY, [], top_k=5)
        assert results == []

    def test_empty_query_raises(self, mock_reranker):
        with pytest.raises(ValueError, match="empty"):
            mock_reranker.rerank("", SAMPLE_DOCS, top_k=5)

    def test_top_k_larger_than_docs_handled(self, mock_reranker):
        results = mock_reranker.rerank(QUERY, SAMPLE_DOCS[:3], top_k=10)
        assert len(results) == 3

    def test_all_docs_in_results_are_from_input(self, mock_reranker):
        results = mock_reranker.rerank(QUERY, SAMPLE_DOCS, top_k=5)
        for text, _ in results:
            assert text in SAMPLE_DOCS


class TestRerankWithIds:
    def test_returns_id_text_score_tuples(self, mock_reranker):
        pairs = [(f"id-{i}", doc) for i, doc in enumerate(SAMPLE_DOCS)]
        results = mock_reranker.rerank_with_ids(QUERY, pairs, top_k=5)
        assert len(results) == 5
        for doc_id, text, score in results:
            assert isinstance(doc_id, str)
            assert isinstance(text, str)
            assert isinstance(score, float)

    def test_ids_preserved_correctly(self, mock_reranker):
        pairs = [(f"id-{i}", doc) for i, doc in enumerate(SAMPLE_DOCS)]
        results = mock_reranker.rerank_with_ids(QUERY, pairs, top_k=5)
        id_map = {doc: did for did, doc in pairs}
        for doc_id, text, _ in results:
            assert doc_id == id_map[text]

    def test_empty_pairs_returns_empty(self, mock_reranker):
        results = mock_reranker.rerank_with_ids(QUERY, [], top_k=5)
        assert results == []


class TestWeek2InterfaceContract:
    def test_import_from_package(self):
        from retrieval.reranker import CrossEncoderReranker as CE

        assert CE is not None

    def test_rerank_signature(self, mock_reranker):
        results = mock_reranker.rerank(QUERY, SAMPLE_DOCS[:10], top_k=5)
        assert isinstance(results, list)
        assert all(len(t) == 2 for t in results)

    def test_rerank_with_ids_signature(self, mock_reranker):
        pairs = [(f"id-{i}", doc) for i, doc in enumerate(SAMPLE_DOCS[:10])]
        results = mock_reranker.rerank_with_ids(QUERY, pairs, top_k=5)
        assert isinstance(results, list)
        assert all(len(t) == 3 for t in results)

    def test_model_name_property(self, mock_reranker):
        assert mock_reranker.model_name == "mock-model"
