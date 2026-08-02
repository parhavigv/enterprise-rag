import pickle
import time
import uuid
from unittest.mock import MagicMock

import pytest

from retrieval.bm25 import BM25Indexer


def _make_node(text, node_id=None):
    node = MagicMock()
    node.node_id = node_id or str(uuid.uuid4())
    node.text = text
    return node


SAMPLE_TEXTS = [
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

SAMPLE_NODES = [_make_node(t) for t in SAMPLE_TEXTS]


class TestBM25Build:
    def test_build_creates_index_file(self, tmp_path):
        idx_path = tmp_path / "bm25_index.pkl"
        indexer = BM25Indexer(index_path=idx_path)
        indexer.build(SAMPLE_NODES)
        assert idx_path.exists()

    def test_build_count_matches_node_count(self, tmp_path):
        indexer = BM25Indexer(index_path=tmp_path / "idx.pkl")
        indexer.build(SAMPLE_NODES)
        assert indexer.count() == len(SAMPLE_NODES)

    def test_build_raises_on_empty_nodes(self, tmp_path):
        indexer = BM25Indexer(index_path=tmp_path / "idx.pkl")
        with pytest.raises(ValueError, match="empty"):
            indexer.build([])

    def test_build_is_fast_enough(self, tmp_path):
        large_nodes = [
            _make_node(f"chunk text sample number {i} for bm25 performance test")
            for i in range(5000)
        ]
        indexer = BM25Indexer(index_path=tmp_path / "large.pkl")
        t0 = time.perf_counter()
        indexer.build(large_nodes)
        elapsed = time.perf_counter() - t0
        assert elapsed < 60, f"Build took {elapsed:.1f}s - must be < 60s for 5k chunks."

    def test_build_skips_rebuild_on_unchanged_corpus(self, tmp_path):
        idx_path = tmp_path / "idx.pkl"
        indexer = BM25Indexer(index_path=idx_path)
        indexer.build(SAMPLE_NODES)
        mtime_1 = idx_path.stat().st_mtime
        time.sleep(0.05)
        indexer2 = BM25Indexer(index_path=idx_path)
        indexer2.build(SAMPLE_NODES)
        mtime_2 = idx_path.stat().st_mtime
        assert mtime_1 == mtime_2

    def test_build_force_always_rebuilds(self, tmp_path):
        idx_path = tmp_path / "idx.pkl"
        indexer = BM25Indexer(index_path=idx_path)
        indexer.build(SAMPLE_NODES)
        mtime_1 = idx_path.stat().st_mtime
        time.sleep(0.05)
        indexer.build(SAMPLE_NODES, force=True)
        mtime_2 = idx_path.stat().st_mtime
        assert mtime_2 > mtime_1


class TestBM25Persistence:
    def test_load_after_build(self, tmp_path):
        idx_path = tmp_path / "idx.pkl"
        BM25Indexer(index_path=idx_path).build(SAMPLE_NODES)
        loaded = BM25Indexer.load(index_path=idx_path)
        assert loaded.count() == len(SAMPLE_NODES)

    def test_load_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            BM25Indexer.load(index_path=tmp_path / "nonexistent.pkl")

    def test_query_works_after_load(self, tmp_path):
        idx_path = tmp_path / "idx.pkl"
        BM25Indexer(index_path=idx_path).build(SAMPLE_NODES)
        loaded = BM25Indexer.load(index_path=idx_path)
        results = loaded.query("BM25 sparse retrieval algorithm", top_k=3)
        assert len(results) == 3
        top_ids = [r[0] for r in results]
        bm25_node_id = SAMPLE_NODES[2].node_id
        assert bm25_node_id in top_ids

    def test_atomic_write_no_corruption(self, tmp_path):
        idx_path = tmp_path / "idx.pkl"
        BM25Indexer(index_path=idx_path).build(SAMPLE_NODES)
        with open(idx_path, "rb") as f:
            payload = pickle.load(f)
        assert len(payload.doc_ids) == len(SAMPLE_NODES)


class TestBM25Query:
    @pytest.fixture(autouse=True)
    def _build(self, tmp_path):
        self.indexer = BM25Indexer(index_path=tmp_path / "idx.pkl")
        self.indexer.build(SAMPLE_NODES)

    def test_query_returns_list_of_tuples(self):
        results = self.indexer.query("ingestion pipeline PDF", top_k=5)
        assert isinstance(results, list)
        for doc_id, score in results:
            assert isinstance(doc_id, str)
            assert isinstance(score, float)

    def test_query_respects_top_k(self):
        for k in [1, 3, 5, 10]:
            results = self.indexer.query("ingestion", top_k=k)
            assert len(results) <= k

    def test_query_sorted_descending(self):
        results = self.indexer.query("ingestion pipeline embedding", top_k=10)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_query_returns_correct_doc_ids(self):
        results = self.indexer.query("ChromaDB cosine similarity vector", top_k=5)
        valid_ids = {n.node_id for n in SAMPLE_NODES}
        for doc_id, _ in results:
            assert doc_id in valid_ids

    def test_query_before_build_raises(self, tmp_path):
        indexer = BM25Indexer(index_path=tmp_path / "unbuilt.pkl")
        with pytest.raises(RuntimeError, match="not loaded"):
            indexer.query("test query")

    def test_empty_query_handled_gracefully(self):
        results = self.indexer.query("", top_k=5)
        assert isinstance(results, list)

    def test_relevant_doc_ranks_higher_than_irrelevant(self):
        results = self.indexer.query(
            "nomic embed text 768 dimensional vectors", top_k=len(SAMPLE_NODES)
        )
        rank = {doc_id: i for i, (doc_id, _) in enumerate(results)}
        nomic_id = SAMPLE_NODES[4].node_id
        bm25_id = SAMPLE_NODES[2].node_id
        assert rank[nomic_id] < rank[bm25_id]


class TestWeek2InterfaceContract:
    def test_query_signature(self, tmp_path):
        indexer = BM25Indexer(index_path=tmp_path / "idx.pkl")
        indexer.build(SAMPLE_NODES)
        results = indexer.query("retrieval pipeline", top_k=20)
        assert isinstance(results, list)
        assert all(isinstance(t, tuple) and len(t) == 2 for t in results)

    def test_import_from_package(self):
        from retrieval.bm25 import BM25Indexer as _BM25Indexer

        assert _BM25Indexer is not None

    def test_classmethod_load(self, tmp_path):
        idx_path = tmp_path / "idx.pkl"
        BM25Indexer(index_path=idx_path).build(SAMPLE_NODES)
        loaded = BM25Indexer.load(index_path=idx_path)
        assert loaded.count() > 0
