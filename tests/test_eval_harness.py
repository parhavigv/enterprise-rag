import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from retrieval.bm25 import BM25Indexer

GOLD_SET_PATH = Path("tests/eval/gold_set.json")

CORPUS = [
    "The ingestion pipeline supports PDF, DOCX, and URL formats for document parsing.",
    "nomic-embed-text produces 768-dimensional embedding vectors via Ollama local inference.",
    "The default chunk size configuration is 512T with 15 percent overlap between chunks.",
    "ChromaDB stores dense vector embeddings using cosine distance similarity metric.",
    "BM25Okapi is used for sparse term-frequency retrieval alongside dense ChromaDB vectors.",
    "Chunk overlap is set to 15 percent of the chunk size to preserve context boundaries.",
    "The cross-encoder ms-marco-MiniLM-L-6-v2 model is used for re-ranking top-k results.",
    "ChromaDB uses cosine distance as its similarity metric for vector search queries.",
    "The target Recall@5 for Week 1 evaluation is 0.75 on the gold-set of 10 questions.",
    "The BM25 index is serialised to disk as bm25_index.pkl using Python pickle format.",
    "LlamaIndex SentenceSplitter is used as the NodeParser for semantic chunking.",
    "The HybridRetriever fuses BM25 sparse scores with ChromaDB cosine distances using RRF.",
    "Week 2 DenseRetriever imports ChromaAdapter directly from retrieval.vector_store.",
    "The ingestion CLI accepts --path, --format, --chunk-size and --rebuild-bm25 arguments.",
    "Embeddings are generated in batches with exponential backoff retry logic on failure.",
]

def _make_node(text):
    node = MagicMock()
    node.node_id = str(uuid.uuid4())
    node.text = text
    return node

@pytest.fixture(scope="module")
def bm25_indexer(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bm25")
    nodes = [_make_node(t) for t in CORPUS]
    indexer = BM25Indexer(index_path=tmp / "test_index.pkl")
    indexer.build(nodes)
    return indexer, nodes


@pytest.fixture(scope="module")
def gold_set():
    with open(GOLD_SET_PATH, "r", encoding="utf-8-sig") as f:
        return json.load(f)


class TestGoldSetEval:
    def test_gold_set_file_exists(self):
        assert GOLD_SET_PATH.exists()

    def test_gold_set_has_10_questions(self, gold_set):
        assert len(gold_set) == 10

    def test_gold_set_schema(self, gold_set):
        for item in gold_set:
            assert "question" in item
            assert "relevant_keywords" in item
            assert len(item["relevant_keywords"]) >= 1

    def test_recall_at_5(self, bm25_indexer, gold_set):
        indexer, nodes = bm25_indexer
        id_to_text = {n.node_id: n.text for n in nodes}
        hits = 0
        for item in gold_set:
            keywords = [k.lower() for k in item["relevant_keywords"]]
            results = indexer.query(item["question"], top_k=5)
            retrieved = [id_to_text.get(doc_id, "").lower() for doc_id, _ in results]
            hit = any(any(kw in text for kw in keywords) for text in retrieved)
            hits += int(hit)
        recall = hits / len(gold_set)
        print(f"\n  Recall@5 = {recall:.2f}  ({hits}/{len(gold_set)})")
        assert recall >= 0.70, f"Recall@5 = {recall:.2f} below target 0.70"

    def test_precision_at_5(self, bm25_indexer, gold_set):
        indexer, nodes = bm25_indexer
        id_to_text = {n.node_id: n.text for n in nodes}
        total = 0.0
        for item in gold_set:
            keywords = [k.lower() for k in item["relevant_keywords"]]
            results = indexer.query(item["question"], top_k=5)
            relevant = sum(1 for doc_id, _ in results if any(kw in id_to_text.get(doc_id, "").lower() for kw in keywords))
            total += relevant / len(results) if results else 0
        avg = total / len(gold_set)
        print(f"\n  Precision@5 = {avg:.2f}")
        assert avg >= 0.40, f"Precision@5 = {avg:.2f} below target 0.40"

    def test_all_questions_return_results(self, bm25_indexer, gold_set):
        indexer, _ = bm25_indexer
        for item in gold_set:
            results = indexer.query(item["question"], top_k=5)
            assert len(results) > 0

    def test_results_sorted_by_score(self, bm25_indexer, gold_set):
        indexer, _ = bm25_indexer
        for item in gold_set:
            results = indexer.query(item["question"], top_k=5)
            scores = [s for _, s in results]
            assert scores == sorted(scores, reverse=True)


