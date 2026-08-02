"""
tests/conftest.py
Shared pytest fixtures for the Enterprise RAG test suite.
"""

from __future__ import annotations

import os
import uuid
from unittest.mock import MagicMock

import pytest

# Must be set before llama_index/nltk is imported anywhere.
os.environ.setdefault("NLTK_DISABLE_IMPORT_SECURITY", "1")

from retrieval.bm25 import BM25Indexer  # noqa: E402

SAMPLE_TEXTS = [
    "The ingestion pipeline supports PDF, DOCX, and URL formats.",
    "ChromaDB stores dense vector embeddings using cosine distance.",
    "BM25Okapi is used for sparse term-frequency retrieval.",
    "The semantic chunker splits documents into 512-token chunks.",
    "nomic-embed-text produces 768-dimensional embedding vectors.",
    "The HybridRetriever fuses BM25 scores with ChromaDB distances.",
    "Cross-encoder re-ranking reorders top-k results for precision.",
    "FastAPI exposes the ingestion endpoint on Day 7.",
    "Pytest evaluates Recall@5 and Precision@5 on the gold-set.",
    "The .env file stores API keys and must never be committed.",
]


def make_node(text, node_id=None):
    node = MagicMock()
    node.node_id = node_id or str(uuid.uuid4())
    node.text = text
    node.metadata = {"source": "fixture"}
    return node


SAMPLE_NODES = [make_node(t) for t in SAMPLE_TEXTS]


@pytest.fixture(scope="session")
def shared_bm25(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("shared_bm25")
    indexer = BM25Indexer(index_path=tmp / "shared.pkl")
    indexer.build(SAMPLE_NODES)
    return indexer, SAMPLE_NODES
