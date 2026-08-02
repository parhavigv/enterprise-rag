"""DenseRetriever unit tests - embedder and Chroma adapter are mocked."""

from __future__ import annotations

import pytest

from retrieval.dense import DenseRetriever
from retrieval.types import RetrievedDocument


class _FakeEmbedder:
    def __init__(self, dim: int = 4, fail: bool = False) -> None:
        self._dim = dim
        self._fail = fail

    def embed_query(self, text: str) -> list[float]:
        if self._fail:
            raise RuntimeError("embedding down")
        if not text.strip():
            raise ValueError("Query text must not be empty.")
        return [0.1] * self._dim


class _FakeAdapter:
    def __init__(self, docs: list[RetrievedDocument] | None = None) -> None:
        self._docs = docs or []
        self.query_calls = []

    def query(self, embedding, top_k: int = 20) -> list[RetrievedDocument]:
        self.query_calls.append((embedding, top_k))
        return self._docs[:top_k]


def _doc(node_id: str, text: str, score: float = 0.9) -> RetrievedDocument:
    return RetrievedDocument(
        node_id=node_id, text=text, score=score, metadata={"source": "t"}, source="dense"
    )


def test_retrieve_embeds_and_queries():
    docs = [_doc("a", "alpha"), _doc("b", "beta")]
    adapter = _FakeAdapter(docs)
    retriever = DenseRetriever(adapter=adapter, embedder=_FakeEmbedder())

    out = retriever.retrieve("alpha query", top_k=5)

    assert len(out) == 2
    assert adapter.query_calls == [([0.1, 0.1, 0.1, 0.1], 5)]
    assert out[0].node_id == "a"


def test_retrieve_respects_top_k():
    adapter = _FakeAdapter([_doc(f"d{i}", f"doc {i}") for i in range(10)])
    retriever = DenseRetriever(adapter=adapter, embedder=_FakeEmbedder())
    assert len(retriever.retrieve("q", top_k=3)) == 3


def test_retrieve_empty_query_raises():
    retriever = DenseRetriever(adapter=_FakeAdapter(), embedder=_FakeEmbedder())
    with pytest.raises(ValueError, match="empty"):
        retriever.retrieve("  ", top_k=5)


def test_retrieve_top_k_lt_one_raises():
    retriever = DenseRetriever(adapter=_FakeAdapter(), embedder=_FakeEmbedder())
    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("query", top_k=0)


def test_embedder_failure_propagates():
    adapter = _FakeAdapter()
    retriever = DenseRetriever(adapter=adapter, embedder=_FakeEmbedder(fail=True))
    with pytest.raises(RuntimeError, match="embedding down"):
        retriever.retrieve("query", top_k=5)


def test_adapter_property_exposed():
    adapter = _FakeAdapter()
    retriever = DenseRetriever(adapter=adapter, embedder=_FakeEmbedder())
    assert retriever.adapter is adapter
