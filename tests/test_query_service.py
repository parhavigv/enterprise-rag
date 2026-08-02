"""QueryService orchestration + caching unit tests."""

from __future__ import annotations

import pytest

from app.core.cache import TTLCache
from app.core.errors import RetrieverNotReadyError
from app.services.query_service import QueryService
from retrieval.types import RetrievedDocument


def _doc(node_id: str, text: str, score: float) -> RetrievedDocument:
    return RetrievedDocument(
        node_id=node_id, text=text, score=score, metadata={"src": node_id}, source="hybrid"
    )


class _FakeHybrid:
    def __init__(self, docs, ready: bool = True) -> None:
        self._docs = docs
        self._ready = ready
        self.calls = []

    def is_ready(self) -> bool:
        return self._ready

    def retrieve(self, query, top_k=20, dense_k=50, sparse_k=50):
        self.calls.append((query, top_k, dense_k, sparse_k))
        return self._docs[:top_k]


class _FakeResearcher:
    async def generate(self, query, documents):
        return SimpleNamespace(
            query=query,
            answer="the answer",
            sources=documents,
            model="m",
            generated=True,
            error=None,
        )


class _FakeReranker:
    def rerank_documents(self, query, docs, top_k=5):
        return docs[:top_k]


from types import SimpleNamespace  # noqa: E402


@pytest.fixture
def service():
    docs = [_doc("a", "alpha text", 0.9), _doc("b", "beta text", 0.8)]
    hybrid = _FakeHybrid(docs)
    qs = QueryService(
        hybrid=hybrid,
        researcher=_FakeResearcher(),
        reranker=_FakeReranker(),
        hybrid_top_k=20,
        dense_top_k=50,
        sparse_top_k=50,
        default_final_top_k=5,
        cache=TTLCache(ttl_seconds=60, max_entries=10),
    )
    return qs


def test_search_returns_documents(service):
    result = service.search("query", top_k=5, rerank=True)
    assert len(result.search) == 2
    assert result.answer is None


def test_search_no_rerank_truncates(service):
    service._reranker = None  # simulate reranker disabled
    result = service.search("query", top_k=1, rerank=False)
    assert len(result.search) == 1


@pytest.mark.asyncio
async def test_answer_generates(service):
    result = await service.answer("query", top_k=5, rerank=True, generate=True)
    assert result.answer == "the answer"
    assert result.generated is True
    assert result.model == "m"
    assert result.cache_hit is False


@pytest.mark.asyncio
async def test_answer_caches_identical_queries(service):
    r1 = await service.answer("Hello World", top_k=5)
    r2 = await service.answer("hello world", top_k=5)  # case-insensitive key
    assert r1.answer == r2.answer == "the answer"
    assert r2.cache_hit is True
    assert len(service._hybrid.calls) == 1  # second call served from cache


@pytest.mark.asyncio
async def test_answer_no_generate_skips_llm(service):
    result = await service.answer("query", top_k=5, rerank=True, generate=False)
    assert result.answer is None
    assert result.search


def test_not_ready_raises():
    qs = QueryService(
        hybrid=_FakeHybrid([], ready=False),
        researcher=_FakeResearcher(),
        reranker=None,
    )
    with pytest.raises(RetrieverNotReadyError):
        qs.search("query")


def test_cache_key_normalisation():
    assert QueryService._cache_key("Hello ", 5, True, True) == QueryService._cache_key(
        "hello", 5, True, True
    )


def test_clear_cache(service):
    service._cache.set("k", "v")
    assert len(service._cache) == 1
    service.clear_cache()
    assert len(service._cache) == 0
