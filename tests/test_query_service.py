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

    def retrieve(self, query, top_k=20, dense_k=50, sparse_k=50, source=None):
        self.calls.append((query, top_k, dense_k, sparse_k, source))
        return self._docs[:top_k]


class _FakeResearcher:
    def __init__(self, name: str = "researcher") -> None:
        self.name = name
        self.calls = []

    async def generate(self, query, documents, images=None):
        self.calls.append((query, documents, images))
        return SimpleNamespace(
            query=query,
            answer=f"the answer from {self.name}",
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
    assert result.answer == "the answer from researcher"
    assert result.generated is True
    assert result.model == "m"
    assert result.cache_hit is False


@pytest.mark.asyncio
async def test_answer_caches_identical_queries(service):
    r1 = await service.answer("Hello World", top_k=5)
    r2 = await service.answer("hello world", top_k=5)  # case-insensitive key
    assert r1.answer == r2.answer == "the answer from researcher"
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


def test_cache_key_distinguishes_model_and_images():
    base = QueryService._cache_key("q", 5, True, True)
    assert base != QueryService._cache_key("q", 5, True, True, "openai", "gpt-4o")
    assert QueryService._cache_key(
        "q", 5, True, True, "openai", "gpt-4o", ["img"]
    ) != QueryService._cache_key("q", 5, True, True, "openai", "gpt-4o")


def test_cache_key_distinguishes_source_scope():
    base = QueryService._cache_key("q", 5, True, True)
    assert base != QueryService._cache_key("q", 5, True, True, source="resume.pdf")
    assert QueryService._cache_key("q", 5, True, True, source="a.pdf") != QueryService._cache_key(
        "q", 5, True, True, source="b.pdf"
    )


@pytest.mark.asyncio
async def test_answer_forwards_source_to_retriever(service):
    result = await service.answer("query", top_k=5, source="resume.pdf")
    assert result.answer is not None
    assert service._hybrid.calls and service._hybrid.calls[0][4] == "resume.pdf"


def test_search_forwards_source_to_retriever(service):
    service.search("query", top_k=5, source="notes.txt")
    assert service._hybrid.calls[0][4] == "notes.txt"


@pytest.mark.asyncio
async def test_answer_uses_overridden_researcher(service):
    override = _FakeResearcher(name="override")
    result = await service.answer("query", top_k=5, researcher=override, images=["img"])
    assert result.answer == "the answer from override"
    assert override.calls and override.calls[0][2] == ["img"]


@pytest.mark.asyncio
async def test_answer_with_images_not_cached_with_plain_query(service):
    await service.answer("same", top_k=5)
    r_image = await service.answer("same", top_k=5, images=["img"])
    assert r_image.cache_hit is False  # distinct cache key
    assert len(service._hybrid.calls) == 2


def test_clear_cache(service):
    service._cache.set("k", "v")
    assert len(service._cache) == 1
    service.clear_cache()
    assert len(service._cache) == 0
