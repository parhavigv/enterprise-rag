"""ResearcherAgent unit tests - the LLM client is mocked."""

from __future__ import annotations

import pytest

from app.agents.llm_client import LLMResponse, LLMUnavailableError
from app.agents.researcher import ResearcherAgent, ResearchResponse
from retrieval.types import RetrievedDocument


def _docs(n: int = 3) -> list[RetrievedDocument]:
    return [
        RetrievedDocument(
            node_id=f"id-{i}",
            text=f"Passage number {i} about enterprise retrieval systems.",
            score=0.9 - i * 0.1,
            metadata={"source": f"doc-{i}.pdf", "page_number": i + 1},
            source="hybrid",
        )
        for i in range(n)
    ]


class _FakeLLM:
    def __init__(
        self, text: str = "Grounded answer [1][2].", model: str = "llama3.1", fail: bool = False
    ) -> None:
        self.text = text
        self.model = model
        self.fail = fail
        self.last_messages: list[dict] | None = None

    async def complete(self, messages, **kwargs):
        self.last_messages = messages
        if self.fail:
            raise LLMUnavailableError("mock failure")
        return LLMResponse(
            text=self.text,
            model=self.model,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            latency_ms=12.0,
        )


@pytest.mark.asyncio
async def test_generate_returns_grounded_answer():
    llm = _FakeLLM()
    agent = ResearcherAgent(llm=llm)
    resp: ResearchResponse = await agent.generate("question?", _docs())

    assert resp.generated is True
    assert resp.model == "llama3.1"
    assert resp.answer == "Grounded answer [1][2]."
    assert len(resp.sources) == 3


@pytest.mark.asyncio
async def test_prompt_includes_context_and_citations():
    llm = _FakeLLM()
    agent = ResearcherAgent(llm=llm)
    await agent.generate("question?", _docs())
    user_msg = llm.last_messages[-1]["content"]
    assert "[1] source=doc-0.pdf, page=1" in user_msg
    assert "Question: question?" in user_msg
    assert "Answer using only the context above" in user_msg


@pytest.mark.asyncio
async def test_llm_failure_falls_back_to_extractive():
    llm = _FakeLLM(fail=True)
    agent = ResearcherAgent(llm=llm)
    resp = await agent.generate("question?", _docs())

    assert resp.generated is False
    assert resp.model == "fallback-extractive"
    assert resp.error is not None
    assert resp.answer  # extractive summary non-empty


@pytest.mark.asyncio
async def test_no_documents_returns_notice():
    agent = ResearcherAgent(llm=_FakeLLM())
    resp = await agent.generate("question?", [])
    assert "No relevant documents" in resp.answer


@pytest.mark.asyncio
async def test_max_context_docs_limits_sources():
    agent = ResearcherAgent(llm=_FakeLLM(), max_context_docs=2)
    resp = await agent.generate("question?", _docs(n=5))
    assert len(resp.sources) == 2


@pytest.mark.asyncio
async def test_empty_query_raises():
    agent = ResearcherAgent(llm=_FakeLLM())
    with pytest.raises(ValueError, match="empty"):
        await agent.generate("   ", _docs())


def test_format_context_uses_metadata():
    formatted = ResearcherAgent._format_context(_docs(n=1))
    assert "doc-0.pdf" in formatted
    assert "page=1" in formatted


def test_extractive_summary_truncates_long_passages():
    long_doc = _docs(n=1)
    long_doc[0].text = "word " * 2000
    out = ResearcherAgent._extractive_summary(long_doc)
    assert len(out) < 700


@pytest.mark.asyncio
async def test_generate_without_images_uses_plain_string_content():
    llm = _FakeLLM()
    agent = ResearcherAgent(llm=llm)
    await agent.generate("question?", _docs())
    assert isinstance(llm.last_messages[-1]["content"], str)


@pytest.mark.asyncio
async def test_generate_with_images_builds_multimodal_content():
    llm = _FakeLLM()
    agent = ResearcherAgent(llm=llm)
    await agent.generate("what is this?", _docs(n=1), images=["data:image/png;base64,AAAA", "BBBB"])
    content = llm.last_messages[-1]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": content[0]["text"]}
    assert any(
        b["type"] == "image_url" and b["image_url"]["url"] == "data:image/png;base64,AAAA"
        for b in content
    )
    assert content[-1]["image_url"]["url"] == "data:image/png;base64,BBBB"


def test_image_data_uri_normalisation():
    assert (
        ResearcherAgent._image_data_uri("data:image/jpeg;base64,xx") == "data:image/jpeg;base64,xx"
    )
    assert ResearcherAgent._image_data_uri("base64,zz") == "data:image/png;base64,zz"
    assert ResearcherAgent._image_data_uri("rawbase64") == "data:image/png;base64,rawbase64"
