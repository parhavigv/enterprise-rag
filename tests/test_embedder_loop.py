"""Regression tests for OllamaEmbedder event-loop safety.

The embedder is called from synchronous retrieval code, but FastAPI runs
request handlers on its event loop where ``asyncio.run()`` raises
``RuntimeError``. ``_embed_sync`` must therefore fall back to a worker
thread when a loop is already running.
"""

from __future__ import annotations

import asyncio

import pytest

from ingestion.embedders.ollama_embedder import EmbedderError, OllamaEmbedder


class _FakeEmbedder(OllamaEmbedder):
    """Stub that never touches the network."""

    def __init__(self, max_retries: int = 3) -> None:
        super().__init__(base_url="http://unused", model="fake-model", max_retries=max_retries)
        self.calls = 0

    async def _embed_async(self, texts):  # noqa: D102
        self.calls += 1
        await asyncio.sleep(0.01)
        return [[float(i)] for i in range(len(texts))]


def test_embed_sync_without_loop():
    emb = _FakeEmbedder()
    result = emb.embed(["a", "b"])
    assert result == [[0.0], [1.0]]


@pytest.mark.asyncio
async def test_embed_sync_inside_running_loop():
    """asyncio.run() must not be called from within a running loop."""
    emb = _FakeEmbedder()

    async def call():
        return emb.embed_query("hello")

    result = await asyncio.wait_for(call(), timeout=10)
    assert result == [0.0]
    assert emb.calls == 1


@pytest.mark.asyncio
async def test_embed_sync_retries_failures_inside_loop():
    class _BoomEmbedder(_FakeEmbedder):
        async def _embed_async(self, texts):
            raise RuntimeError("boom")

    emb = _BoomEmbedder(max_retries=2)
    with pytest.raises(EmbedderError):
        await asyncio.wait_for(asyncio.to_thread(emb.embed_query, "x"), timeout=10)
