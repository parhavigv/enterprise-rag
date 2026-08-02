"""Ingestion pipeline unit tests - embedder/vector store are fakes."""

from __future__ import annotations

import pytest
from llama_index.core import Document

import ingestion.ingest as ingest_mod
from ingestion.chunkers.config import get_chunk_config
from ingestion.chunkers.semantic_chunker import chunk_documents
from ingestion.embedders.ollama_embedder import EmbedderError, OllamaEmbedder
from ingestion.parsers.docx_parser import parse_docx
from ingestion.parsers.pdf_parser import parse_pdf
from ingestion.parsers.url_parser import parse_url


class _FakeEmbedder:
    def __init__(self, dim: int = 4) -> None:
        self._dim = dim

    def embed(self, texts):
        return [[0.1] * self._dim for _ in texts]


class _FakeAdapter:
    def __init__(self) -> None:
        self._count = 0
        self.upserted = []

    def upsert(self, nodes):
        self.upserted.extend(nodes)
        self._count += len(nodes)

    def count(self):
        return self._count


def _fake_parser(path):
    return [
        Document(
            text="The ingestion pipeline supports PDF, DOCX and URL formats. " * 10,
            metadata={"source": path},
        )
    ]


def test_run_ingestion_full_pipeline(tmp_path, monkeypatch):
    monkeypatch.setitem(ingest_mod.PARSERS, "pdf", _fake_parser)

    metrics = ingest_mod.run_ingestion(
        path="fake.pdf",
        fmt="pdf",
        chunk_size="512T",
        collection_name="test",
        rebuild_bm25=True,
        embedder=_FakeEmbedder(),
        adapter=_FakeAdapter(),
        bm25_index_path=str(tmp_path / "bm25.pkl"),
    )

    assert metrics["docs_parsed"] == 1
    assert metrics["chunks"] > 0
    assert metrics["chroma_count"] == metrics["chunks"]
    assert metrics["bm25_count"] == metrics["chunks"]
    assert "total_seconds" in metrics
    assert "embed_throughput_chunks_per_min" in metrics


def test_run_ingestion_invalid_format(tmp_path):
    from app.core.errors import InvalidInputError

    with pytest.raises(InvalidInputError, match="Unsupported format"):
        ingest_mod.run_ingestion(path="x", fmt="exe", bm25_index_path=str(tmp_path / "b.pkl"))


def test_run_ingestion_missing_file_raises(tmp_path):
    with pytest.raises(Exception, match="not found"):
        ingest_mod.run_ingestion(
            path=str(tmp_path / "nope.pdf"),
            fmt="pdf",
            embedder=_FakeEmbedder(),
            adapter=_FakeAdapter(),
            bm25_index_path=str(tmp_path / "b.pkl"),
        )


# --------------------------------------------------------------------- #
# Chunkers
# --------------------------------------------------------------------- #
def test_chunk_documents_validation():
    with pytest.raises(ValueError, match="empty"):
        chunk_documents([])


def test_chunk_documents_produces_nodes_with_metadata():
    doc = Document(
        text="One. Two. Three. Four. Five. Six. Seven. Eight. Nine. Ten. " * 20,
        metadata={"source": "x"},
    )
    nodes = chunk_documents([doc], config="512T")
    assert nodes
    assert all(n.metadata.get("source") == "x" for n in nodes)


def test_chunk_config_lookup():
    cfg = get_chunk_config("512T")
    assert cfg.chunk_size == 512
    with pytest.raises(ValueError, match="Unknown chunk config"):
        get_chunk_config("9999T")


# --------------------------------------------------------------------- #
# Parsers
# --------------------------------------------------------------------- #
def test_parse_pdf_missing_file():
    with pytest.raises(FileNotFoundError):
        parse_pdf("does-not-exist.pdf")


def test_parse_pdf_bad_extension(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("hello")
    with pytest.raises(ValueError, match="Unsupported file extension"):
        parse_pdf(str(f))


def test_parse_docx_missing_file():
    with pytest.raises(FileNotFoundError):
        parse_docx("does-not-exist.docx")


def test_parse_url_bad_scheme():
    with pytest.raises(ValueError, match="http"):
        parse_url("not-a-url")


def test_parse_url_fetch_failure(monkeypatch):
    def _fail(url):
        raise ValueError(f"Failed to fetch URL {url}: boom")

    monkeypatch.setattr("ingestion.parsers.url_parser._fetch", _fail)
    with pytest.raises(ValueError, match="boom"):
        parse_url("http://example.com")


# --------------------------------------------------------------------- #
# Embedder
# --------------------------------------------------------------------- #
def test_embedder_retries_then_succeeds(monkeypatch):
    embedder = OllamaEmbedder(base_url="http://x", model="m", max_retries=3, timeout=5)
    calls = {"n": 0}

    async def flaky(texts):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("boom")
        return [[1.0, 0.0]]

    monkeypatch.setattr(embedder, "_embed_async", flaky)
    monkeypatch.setattr("ingestion.embedders.ollama_embedder.time.sleep", lambda s: None)

    assert embedder.embed(["a"]) == [[1.0, 0.0]]
    assert calls["n"] == 3


def test_embedder_raises_after_max_retries(monkeypatch):
    embedder = OllamaEmbedder(base_url="http://x", model="m", max_retries=2, timeout=5)

    async def always_fail(texts):
        raise ConnectionError("down")

    monkeypatch.setattr(embedder, "_embed_async", always_fail)
    monkeypatch.setattr("ingestion.embedders.ollama_embedder.time.sleep", lambda s: None)

    with pytest.raises(EmbedderError, match="after 2 attempts"):
        embedder.embed(["a"])


def test_embedder_empty_input():
    embedder = OllamaEmbedder(base_url="http://x", model="m")
    assert embedder.embed([]) == []


def test_embedder_blank_query_raises():
    embedder = OllamaEmbedder(base_url="http://x", model="m")
    with pytest.raises(ValueError, match="empty"):
        embedder.embed_query("  ")


def test_embedder_health_returns_false_when_down():
    embedder = OllamaEmbedder(base_url="http://localhost:1", model="m")
    assert embedder.health() is False
