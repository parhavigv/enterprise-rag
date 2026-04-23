"""
Day 2 Smoke Test — Chunker + Embedder + ChromaDB
Run: python -m ingestion.test_day2_smoke
"""
import logging
from ingestion.chunkers.text_chunker import chunk_text
from ingestion.embedders.ollama_embedder import embed_chunks
from ingestion.store import save_chunks, get_collection_count

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SAMPLE_TEXT = """
Artificial intelligence is transforming the enterprise landscape.
Companies are adopting RAG pipelines to make their internal knowledge searchable.
A RAG pipeline typically consists of an ingestion phase and a retrieval phase.
During ingestion, documents are parsed, chunked, embedded, and stored in a vector database.
During retrieval, a user query is embedded and matched against stored vectors.
The most relevant chunks are then passed to a language model to generate an answer.
This approach grounds the model's response in real company data.
It reduces hallucinations and improves accuracy significantly.
Enterprise RAG systems must handle PDFs, Word documents, and web pages.
Chunking strategy and embedding quality are critical to retrieval performance.
""".strip()


def test_chunker():
    logger.info("=== TEST 1: Chunker ===")
    chunks = chunk_text(
        text=SAMPLE_TEXT,
        source="test_document.txt",
        doc_type="txt",
        chunk_size=200,
        chunk_overlap=30,
    )
    assert len(chunks) > 0, "Chunker returned no chunks!"
    for c in chunks:
        assert "text" in c
        assert "source" in c
        assert "chunk_index" in c
    logger.info(f"✅ Chunker OK — {len(chunks)} chunks created")
    return chunks


def test_embedder(chunks):
    logger.info("=== TEST 2: Embedder ===")
    embedded = embed_chunks(chunks)
    assert len(embedded) == len(chunks), "Mismatch between chunks and embeddings!"
    for c in embedded:
        assert "embedding" in c, "Missing embedding key!"
        assert len(c["embedding"]) > 0, "Empty embedding vector!"
    logger.info(f"✅ Embedder OK — vector size: {len(embedded[0]['embedding'])}")
    return embedded


def test_store(embedded_chunks):
    logger.info("=== TEST 3: ChromaDB Store ===")
    before = get_collection_count()
    saved = save_chunks(embedded_chunks)
    after = get_collection_count()
    assert saved == len(embedded_chunks), "Not all chunks were saved!"
    assert after >= before, "Collection count did not increase!"
    logger.info(f"✅ Store OK — {saved} chunks saved, total in DB: {after}")


if __name__ == "__main__":
    logger.info("Starting Day 2 Smoke Test...")
    chunks = test_chunker()
    embedded = test_embedder(chunks)
    test_store(embedded)
    logger.info("🎉 All Day 2 tests passed!")