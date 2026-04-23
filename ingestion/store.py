import chromadb
import logging
from typing import List, Dict, Any
from ingestion.config import CHROMA_PATH, CHROMA_COLLECTION

logger = logging.getLogger(__name__)

_client = None
_collection = None


def _get_collection():
    """Lazy-initialize ChromaDB client and collection."""
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = _client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"ChromaDB collection '{CHROMA_COLLECTION}' ready at '{CHROMA_PATH}'")
    return _collection


def save_chunks(chunks: List[Dict[str, Any]]) -> int:
    """
    Persist embedded chunks into ChromaDB.

    Args:
        chunks: list of dicts with keys:
                text, source, doc_type, chunk_index, char_start, embedding

    Returns:
        number of chunks saved
    """
    collection = _get_collection()

    ids, embeddings, documents, metadatas = [], [], [], []

    for chunk in chunks:
        uid = f"{chunk['source']}::chunk{chunk['chunk_index']}"
        ids.append(uid)
        embeddings.append(chunk["embedding"])
        documents.append(chunk["text"])
        metadatas.append({
            "source":      chunk["source"],
            "doc_type":    chunk["doc_type"],
            "chunk_index": chunk["chunk_index"],
            "char_start":  chunk["char_start"],
        })

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    logger.info(f"Saved {len(chunks)} chunks to ChromaDB.")
    return len(chunks)


def get_collection_count() -> int:
    """Return total number of chunks stored."""
    return _get_collection().count()