import requests
import logging
from typing import List, Dict, Any
from ingestion.config import OLLAMA_BASE_URL, EMBED_MODEL, EMBED_BATCH_SIZE

logger = logging.getLogger(__name__)


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Send texts to Ollama and return embedding vectors.

    Args:
        texts: list of strings to embed

    Returns:
        list of float vectors (one per input text)
    """
    embeddings = []

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        for text in batch:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": text},
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            embeddings.append(data["embedding"])
            logger.debug(f"Embedded 1 text, vector size: {len(data['embedding'])}")

    logger.info(f"Embedded {len(embeddings)} chunks total.")
    return embeddings


def embed_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add 'embedding' key to each chunk dict.

    Args:
        chunks: output from chunk_text()

    Returns:
        same chunks with 'embedding' field added
    """
    texts = [c["text"] for c in chunks]
    vectors = get_embeddings(texts)

    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector

    return chunks