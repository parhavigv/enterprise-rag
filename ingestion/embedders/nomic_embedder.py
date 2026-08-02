"""Backward-compatible module facade for the Ollama embedder.

Existing code (`from ingestion.embedders.nomic_embedder import embed_documents`)
keeps working; new code should prefer ``OllamaEmbedder``.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from ingestion.embedders.ollama_embedder import EmbedderError, OllamaEmbedder  # noqa: F401

load_dotenv()

_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))

_default_embedder = OllamaEmbedder(base_url=_OLLAMA_URL, model=_MODEL, batch_size=_BATCH_SIZE)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed a list of document texts (module-level convenience)."""
    return _default_embedder.embed(texts)


def embed_query(text: str) -> list[float]:
    """Embed a single query text."""
    return _default_embedder.embed_query(text)
