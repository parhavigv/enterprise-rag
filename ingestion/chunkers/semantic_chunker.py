"""Semantic chunking via LlamaIndex ``SentenceSplitter``.

Documents are split on sentence boundaries into token-sized chunks with
overlap to preserve context across cut points.
"""

from __future__ import annotations

from typing import Any

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

from app.core.logging import get_logger
from ingestion.chunkers.config import DEFAULT_CHUNK_CONFIG, get_chunk_config

logger = get_logger(__name__)


def chunk_documents(
    docs: list[Document],
    config: str = DEFAULT_CHUNK_CONFIG,
    chunk_overlap: int | None = None,
) -> list[TextNode]:
    """Chunk ``docs`` into ``TextNode`` objects using a token-size config.

    Args:
        docs: Source documents (must be non-empty).
        config: One of the keys in ``CHUNK_CONFIGS`` (e.g. "512T").
        chunk_overlap: Optional override for the overlap (in tokens).

    Returns:
        List of :class:`TextNode` with metadata preserved from the document.

    Raises:
        ValueError: if ``docs`` is empty or ``config`` is unknown.
    """
    if not docs:
        raise ValueError("chunk_documents called with an empty document list.")

    cfg = get_chunk_config(config)
    overlap = chunk_overlap if chunk_overlap is not None else cfg.chunk_overlap

    parser = SentenceSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=overlap,
        paragraph_separator="\n\n",
        secondary_chunking_regex="[^,.;。]+[,.;。]?",
    )
    nodes = parser.get_nodes_from_documents(docs)

    if not nodes:
        raise ValueError("Chunking produced zero nodes from the supplied documents.")

    logger.info(
        "Chunked {} docs into {} nodes | config={} | overlap={}",
        len(docs),
        len(nodes),
        cfg.label,
        overlap,
    )
    return nodes


def chunk_metrics(nodes: list[TextNode]) -> dict[str, Any]:
    """Lightweight chunk-level statistics for ingestion reporting."""
    if not nodes:
        return {"chunks": 0, "avg_tokens": 0.0}
    token_counts = [len(n.get_content().split()) for n in nodes]
    return {
        "chunks": len(nodes),
        "avg_tokens": round(sum(token_counts) / len(token_counts), 1),
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
    }
