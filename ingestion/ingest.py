"""Ingestion orchestration: parse -> chunk -> embed -> vector store -> BM25.

Importable from the CLI (``python -m ingestion.ingest``) and from the API
(``IngestService``) without duplication.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from app.core.config import get_settings
from app.core.errors import IngestionError, InvalidInputError
from app.core.logging import get_logger
from ingestion.chunkers.semantic_chunker import chunk_documents
from ingestion.embedders.ollama_embedder import EmbedderError, OllamaEmbedder
from ingestion.parsers import PARSERS
from retrieval.bm25 import BM25Indexer
from retrieval.vector_store import ChromaAdapter

logger = get_logger(__name__)


def run_ingestion(
    path: str,
    fmt: str,
    chunk_size: str = "512T",
    collection_name: str = "enterprise_rag",
    rebuild_bm25: bool = False,
    embedder: OllamaEmbedder | None = None,
    adapter: ChromaAdapter | None = None,
    bm25_index_path: str | None = None,
) -> dict:
    """Run the full ingestion pipeline and return structured metrics.

    Raises:
        InvalidInputError: unknown format or unusable source.
        IngestionError: any pipeline-stage failure.
    """
    if fmt not in PARSERS:
        raise InvalidInputError(
            f"Unsupported format '{fmt}'. Allowed: {', '.join(sorted(PARSERS))}"
        )

    metrics: dict = {}
    t_total = time.perf_counter()
    try:
        logger.info("[1/5] Parsing {} as {} ...", path, fmt)
        docs = PARSERS[fmt](path)
        metrics["docs_parsed"] = len(docs)

        logger.info("[2/5] Chunking with config={} ...", chunk_size)
        nodes = chunk_documents(docs, config=chunk_size)
        metrics["chunks"] = len(nodes)

        logger.info("[3/5] Embedding {} chunks ...", len(nodes))
        embedder = embedder or OllamaEmbedder.from_env(get_settings())
        t0 = time.perf_counter()
        texts = [n.text for n in nodes]
        embeddings = embedder.embed(texts)
        for node, emb in zip(nodes, embeddings, strict=False):
            node.embedding = emb
        embed_time = time.perf_counter() - t0
        metrics["embed_throughput_chunks_per_min"] = (
            round(len(nodes) / embed_time * 60, 1) if embed_time > 0 else 0.0
        )

        logger.info("[4/5] Upserting {} nodes to ChromaDB ...", len(nodes))
        adapter = adapter or ChromaAdapter.from_env(get_settings())
        adapter.upsert(nodes)
        metrics["chroma_count"] = adapter.count()

        logger.info("[5/5] Building BM25 index ...")
        idx_path = Path(bm25_index_path or get_settings().bm25_index_path)
        indexer = BM25Indexer(index_path=idx_path)
        try:
            indexer = BM25Indexer.load(index_path=idx_path)
        except FileNotFoundError:
            logger.info("No existing BM25 index; building from scratch.")
        except Exception as e:  # noqa: BLE001 - corrupt/outdated index
            logger.warning("Could not load BM25 index ({}); rebuilding.", e)
        indexer.add(nodes)
        metrics["bm25_count"] = indexer.count()

    except InvalidInputError:
        raise
    except (FileNotFoundError, ValueError, EmbedderError, RuntimeError) as e:
        logger.error("Ingestion failed at stage: {}", e)
        raise IngestionError(str(e)) from e
    metrics["total_seconds"] = round(time.perf_counter() - t_total, 2)
    logger.info(
        "Ingestion complete in {:.2f}s | chunks={} | chroma={} | bm25={}",
        metrics["total_seconds"],
        metrics.get("chunks"),
        metrics.get("chroma_count"),
        metrics.get("bm25_count"),
    )
    return metrics


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Enterprise RAG - Ingestion CLI")
    p.add_argument("--path", required=True, help="File path or URL")
    p.add_argument("--format", required=True, choices=sorted(PARSERS))
    p.add_argument("--chunk-size", default="512T")
    p.add_argument("--collection-name", default="enterprise_rag")
    p.add_argument("--rebuild-bm25", action="store_true")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    try:
        result = run_ingestion(
            path=args.path,
            fmt=args.format,
            chunk_size=args.chunk_size,
            collection_name=args.collection_name,
            rebuild_bm25=args.rebuild_bm25,
        )
    except InvalidInputError as e:
        logger.error(str(e))
        raise SystemExit(2) from None
    except IngestionError as e:
        logger.error(str(e))
        raise SystemExit(1) from None

    print("\n── Ingestion Metrics ──────────────────────────────")
    for k, v in result.items():
        print(f"  {k:<40} {v}")
    print("───────────────────────────────────────────────────")
