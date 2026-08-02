"""Seed the index with documents.

Usage:
    python -m scripts.seed --path <file-or-url>
    python -m scripts.seed --dir <folder-with-pdfs-docx>
    python -m scripts.seed --sample          # embed a built-in demo corpus
"""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

from llama_index.core.schema import TextNode

from app.core.config import get_settings
from ingestion.chunkers.semantic_chunker import chunk_metrics
from ingestion.embedders.ollama_embedder import OllamaEmbedder
from ingestion.ingest import run_ingestion
from retrieval.bm25 import BM25Indexer
from retrieval.vector_store import ChromaAdapter

SAMPLE_CORPUS = [
    "The Enterprise RAG pipeline parses PDF, DOCX and URL sources into searchable chunks.",
    "Hybrid retrieval fuses ChromaDB dense vectors with BM25 sparse term scores using Reciprocal Rank Fusion.",
    "Cross-encoder re-ranking (ms-marco-MiniLM-L-6-v2) reorders the top candidates for precision.",
    "Embeddings are produced locally with Ollama using the nomic-embed-text model (768 dimensions).",
    "The default chunk size is 512 tokens with an 80-token overlap to preserve context.",
    "ChromaDB stores embeddings with cosine distance and supports idempotent upserts.",
    "The BM25 index is serialised atomically to bm25_index.pkl and reused across restarts.",
    "The FastAPI service exposes /query, /search, /ingest and /health endpoints.",
    "Caching reduces duplicate-query latency via an in-memory TTL cache.",
    "Readiness probes verify Ollama, ChromaDB and the BM25 index before serving traffic.",
]


def _format_for(path: Path) -> str | None:
    return {"pdf": "pdf", "docx": "docx", ".pdf": "pdf", ".docx": "docx"}.get(path.suffix.lower())


def ingest_path(path: str, chunk_size: str, rebuild: bool) -> dict:
    fmt = _format_for(Path(path))
    if fmt is None:
        if path.startswith(("http://", "https://")):
            fmt = "url"
        else:
            raise SystemExit(f"Cannot infer format for {path}")
    return run_ingestion(path=path, fmt=fmt, chunk_size=chunk_size, rebuild_bm25=rebuild)


def ingest_dir(dir_path: Path, chunk_size: str, rebuild: bool) -> None:
    files = [p for p in dir_path.iterdir() if p.is_file() and _format_for(p)]
    if not files:
        print(f"No supported files (pdf/docx) found in {dir_path}")
        return
    for i, f in enumerate(files, start=1):
        print(f"\n=== [{i}/{len(files)}] {f.name} ===")
        try:
            result = ingest_path(str(f), chunk_size, rebuild)
            print(f"  OK: {result['chunks']} chunks")
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {e}")


def ingest_sample(chunk_size: str) -> dict:
    """Build, embed and index the built-in demo corpus (no files required)."""
    settings = get_settings()
    nodes = [
        TextNode(
            text=text,
            node_id=str(uuid.uuid4()),
            metadata={"format": "sample", "source": "scripts.seed"},
        )
        for text in SAMPLE_CORPUS
    ]
    embedder = OllamaEmbedder.from_env(settings)
    embeddings = embedder.embed([n.text for n in nodes])
    for node, emb in zip(nodes, embeddings, strict=False):
        node.embedding = emb

    adapter = ChromaAdapter.from_env(settings)
    adapter.upsert(nodes)

    indexer = BM25Indexer(index_path=Path(settings.bm25_index_path))
    indexer.build(nodes, force=True)

    result = {
        "chunks": len(nodes),
        "chroma_count": adapter.count(),
        "bm25_count": indexer.count(),
        "chunk_stats": chunk_metrics(nodes),
    }
    print(f"Seeded demo corpus: {result}")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Seed the RAG index with documents")
    p.add_argument("--path", help="Single file path or URL")
    p.add_argument("--dir", help="Directory of PDF/DOCX files")
    p.add_argument("--sample", action="store_true", help="Embed the built-in demo corpus")
    p.add_argument("--chunk-size", default="512T")
    p.add_argument("--rebuild-bm25", action="store_true")
    args = p.parse_args()

    if args.sample:
        ingest_sample(args.chunk_size)
    elif args.path:
        result = ingest_path(args.path, args.chunk_size, args.rebuild_bm25)
        print(f"\nIngested {result['chunks']} chunks into Chroma + BM25.")
    elif args.dir:
        ingest_dir(Path(args.dir), args.chunk_size, args.rebuild_bm25)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
