from __future__ import annotations
import argparse
import logging
import os
import time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from ingestion.parsers.pdf_parser import parse_pdf
from ingestion.parsers.docx_parser import parse_docx
from ingestion.parsers.url_parser import parse_url
from ingestion.chunkers.semantic_chunker import chunk_documents
from ingestion.embedders.nomic_embedder import embed_documents
from retrieval.vector_store import ChromaAdapter
from retrieval.bm25 import BM25Indexer

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)
PARSERS = {"pdf": parse_pdf, "docx": parse_docx, "url": parse_url}

def run_ingestion(path, fmt, chunk_size="512T", collection_name="enterprise_rag", rebuild_bm25=False):
    metrics = {}
    t_total = time.perf_counter()
    logger.info("[1/5] Parsing %s as %s ...", path, fmt)
    docs = PARSERS[fmt](path)
    metrics["docs_parsed"] = len(docs)
    logger.info("[2/5] Chunking with config=%s ...", chunk_size)
    nodes = chunk_documents(docs, config=chunk_size)
    metrics["chunks"] = len(nodes)
    logger.info("[3/5] Embedding %d chunks ...", len(nodes))
    t0 = time.perf_counter()
    texts = [n.text for n in nodes]
    embeddings = embed_documents(texts)
    for node, emb in zip(nodes, embeddings):
        node.embedding = emb
    embed_time = time.perf_counter() - t0
    metrics["embed_throughput_chunks_per_min"] = round(len(nodes) / embed_time * 60, 1) if embed_time > 0 else 0
    logger.info("[4/5] Upserting to ChromaDB collection=%s ...", collection_name)
    adapter = ChromaAdapter(collection=collection_name)
    adapter.upsert(nodes)
    metrics["chroma_count"] = adapter.count()
    bm25_path = Path(os.getenv("BM25_INDEX_PATH", "./bm25_index.pkl"))
    indexer = BM25Indexer(index_path=bm25_path)
    if rebuild_bm25 or not bm25_path.exists():
        logger.info("[5/5] Building BM25 index ...")
        indexer.build(nodes, force=rebuild_bm25)
    else:
        logger.info("[5/5] Loading existing BM25 index ...")
        try:
            indexer = BM25Indexer.load(index_path=bm25_path)
        except Exception as e:
            logger.warning("Could not load BM25 (%s) - rebuilding.", e)
            indexer.build(nodes)
    metrics["bm25_count"] = indexer.count()
    metrics["total_seconds"] = round(time.perf_counter() - t_total, 2)
    logger.info("Ingestion complete in %.2fs | chunks=%d | chroma=%d | bm25=%d",
        metrics["total_seconds"], metrics["chunks"], metrics["chroma_count"], metrics["bm25_count"])
    return metrics

def _build_parser():
    p = argparse.ArgumentParser(description="Enterprise RAG - Ingestion CLI")
    p.add_argument("--path",            required=True,  help="File path or URL")
    p.add_argument("--format",          required=True,  choices=["pdf","docx","url"])
    p.add_argument("--chunk-size",      default="512T", choices=["256T","512T","1024T"])
    p.add_argument("--collection-name", default="enterprise_rag")
    p.add_argument("--rebuild-bm25",    action="store_true")
    return p

if __name__ == "__main__":
    args = _build_parser().parse_args()
    metrics = run_ingestion(
        path=args.path, fmt=args.format,
        chunk_size=args.chunk_size,
        collection_name=args.collection_name,
        rebuild_bm25=args.rebuild_bm25,
    )
    print("\n── Ingestion Metrics ──────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<40} {v}")
    print("───────────────────────────────────────────────────")