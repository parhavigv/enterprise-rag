# ingestion/ingest.py  — add these lines after chroma_adapter.upsert(nodes)

from retrieval.bm25 import BM25Indexer
import os

def run_ingestion(path, fmt, chunk_config="512T",
                  collection_name="enterprise_rag",
                  rebuild_bm25=False):

    # --- existing code: parse → chunk → embed → upsert to ChromaDB ---
    # ...

    # ---- Day 5: BM25 sparse index ----
    bm25_path = os.getenv("BM25_INDEX_PATH", "./bm25_index.pkl")
    indexer = BM25Indexer(index_path=bm25_path)

    if rebuild_bm25 or not os.path.exists(bm25_path):
        indexer.build(nodes)
        indexer.save()
    else:
        # append not supported by BM25Okapi — full rebuild on next --rebuild-bm25
        indexer.load()

    return indexer