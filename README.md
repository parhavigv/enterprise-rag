# Enterprise AI Research Assistant

A production-grade RAG (Retrieval-Augmented Generation) pipeline built over 4 weeks.

## Stack
- **Embeddings:** nomic-embed-text via Ollama (local, zero-cost)
- **Vector Store:** ChromaDB (dense retrieval)
- **Sparse Index:** BM25Okapi (keyword retrieval)
- **Re-ranker:** cross-encoder/ms-marco-MiniLM-L-6-v2
- **Framework:** LlamaIndex

## Week 1 — Ingestion Pipeline (In Progress)
- [x] Day 1: Repo scaffold, environment setup, smoke tests
- [x] Day 2: Multi-format parsers (PDF, DOCX, URL)
- [ ] Day 3: Semantic chunker + embedder
- [ ] Day 4: ChromaDB dense store
- [ ] Day 5: BM25 sparse indexer
- [ ] Day 6: Cross-encoder re-ranker
- [ ] Day 7: CLI + evaluation harness
