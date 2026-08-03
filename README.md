# Enterprise RAG

A production-grade retrieval-augmented generation platform: multi-format ingestion,
hybrid retrieval (dense + sparse fused with Reciprocal Rank Fusion), cross-encoder
re-ranking, and grounded LLM answers — exposed as a containerised FastAPI service with
health probes, structured logging, caching, and CI.

**Author:** Parhavi G.V | Senior AI Engineer
**Stack:** Python 3.10 · FastAPI · LlamaIndex · ChromaDB · BM25Okapi · nomic-embed-text · sentence-transformers · Ollama · OpenAI (GPT-4o/4.1, Whisper) · Docker

---

## Architecture

```
                    ┌────────────────────────────────────────────────────────┐
                    │                     FastAPI service                    │
                    │  /api/v1/ingest  /api/v1/upload  /api/v1/search       │
                    │  /api/v1/query  /api/v1/transcribe  /health           │
                    │  /health/ready  /api/v1/stats                         │
                    │  /ui  (modern browser workspace)                      │
                    └───────┬───────────────┬────────────────┬───────────────┘
                            │               │                │
              ingest        │     retrieve   │     generate   │
                    ▼       │               ▼                ▼
        ┌─────────────────┐ │   ┌────────────────────┐  ┌──────────────────┐
        │ Parser +        │ │   │ HybridRetriever    │  │ ResearcherAgent  │
        │ SemanticChunker │ │   │  ├ Dense (ChromaDB)│  │  └ AsyncLLMClient│
        └───────┬─────────┘ │   │  └ Sparse (BM25)   │  │   (Ollama/OpenAI)│
                ▼           │   └────────┬───────────┘  └──────────────────┘
        ┌─────────────────┐ │            ▼
        │ OllamaEmbedder  │ │   ┌────────────────────┐
        │ (nomic-embed-   │ │   │ CrossEncoderReranker│
        │  text, 768d)    │ │   └────────────────────┘
        └─────────────────┘ │
                            │   in-memory TTL cache (query responses)
```

- **Ingestion** — `ingestion/ingest.py` runs a pluggable parser → semantic chunker →
  Ollama embedder → ChromaDB upsert → BM25 incremental merge pipeline (new chunks are
  merged into the persisted BM25 corpus instead of a full rebuild). PDFs also get
  **best-effort OCR** of embedded images (PyMuPDF + RapidOCR), so scanned documents are
  searchable too.
- **Retrieval** — `retrieval/hybrid/hybrid_retriever.py` fuses dense hits (ChromaDB,
  cosine) with sparse hits (BM25Okapi) via RRF, then `retrieval/reranker/cross_encoder.py`
  re-orders the top candidates.
- **Answering** — `app/agents/researcher.py` builds a grounded prompt from the cited
  passages and streams through `app/agents/llm_client.py` (Ollama or OpenAI; multimodal
  images supported), with an extractive fallback when the LLM is unavailable.
- **Voice** — `app/api/routes/speech.py` transcribes browser-recorded audio via OpenAI
  Whisper so you can ask questions by speaking.
- **Serving** — `app/main.py` app factory with CORS, request-ID propagation, structured
  logging, typed exception handlers, and JSON health/stats endpoints.

---

## Quick Start (local)

Requires Python 3.10+ and a running [Ollama](https://ollama.com) server.

```bash
# 1. Environment
python -m venv .venv
.\.venv\Scripts\activate            # Windows
pip install -e .                    # installs app + dev deps (pytest, ruff)
pip install -e ".[ocr]"             # optional: OCR for scanned PDF images

# 2. Models (once)
ollama pull nomic-embed-text
ollama pull llama3.1                # or any chat model; set LLM_MODEL accordingly

# 3. Config
copy .env.example .env              # adjust LLM_MODEL / endpoints as needed

# 4. Seed the index with the built-in demo corpus (no files needed)
python -m scripts.seed --sample

# 5. Run the API
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# 6. (Optional) Upload your own files in the browser
#    Open http://127.0.0.1:8000/ui
```

Try it:

```bash
curl http://127.0.0.1:8000/api/v1/query ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"What is the default chunk size?\", \"top_k\": 3}"

curl http://127.0.0.1:8000/api/v1/stats
curl http://127.0.0.1:8000/health/ready
```

Interactive docs: <http://127.0.0.1:8000/docs>.

---

## API Reference

All business endpoints live under `/api/v1`.

### `POST /api/v1/query`
Full RAG round-trip: hybrid retrieve → (optional) rerank → grounded LLM answer.
Supports per-request **provider/model overrides** and **vision chat** (attach
base64 images that are sent to the model alongside the question).

```jsonc
// Request
{
  "query": "What formats does ingestion support?",
  "top_k": 5,        // 1..20
  "rerank": true,    // cross-encoder re-ranking
  "generate": true,  // LLM answer; false → retrieval only
  "provider": "openai",   // optional override: ollama | openai
  "model": "gpt-4o",      // optional override, e.g. gpt-4o / gpt-4.1 / llama3
  "source": "resume.pdf", // optional: restrict retrieval to one uploaded file
  "images": ["data:image/png;base64,..."]   // optional vision input (max 4)
}

// Response
{
  "query": "...",
  "search": [{"node_id": "...", "text": "...", "score": 0.9, "metadata": {...}, "source": "hybrid"}],
  "answer": "Ingestion supports PDF, DOCX and URL sources. [1]",
  "model": "gpt-4o",
  "generated": true,
  "latency_ms": 1200.5,
  "error": null,
  "cache_hit": false
}
```

### `POST /api/v1/transcribe`
Voice question input. Multipart audio upload (webm/ogg/mp3/wav) transcribed
with OpenAI Whisper (`whisper-1` by default) and returned as text, ready to be
asked. Requires `OPENAI_API_KEY`.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/transcribe -F "file=@question.webm"
# {"text": "how much does the business plan cost?", "model": "whisper-1", ...}
```

### `POST /api/v1/search`
Retrieval-only variant of `/query` (`generate: false`), returning the same `search` array.

### `POST /api/v1/ingest`
Parse, chunk, embed, and index a document or URL.

```jsonc
{
  "path": "./data/raw/sample.pdf",   // or https://...
  "format": "pdf",                   // pdf | docx | txt | url
  "chunk_size": "512T",              // 256T | 512T | 1024T
  "collection_name": "enterprise_rag",
  "rebuild_bm25": false,
  "background": false                // true → returns immediately, runs as a task
}
```

### `POST /api/v1/upload`
Multipart file upload — the browser-friendly way to add documents. Accepts
`.pdf`, `.docx`, `.txt`, `.md`, `.markdown`, `.csv`, and `.json` (up to 50 MB).
The file is staged, parsed, chunked, embedded, and indexed into both dense and
BM25 stores, then deleted. Returns the same metrics as `/ingest`.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/upload ^
  -F "file=@guide.txt" -F "chunk_size=512T"
```

```jsonc
// Response
{
  "status": "ok",
  "background": false,
  "metrics": { "docs_parsed": 1, "chunks": 3,
               "chroma_count": 13, "bm25_count": 13, ... },
  "filename": "guide.txt"
}
```

### `GET /ui`
Modern browser workspace (no build step): drag-and-drop upload, chat with
sources, **attach images** for vision chat, a **record-your-voice** mic button
(Whisper), and a **scope selector** that answers from a single uploaded file
("Review this document" runs a structured Overview / Strengths / Gaps & risks /
Recommendations / Bottom line analysis). The **Answer engine** card changes
provider / model / API key at runtime, exactly like `PUT /api/v1/settings`.

### `GET /api/v1/stats`
Index statistics: Chroma collection + count, BM25 path + count, active embed/LLM models.

### `GET` / `PUT /api/v1/settings`
Runtime answer-engine configuration — switch provider / model / API key **without
a server restart**. `PUT` accepts any subset of `{provider, model, api_key}` and
immediately affects the next question (memoised LLM clients are rebuilt).

```bash
curl -X PUT http://127.0.0.1:8000/api/v1/settings ^
  -H "Content-Type: application/json" ^
  -d '{"provider": "openai", "model": "gpt-4o", "api_key": "sk-..."}'
```

```jsonc
// GET response
{
  "provider": "openai",
  "model": "gpt-4o",
  "base_url": "https://api.openai.com/v1",
  "api_key_masked": "sk-…123",        // null when no key is configured
  "whisper_model": "whisper-1",
  "providers": { "ollama": ["llama3", ...], "openai": ["gpt-4o", ...] }
}
```

### Health
- `GET /health` — liveness (always 200 when the process is up).
- `GET /health/ready` — readiness; 200 only when Chroma, BM25, and the Ollama embedder
  all probe green (used by the Docker healthcheck and orchestration).

---

## Docker / Compose

```bash
# Local Compose stack: API + Ollama + models pre-pull (nomic-embed-text + LLM_MODEL)
copy .env.example .env
docker compose up --build -d
```

- `Dockerfile` — slim Python 3.10-slim image, tini init, non-root user, healthcheck,
  NLTK resources pre-fetched at build time.
- `docker-compose.yml` — `api` waits for `ollama` (service_healthy) and `models-init`
  (service_completed_successfully) before starting; data lives on the `rag_data` volume.
- `models-init` — one-shot service that pulls `nomic-embed-text` and `${LLM_MODEL}`.

## CI

`.github/workflows/ci.yml` runs on `main`/`master`:

1. **Lint** — `ruff check` + `ruff format --check` (line length 100).
2. **Unit tests** — full pytest suite (146 tests); Ollama-dependent tests are
   marker-deselected in CI (`-m "not ollama"`).
3. **Docker build** — verifies the image builds and healthchecks pass.

---

## Configuration

Everything is driven by environment variables (see `.env.example`). Key settings:

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_PROVIDER` | `ollama` | `ollama` or `openai` (any OpenAI-compatible API) |
| `LLM_MODEL` | `llama3.1` | Chat model used for answer generation |
| `LLM_BASE_URL` | `http://localhost:11434/v1` | Chat-completions endpoint |
| `OPENAI_API_KEY` | — | OpenAI key; required for GPT, vision, and Whisper |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible base URL |
| `WHISPER_MODEL` | `whisper-1` | Speech-to-text model for the mic button |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Embedding endpoint (`/api/embeddings`) |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model (768-d vectors) |
| `DENSE_TOP_K` / `SPARSE_TOP_K` | `50` / `50` | Candidates per retriever |
| `HYBRID_TOP_K` | `20` | Post-fusion candidates |
| `FINAL_TOP_K` | `5` | Results returned to the client |
| `RRF_K` | `60` | RRF fusion constant |
| `RERANKER_ENABLED` | `true` | Toggle cross-encoder re-ranking |
| `CACHE_ENABLED` / `CACHE_TTL_SECONDS` | `true` / `600` | In-memory response cache |
| `LOG_FORMAT` | `json` | `json` (prod) or `console` (dev) |
| `CHROMA_SERVER_ENABLED` | `false` | Use embedded Chroma (`true` for a remote server) |

---

## Evaluation

`scripts/eval.py` runs the gold-set harness (Recall@5 / Precision@5) against the
index, optionally through the hybrid pipeline:

```bash
# Sparse-only eval (no Ollama needed)
python -m scripts.eval

# Hybrid eval (requires Ollama + nomic-embed-text)
python -m scripts.eval --hybrid
```

Week 1 baseline (BM25, 512T chunks):

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Recall@5 | >= 0.75 | 1.00 | PASS |
| Precision@5 | >= 0.70 | 0.50 | PASS |
| BM25 build time (5k chunks) | < 60s | ~2.4s | PASS |
| ChromaDB upsert integrity | Zero loss | Confirmed | PASS |

---

## Tests

```bash
# Full suite (Ollama-dependent tests are deselected by default)
python -m pytest tests/ -q

# With a local Ollama running, also exercise the live embedding tests
python -m pytest tests/ -q -m ollama

# Live end-to-end API smoke test (spawns uvicorn, requires Ollama)
python -m scripts.smoke_test

# Lint + format
python -m ruff check .
python -m ruff format --check .
```

---

## Repository Structure

```
app/                 # FastAPI service
  api/               # routes (health, query, ingest, speech), Pydantic schemas, deps
  core/              # config, logging, errors, cache
  services/          # container (DI), query service, ingest service
  agents/            # LLM client (multimodal), researcher agent
ingestion/
  parsers/           # pdf (with OCR), docx, txt, url
  chunkers/          # semantic chunker + size config
  embedders/         # Ollama embedder (retries, health), nomic facade
  ingest.py          # run_ingestion() pipeline
retrieval/
  dense/             # ChromaDB adapter
  bm25/              # BM25Okapi indexer (persisted)
  hybrid/            # RRF fusion
  reranker/          # cross-encoder
  vector_store/      # ChromaAdapter
scripts/
  seed.py            # index documents / built-in demo corpus
  eval.py            # gold-set evaluation
  smoke_test.py      # live end-to-end API verification
tests/               # 146 unit tests
Dockerfile           # production image
docker-compose.yml   # api + ollama + models-init
.github/workflows/   # CI: lint → test → docker
```
