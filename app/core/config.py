"""Centralised, type-safe application configuration.

All runtime behaviour is driven from environment variables so the same
codebase runs locally, in CI, and in containerised production deployments
without modification.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# nltk ships an import-security hook that blocks dependency imports that
# resolve inside the working directory (e.g. when a venv lives inside the
# repo). The library documents NLTK_DISABLE_IMPORT_SECURITY as the escape
# hatch; we set it *before* any llama_index / nltk import happens.
os.environ.setdefault("NLTK_DISABLE_IMPORT_SECURITY", "1")

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- App ---
    app_name: str = "enterprise-rag"
    environment: str = "production"
    debug: bool = False
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    log_level: str = "INFO"
    log_format: str = "json"  # "json" | "console"
    request_id_header: str = "X-Request-ID"
    cors_origins: str = "*"

    # --- Embedding (Ollama) ---
    ollama_base_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"
    embed_batch_size: int = 32
    embed_timeout_seconds: float = 60.0
    embed_max_retries: int = 3

    # --- LLM (answer generation) ---
    llm_provider: str = "ollama"  # "ollama" | "openai"
    llm_model: str = "llama3.1"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    llm_timeout_seconds: float = 120.0
    llm_request_timeout_seconds: float = 10.0

    # --- Vector store ---
    chroma_path: str = "./data/chroma"
    chroma_collection: str = "enterprise_rag"
    chroma_server_enabled: bool = False
    chroma_server_host: str = "chroma"
    chroma_server_port: int = 8000

    # --- Sparse index ---
    bm25_index_path: str = "./data/bm25_index.pkl"

    # --- Retrieval ---
    dense_top_k: int = 50
    sparse_top_k: int = 50
    hybrid_top_k: int = 20
    final_top_k: int = 5
    rrf_k: int = 60

    # --- Re-ranking ---
    reranker_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_max_docs: int = 50

    # --- Caching ---
    cache_enabled: bool = True
    cache_ttl_seconds: int = 600
    cache_max_entries: int = 512

    # --- Ingestion ---
    default_chunk_size: str = "512T"
    allowed_formats: str = "pdf,docx,url"

    # --- Persistence for local dev ---
    data_dir: str = "./data"

    @field_validator("allowed_formats")
    @classmethod
    def _split_formats(cls, v: str) -> str:
        return ",".join(f.strip() for f in v.split(",") if f.strip())

    @property
    def allowed_formats_list(self) -> list[str]:
        return [f.strip() for f in self.allowed_formats.split(",")]

    @property
    def cors_origin_list(self) -> list[str]:
        origins = [o.strip() for o in self.cors_origins.split(",") if o.strip()]
        return origins if origins else ["*"]

    @property
    def project_root(self) -> Path:
        return DEFAULT_PROJECT_ROOT


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def get_project_root() -> Path:
    return get_settings().project_root
