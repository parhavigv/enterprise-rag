"""Pydantic request/response schemas for the HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------- #
# Query
# --------------------------------------------------------------------- #
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000, description="Search query")
    top_k: int = Field(5, ge=1, le=20, description="Number of final results")
    rerank: bool = Field(True, description="Apply cross-encoder re-ranking")
    generate: bool = Field(True, description="Generate a grounded LLM answer")
    provider: str | None = Field(None, description="LLM provider override: ollama | openai")
    model: str | None = Field(None, description="Model override, e.g. gpt-4o or llama3")
    images: list[str] | None = Field(
        None,
        max_length=4,
        description="Base64 images (or data URIs) attached to the LLM call for vision chat",
    )

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v.strip()

    @field_validator("provider")
    @classmethod
    def _validate_provider(cls, v: str | None) -> str | None:
        if v is not None and v not in {"ollama", "openai"}:
            raise ValueError("provider must be one of: ollama, openai")
        return v

    @field_validator("images")
    @classmethod
    def _validate_images(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        cleaned: list[str] = []
        for image in v:
            if not image or not image.strip():
                raise ValueError("images must not contain empty entries")
            if len(image) > 14 * 1024 * 1024:  # ~10 MB raw binary in base64
                raise ValueError("each image must be at most ~10 MB")
            cleaned.append(image.strip())
        return cleaned or None


class SourceModel(BaseModel):
    node_id: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: str = "hybrid"


class QueryResponse(BaseModel):
    query: str
    search: list[SourceModel] = Field(default_factory=list)
    answer: str | None = None
    model: str | None = None
    generated: bool = False
    latency_ms: float = 0.0
    error: str | None = None
    cache_hit: bool = False


# --------------------------------------------------------------------- #
# Ingest
# --------------------------------------------------------------------- #
class IngestRequest(BaseModel):
    path: str = Field(..., min_length=1, max_length=2048, description="File path or URL")
    format: str = Field(..., description="One of: pdf, docx, url")
    chunk_size: str = Field("512T", description="One of: 256T, 512T, 1024T")
    collection_name: str = Field("enterprise_rag", min_length=1, max_length=128)
    rebuild_bm25: bool = Field(False, description="Force BM25 index rebuild")
    background: bool = Field(False, description="Run in a background task")

    @field_validator("format")
    @classmethod
    def _validate_format(cls, v: str) -> str:
        if v not in {"pdf", "docx", "url"}:
            raise ValueError("format must be one of: pdf, docx, url")
        return v

    @field_validator("chunk_size")
    @classmethod
    def _validate_chunk_size(cls, v: str) -> str:
        if v not in {"256T", "512T", "1024T"}:
            raise ValueError("chunk_size must be one of: 256T, 512T, 1024T")
        return v


class IngestResponse(BaseModel):
    status: str = "started"
    background: bool = False
    metrics: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------- #
# Health / stats
# --------------------------------------------------------------------- #
class HealthCheck(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    checks: dict[str, Any] = Field(default_factory=dict)


class StatsResponse(BaseModel):
    chroma_collection: str
    chroma_count: int
    bm25_index_path: str
    bm25_count: int
    embed_model: str
    llm_model: str


class ErrorModel(BaseModel):
    error: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------- #
# Speech
# --------------------------------------------------------------------- #
class TranscribeResponse(BaseModel):
    text: str = Field("", description="Transcribed text from the uploaded audio")
    model: str = Field("whisper-1", description="Transcription model used")
    duration_ms: float = Field(0.0, description="Round-trip transcription latency")
