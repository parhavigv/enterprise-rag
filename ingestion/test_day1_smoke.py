"""
Day 1 smoke tests — all must pass before proceeding to Day 2.
Run: pytest tests/test_day1_smoke.py -v
"""
import importlib
import httpx
import pytest


def test_python_version():
    import sys
    assert sys.version_info >= (3, 10), "Python 3.10+ required"


def test_core_imports():
    modules = [
        "llama_index.core",
        "chromadb",
        "rank_bm25",
        "sentence_transformers",
        "dotenv",
        "docx",
        "trafilatura",
        "bs4",
        "httpx",
        "loguru",
    ]
    for mod in modules:
        assert importlib.util.find_spec(mod) is not None, f"Missing: {mod}"


def test_env_config():
    from ingestion.config import OLLAMA_BASE_URL, EMBED_MODEL, CHROMA_PATH
    assert OLLAMA_BASE_URL.startswith("http"), "OLLAMA_BASE_URL malformed"
    assert EMBED_MODEL == "nomic-embed-text", "Wrong embed model"
    assert CHROMA_PATH is not None


def test_ollama_reachable():
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=5)
        assert r.status_code == 200
    except Exception as e:
        pytest.fail(f"Ollama unreachable: {e}")


def test_nomic_model_available():
    r = httpx.get("http://localhost:11434/api/tags", timeout=5)
    models = [m["name"] for m in r.json().get("models", [])]
    assert any("nomic-embed-text" in m for m in models), \
        f"nomic-embed-text not found. Found: {models}"


def test_embedding_end_to_end():
    r = httpx.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": "hello world smoke test"},
        timeout=30,
    )
    assert r.status_code == 200
    embedding = r.json().get("embedding", [])
    assert len(embedding) > 100, f"Embedding too short: {len(embedding)}"
    print(f"\n  Embedding dim: {len(embedding)}")


def test_chromadb_init():
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(
        path="./chroma_data",
        settings=Settings(anonymized_telemetry=False)
    )
    col = client.get_or_create_collection("smoke_test")
    assert col is not None
    client.delete_collection("smoke_test")


def test_directory_structure():
    from pathlib import Path
    required = [
        "ingestion/parsers", "ingestion/chunkers", "ingestion/embedders",
        "retrieval/vector_store", "retrieval/bm25", "retrieval/reranker",
        "tests/eval", "notebooks", "data/raw", "data/processed",
    ]
    for d in required:
        assert Path(d).is_dir(), f"Missing directory: {d}"