"""FastAPI integration tests - services are replaced with fakes."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_container, get_ingest_service, get_query_service
from app.main import create_app
from app.services.query_service import QueryResult


class _HealthyProbe:
    def health(self) -> bool:
        return True


class _EmptyBm25:
    def count(self) -> int:
        return 10


class _FakeContainer:
    def __init__(self) -> None:
        self.query_service_obj = _FakeQueryService()
        self.ingest_service_obj = _FakeIngestService()
        self.settings = SimpleNamespace(
            uploads_dir="./data/uploads",
            chroma_collection="enterprise_rag",
        )

    def adapter(self):
        return _HealthyProbe()

    def embedder(self):
        return _HealthyProbe()

    def bm25_indexer(self):
        return _EmptyBm25()

    def query_service(self):
        return self.query_service_obj

    def ingest_service(self):
        return self.ingest_service_obj

    def researcher_for(self, provider=None, model=None):
        return SimpleNamespace(provider=provider, model=model)

    def stats(self):
        return {"chroma_count": 10, "bm25_count": 10}

    def refresh_index(self):
        return None


class _FakeQueryService:
    def __init__(self) -> None:
        self.last_answer_kwargs: dict | None = None

    async def answer(self, query, top_k=5, rerank=True, generate=True, **kwargs):
        self.last_answer_kwargs = kwargs
        if query == "boom":
            from app.core.errors import UpstreamUnavailableError

            raise UpstreamUnavailableError("LLM down")
        return QueryResult(
            query=query,
            search=[
                {"node_id": "a", "text": "t", "score": 0.9, "metadata": {}, "source": "hybrid"}
            ],
            answer="Grounded answer.",
            model="llama3.1",
            generated=generate,
            latency_ms=1.5,
        )

    def search(self, query, top_k=5, rerank=True):
        return QueryResult(
            query=query,
            search=[
                {"node_id": "a", "text": "t", "score": 0.9, "metadata": {}, "source": "hybrid"}
            ],
            latency_ms=0.5,
        )


class _FakeIngestService:
    def ingest(self, **kwargs):
        return {"chunks": 12, "chroma_count": 12, "bm25_count": 12, "total_seconds": 0.1}

    def ingest_background(self, **kwargs):
        def _run():
            return None

        return _run

    @property
    def last_run(self):
        return {"metrics": {"chunks": 12}, "error": None}


@pytest.fixture()
def client():
    app = create_app(
        settings=SimpleNamespace(
            api_prefix="/api/v1",
            request_id_header="X-Request-ID",
            cors_origins="*",
            cors_origin_list=["*"],
            environment="test",
            app_name="enterprise-rag",
            debug=False,
            log_level="WARNING",
            log_format="console",
            chroma_collection="enterprise_rag",
            bm25_index_path="./data/bm25_index.pkl",
            llm_model="llama3.1",
            embed_model="nomic-embed-text",
        )
    )
    fake = _FakeContainer()
    app.dependency_overrides[get_container] = lambda: fake
    app.dependency_overrides[get_query_service] = lambda: fake.query_service_obj
    app.dependency_overrides[get_ingest_service] = lambda: fake.ingest_service_obj
    with TestClient(app) as c:
        c.app.state.fake_query_service = fake.query_service_obj
        yield c


def test_liveness(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_readiness(client):
    r = client.get("/health/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in {"ok", "degraded"}
    assert body["checks"]["bm25"] is True


def test_query_endpoint(client):
    r = client.post("/api/v1/query", json={"query": "what is RAG?", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Grounded answer."
    assert body["generated"] is True
    assert len(body["search"]) == 1


def test_search_endpoint(client):
    r = client.post("/api/v1/search", json={"query": "find docs", "generate": False})
    assert r.status_code == 200
    assert r.json()["answer"] is None
    assert r.json()["search"]


def test_ingest_endpoint(client):
    r = client.post("/api/v1/ingest", json={"path": "http://example.com", "format": "url"})
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["metrics"]["chunks"] == 12


def test_ingest_background(client):
    r = client.post("/api/v1/ingest", json={"path": "x.pdf", "format": "pdf", "background": True})
    assert r.status_code == 200
    assert r.json()["background"] is True


def test_ingest_invalid_format_422(client):
    r = client.post("/api/v1/ingest", json={"path": "x", "format": "exe"})
    assert r.status_code == 422


def test_query_blank_query_422(client):
    r = client.post("/api/v1/query", json={"query": "   "})
    assert r.status_code == 422


def test_query_with_model_override(client):
    r = client.post(
        "/api/v1/query",
        json={"query": "what is RAG?", "provider": "openai", "model": "gpt-4o"},
    )
    assert r.status_code == 200
    assert client.app.state.fake_query_service.last_answer_kwargs["provider"] == "openai"
    assert client.app.state.fake_query_service.last_answer_kwargs["model"] == "gpt-4o"
    assert client.app.state.fake_query_service.last_answer_kwargs["researcher"] is not None


def test_query_with_images_passes_through(client):
    img = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="
    r = client.post("/api/v1/query", json={"query": "what is in the image?", "images": [img]})
    assert r.status_code == 200
    assert client.app.state.fake_query_service.last_answer_kwargs["images"] == [img]


def test_query_invalid_provider_422(client):
    r = client.post("/api/v1/query", json={"query": "q", "provider": "gemini"})
    assert r.status_code == 422


def test_query_oversized_image_422(client):
    r = client.post("/api/v1/query", json={"query": "q", "images": ["x" * (14 * 1024 * 1024 + 1)]})
    assert r.status_code == 422


def test_query_error_returns_problem_json(client):
    r = client.post("/api/v1/query", json={"query": "boom"})
    assert r.status_code == 503
    body = r.json()
    assert body["error"]["code"] == "upstream_unavailable"


def test_unknown_route_404(client):
    assert client.get("/nope").status_code == 404


def test_request_id_header_set(client):
    r = client.get("/health", headers={"X-Request-ID": "abc123"})
    assert r.headers.get("X-Request-ID") == "abc123"


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "enterprise-rag"


def test_ui_page(client):
    r = client.get("/ui")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Upload" in r.text and "/api/v1/query" in r.text


def test_upload_txt(client):
    r = client.post(
        "/api/v1/upload",
        files={
            "file": ("notes.txt", b"Hybrid retrieval fuses dense and sparse scores.", "text/plain")
        },
        data={"chunk_size": "512T"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["metrics"]["chunks"] == 12
    assert body["metrics"]["filename"] == "notes.txt"


def test_upload_rejects_unknown_type(client):
    r = client.post(
        "/api/v1/upload",
        files={"file": ("virus.exe", b"MZ...", "application/octet-stream")},
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_input"


def test_upload_rejects_empty_file(client):
    r = client.post(
        "/api/v1/upload",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_input"
