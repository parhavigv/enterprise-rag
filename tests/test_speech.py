"""Voice transcription endpoint tests (upstream HTTP is faked)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.api.routes.speech as speech_mod
from app.core.config import Settings, get_settings
from app.main import create_app


def _settings(**overrides) -> Settings:
    base = dict(
        api_prefix="/api/v1",
        request_id_header="X-Request-ID",
        cors_origins="*",
        environment="test",
        app_name="enterprise-rag",
        debug=False,
        log_level="WARNING",
        log_format="console",
        llm_provider="ollama",
        llm_model="llama3.1",
        llm_base_url="http://localhost:11434/v1",
        llm_api_key="ollama",
        openai_base_url="https://api.openai.com/v1",
        openai_api_key="sk-test-123",
        whisper_model="whisper-1",
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def client():
    app = create_app(settings=_settings())
    app.dependency_overrides[get_settings] = lambda: _settings()
    with TestClient(app) as c:
        yield c


class _FakeResponse:
    def __init__(self, payload=None, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = "ok"

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError(
                "err", request=httpx.Request("POST", "http://x"), response=self
            )

    def json(self):
        return self._payload


class _FakeAsyncClient:
    calls: list[tuple[str, dict]] = []

    def __init__(self, **kwargs) -> None:
        self._kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, **kwargs):
        _FakeAsyncClient.calls.append((url, kwargs))
        return _FakeResponse(payload={"text": "hello from voice"})


def test_transcribe_happy_path(monkeypatch, client):
    _FakeAsyncClient.calls.clear()
    monkeypatch.setattr(speech_mod.httpx, "AsyncClient", _FakeAsyncClient)

    r = client.post(
        "/api/v1/transcribe",
        files={"file": ("note.webm", b"\x1a\x45\xdf\xa3fakeaudio", "audio/webm")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "hello from voice"
    assert body["model"] == "whisper-1"

    url, kwargs = _FakeAsyncClient.calls[0]
    assert url == "https://api.openai.com/v1/audio/transcriptions"
    assert kwargs["headers"]["Authorization"] == "Bearer sk-test-123"
    assert kwargs["data"] == {"model": "whisper-1"}


def test_transcribe_rejects_empty_file(client):
    r = client.post("/api/v1/transcribe", files={"file": ("e.webm", b"", "audio/webm")})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_input"


def test_transcribe_missing_key_503(client):
    client.app.dependency_overrides[get_settings] = lambda: _settings(
        openai_api_key="", llm_api_key="ollama"
    )
    r = client.post(
        "/api/v1/transcribe",
        files={"file": ("a.webm", b"\x1a\x45\xdf\xa3", "audio/webm")},
    )
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "upstream_unavailable"
