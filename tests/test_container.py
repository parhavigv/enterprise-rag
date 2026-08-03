"""Service container tests - LLM client factory + researcher overrides."""

from __future__ import annotations

from app.core.config import Settings
from app.services.container import Container


def _settings(**overrides) -> Settings:
    base = dict(
        llm_provider="ollama",
        llm_model="llama3.1",
        llm_base_url="http://localhost:11434/v1",
        llm_api_key="ollama",
        openai_base_url="https://api.openai.com/v1",
        openai_api_key="",
        llm_temperature=0.1,
        llm_max_tokens=1024,
        llm_timeout_seconds=120.0,
        llm_request_timeout_seconds=10.0,
        final_top_k=5,
    )
    base.update(overrides)
    return Settings(**base)


def test_default_llm_client_is_memoised():
    container = Container(_settings())
    assert container.llm_client() is container.llm_client()


def test_openai_override_uses_openai_endpoint_and_key():
    container = Container(_settings(openai_api_key="sk-abc"))
    client = container.llm_client(provider="openai", model="gpt-4o")
    assert client.provider == "openai"
    assert client.model == "gpt-4o"
    assert client._base_url == "https://api.openai.com/v1"
    assert client._api_key == "sk-abc"


def test_openai_override_falls_back_to_llm_key():
    container = Container(_settings(openai_api_key="", llm_api_key="sk-fallback"))
    client = container.llm_client(provider="openai", model="gpt-4o")
    assert client._api_key == "sk-fallback"


def test_ollama_default_uses_llm_base_url():
    container = Container(_settings())
    client = container.llm_client()
    assert client.provider == "ollama"
    assert client._base_url == "http://localhost:11434/v1"


def test_researcher_for_binds_provider_and_model():
    container = Container(_settings())
    researcher = container.researcher_for(provider="openai", model="gpt-4o-mini")
    assert researcher._llm.provider == "openai"
    assert researcher._llm.model == "gpt-4o-mini"
