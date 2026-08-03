"""Runtime LLM settings - switch provider / model / API key without a restart.

The UI reads current settings via ``GET /api/v1/settings`` and applies
changes with ``PUT /api/v1/settings`` so users can plug in an OpenAI key
(or swap to a remote GPT model) for richer, explained answers.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_container
from app.api.models import SettingsUpdate
from app.core.logging import get_logger
from app.services.container import Container

logger = get_logger(__name__)

router = APIRouter(tags=["settings"])

PROVIDER_MODELS = {
    "ollama": ["llama3", "llama3.1", "qwen2.5", "mistral", "phi3"],
    "openai": ["gpt-4o", "gpt-4.1", "gpt-4o-mini"],
}

_PLACEHOLDER_KEYS = {"", "ollama", "sk-none", "sk-null"}


def _mask(api_key: str | None) -> str | None:
    """Mask a secret for display; ``None`` when no real key is configured."""
    if not api_key or api_key in _PLACEHOLDER_KEYS:
        return None
    if len(api_key) <= 6:
        return "*" * len(api_key)
    return api_key[:3] + "\u2026" + api_key[-3:]


def _settings_view(container: Container) -> dict:
    s = container.settings
    provider = getattr(s, "llm_provider", "ollama")
    if provider == "openai":
        key = getattr(s, "openai_api_key", "") or getattr(s, "llm_api_key", "")
        base_url = getattr(s, "openai_base_url", "") or getattr(s, "llm_base_url", "")
    else:
        key = getattr(s, "llm_api_key", "")
        base_url = getattr(s, "llm_base_url", "")
    return {
        "provider": provider,
        "model": getattr(s, "llm_model", "llama3.1"),
        "base_url": base_url,
        "api_key_masked": _mask(key),
        "whisper_model": getattr(s, "whisper_model", "whisper-1"),
        "providers": PROVIDER_MODELS,
    }


@router.get("/settings", summary="Current LLM settings")
def get_settings_view(container: Container = Depends(get_container)) -> dict:
    return _settings_view(container)


@router.put("/settings", summary="Update LLM settings")
def update_settings(payload: SettingsUpdate, container: Container = Depends(get_container)) -> dict:
    container.update_llm_settings(
        provider=payload.provider, model=payload.model, api_key=payload.api_key
    )
    logger.info(
        "LLM settings updated via API | provider={} | model={}",
        payload.provider or "-",
        payload.model or "-",
    )
    return _settings_view(container)
