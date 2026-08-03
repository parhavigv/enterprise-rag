"""Voice input: transcribe audio with an OpenAI-compatible Whisper endpoint.

The UI records audio in the browser (MediaRecorder) and posts it here; the
server forwards it to ``{openai_base_url}/audio/transcriptions`` using the
configured OpenAI credentials, and returns plain text.
"""

from __future__ import annotations

import time

import httpx
from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.models import TranscribeResponse
from app.core.config import Settings, get_settings
from app.core.errors import InvalidInputError, UpstreamUnavailableError
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["speech"])

MAX_AUDIO_BYTES = 25 * 1024 * 1024  # OpenAI Whisper limit


def _api_credentials(settings: Settings) -> tuple[str, str]:
    """Resolve the OpenAI-compatible base URL + API key for transcription."""
    base_url = (settings.openai_base_url or settings.llm_base_url).rstrip("/")
    api_key = settings.openai_api_key or settings.llm_api_key
    if not api_key or api_key in {"ollama", "sk-none", "sk-null"}:
        raise UpstreamUnavailableError(
            "Speech transcription needs an OpenAI API key. Set OPENAI_API_KEY "
            "(or LLM_API_KEY) and restart the service."
        )
    return base_url, api_key


@router.post("/transcribe", response_model=TranscribeResponse, summary="Transcribe voice input")
async def transcribe(
    file: UploadFile = File(..., description="Audio recording (webm/ogg/mp3/wav)"),
    model: str = Form("whisper-1", description="Whisper model"),
    settings: Settings = Depends(get_settings),
) -> TranscribeResponse:
    base_url, api_key = _api_credentials(settings)
    if not model.strip():
        raise InvalidInputError("model must not be blank")

    data = await file.read()
    if not data:
        raise InvalidInputError("Audio file is empty.")
    if len(data) > MAX_AUDIO_BYTES:
        raise InvalidInputError(
            f"Audio file too large: {len(data)} bytes (limit {MAX_AUDIO_BYTES})."
        )

    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
            resp = await client.post(
                f"{base_url}/audio/transcriptions",
                headers={"Authorization": f"Bearer {api_key}"},
                files={
                    "file": (
                        file.filename or "audio.webm",
                        data,
                        file.content_type or "audio/webm",
                    )
                },
                data={"model": model.strip()},
            )
        resp.raise_for_status()
        text = (resp.json().get("text") or "").strip()
    except httpx.HTTPStatusError as e:
        logger.error(
            "Transcription failed | status={} | body={}",
            e.response.status_code,
            e.response.text[:300],
        )
        raise UpstreamUnavailableError(
            f"Transcription upstream returned {e.response.status_code}. "
            f"Check OPENAI_API_KEY and model '{model}'."
        ) from e
    except httpx.RequestError as e:
        raise UpstreamUnavailableError(f"Could not reach transcription endpoint: {e}") from e

    if not text:
        raise UpstreamUnavailableError("Transcription returned no text.")

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info("Transcribed {} chars in {:.0f}ms", len(text), elapsed)
    return TranscribeResponse(text=text, model=model.strip(), duration_ms=elapsed)
