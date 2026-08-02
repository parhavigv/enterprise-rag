"""Async LLM client supporting Ollama and OpenAI-compatible endpoints.

Both backends speak the same OpenAI-shaped ``/chat/completions`` JSON schema
in modern versions:
  - Ollama exposes ``{base_url}/v1/chat/completions``,
  - OpenAI exposes ``{base_url}/chat/completions``.

A small request-timeout heuristic keeps the API responsive when the LLM is
slow or unreachable, without hard-coding the total generation timeout.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import httpx

from app.core.logging import get_logger

logger = get_logger(__name__)

_LLM_TIMEOUT_HINT = 6.0  # seconds before a *large* model may stream its first token


@dataclass(slots=True)
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    raw: dict = field(default_factory=dict)


class LLMUnavailableError(RuntimeError):
    pass


class AsyncLLMClient:
    """A minimal, retrying chat-completion client."""

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "llama3.1",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout: float = 120.0,
        connect_timeout: float = 10.0,
    ) -> None:
        if provider not in {"ollama", "openai"}:
            raise ValueError("llm_provider must be 'ollama' or 'openai'")
        self._provider = provider
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._connect_timeout = connect_timeout

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    def _chat_url(self) -> str:
        if self._provider == "ollama":
            return f"{self._base_url}/chat/completions"
        return f"{self._base_url}/chat/completions"

    async def complete(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        if not messages or not messages[-1].get("content", "").strip():
            raise ValueError("At least one non-empty message is required.")

        payload: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature if temperature is None else temperature,
            "max_tokens": self._max_tokens if max_tokens is None else max_tokens,
            "stream": False,
        }

        t0 = time.perf_counter()
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(self._timeout, connect=self._connect_timeout)
                ) as client:
                    headers = {"Content-Type": "application/json"}
                    if self._api_key and self._provider == "openai":
                        headers["Authorization"] = f"Bearer {self._api_key}"
                    resp = await client.post(self._chat_url(), json=payload, headers=headers)
                if resp.status_code >= 500 or resp.status_code == 429:
                    resp.raise_for_status()
                resp.raise_for_status()
                data = resp.json()

                content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
                if content is None:
                    raise LLMUnavailableError(f"LLM returned no content. Body: {str(data)[:300]}")

                usage = data.get("usage") or {}
                latency_ms = (time.perf_counter() - t0) * 1000
                return LLMResponse(
                    text=content.strip(),
                    model=data.get("model", self._model),
                    prompt_tokens=int(usage.get("prompt_tokens", 0)),
                    completion_tokens=int(usage.get("completion_tokens", 0)),
                    total_tokens=int(usage.get("total_tokens", 0)),
                    latency_ms=latency_ms,
                    raw=data,
                )
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                last_error = e
                if attempt < 2:
                    await asyncio_sleep(2**attempt)
        raise LLMUnavailableError(f"LLM call failed after 3 attempts: {last_error}") from last_error

    async def health(self) -> bool:
        try:
            root = self._base_url
            if self._provider == "ollama":
                # Ollama exposes /api/tags on the server root (not under /v1).
                root = root.replace("/v1", "").rstrip("/")
                probe = root + "/api/tags"
            else:
                probe = self._base_url.rstrip("/") + "/models"
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(probe)
            resp.raise_for_status()
            return True
        except Exception:  # noqa: BLE001
            return False


async def asyncio_sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)
