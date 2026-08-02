"""Embedding provider backed by Ollama's ``/api/embeddings`` endpoint.

Production behaviours included:
  - configurable batch size and request timeout,
  - retries with exponential backoff + jitter,
  - validation of embedding dimensionality,
  - a ``health()`` probe used by the readiness endpoint.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable

import httpx

from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbedderError(RuntimeError):
    pass


class OllamaEmbedder:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        batch_size: int = 32,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._batch_size = max(1, batch_size)
        self._timeout = timeout
        self._max_retries = max_retries
        self._dim: int | None = None

    @property
    def model(self) -> str:
        return self._model

    @property
    def dim(self) -> int | None:
        """Embedding dimensionality, cached from the first successful call."""
        return getattr(self, "_dim", None)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            embeddings.extend(self._embed_sync(batch))
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Query text must not be empty.")
        return self._embed_sync([text])[0]

    def health(self) -> bool:
        try:
            with httpx.Client(timeout=5.0) as client:
                r = client.get(f"{self._base_url}/api/tags")
            r.raise_for_status()
            models = {m.get("name", "").split(":")[0] for m in r.json().get("models", [])}
            return self._model.split(":")[0] in models
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # Implementation
    # ------------------------------------------------------------------ #
    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                return self._run_async(self._embed_async(texts))
            except Exception as e:  # noqa: BLE001 - deliberate catch-all for retries
                last_error = e
                if attempt == self._max_retries - 1:
                    break
                sleep = (2**attempt) + random.uniform(0, 0.5)
                logger.warning(
                    "Embedding attempt {}/{} failed: {} - retrying in {:.1f}s",
                    attempt + 1,
                    self._max_retries,
                    e,
                    sleep,
                )
                time.sleep(sleep)
        raise EmbedderError(
            f"Embedding failed after {self._max_retries} attempts: {last_error}"
        ) from last_error

    @staticmethod
    def _run_async(coro: Awaitable[list[list[float]]]) -> list[list[float]]:
        """Run a coroutine from sync code, safe inside a running event loop.

        FastAPI executes request handlers on its event loop, where
        ``asyncio.run()`` raises ``RuntimeError``. When a loop is already
        running we spin up a dedicated worker thread that owns its own loop.
        """
        import threading

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        result: list[list[float]] = []
        error: BaseException | None = None

        def _worker() -> None:
            nonlocal result, error
            try:
                result = asyncio.run(coro)
            except BaseException as exc:  # noqa: BLE001
                error = exc

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        thread.join()
        if error is not None:
            raise error
        return result

    async def _embed_async(self, texts: list[str]) -> list[list[float]]:
        url = f"{self._base_url}/api/embeddings"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            tasks = [client.post(url, json={"model": self._model, "prompt": t}) for t in texts]
            responses = await asyncio.gather(*tasks)

        embeddings: list[list[float]] = []
        for r in responses:
            if r.status_code != 200:
                raise EmbedderError(f"Ollama returned HTTP {r.status_code}: {r.text[:200]}")
            body = r.json()
            emb = body.get("embedding")
            if not emb:
                raise EmbedderError(f"Ollama response missing embedding: {body}")
            embeddings.append(emb)

        dims = {len(e) for e in embeddings}
        if len(dims) > 1:
            raise EmbedderError(f"Inconsistent embedding dimensions: {dims}")
        if self._dim is None:
            self._dim = dims.pop()
        return embeddings

    # ------------------------------------------------------------------ #
    # Backward-compatible module-style helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def from_env(settings) -> OllamaEmbedder:
        return OllamaEmbedder(
            base_url=settings.ollama_base_url,
            model=settings.embed_model,
            batch_size=settings.embed_batch_size,
            timeout=settings.embed_timeout_seconds,
            max_retries=settings.embed_max_retries,
        )
