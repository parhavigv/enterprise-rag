# syntax=docker/dockerfile:1
FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NLTK_DISABLE_IMPORT_SECURITY=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv/app

# System deps (minimal): tini for signal handling + curl for healthchecks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tini curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Pre-fetch nltk punkt (used by llama_index tokenisers) so first request
# at runtime never hits the network.
RUN python - <<'PY'
import os
os.environ["NLTK_DISABLE_IMPORT_SECURITY"] = "1"
try:
    import nltk
    nltk.download("punkt_tab", quiet=True)
    nltk.download("punkt", quiet=True)
except Exception as exc:
    print(f"nltk pre-download skipped: {exc}")
PY

COPY . .

RUN mkdir -p /srv/app/data && \
    useradd --create-home --uid 10001 appuser && \
    chown -R appuser:appuser /srv/app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
