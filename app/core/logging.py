"""Structured logging via loguru.

Two formats are supported:
  - ``console``: human-readable output for local development.
  - ``json`` (default): one JSON object per line for production /
    log-aggregator pipelines (CloudWatch, Loki, Datadog, ...).
"""

from __future__ import annotations

import json
import logging
import sys

from loguru import logger


class _JsonSink:
    """Callable sink that serialises each record as a single JSON line.

    loguru treats a callable ``format=`` as a *format-string factory* whose
    return value is itself ``str.format_map``-ed, so JSON (with braces) must
    be produced in the sink instead, where we receive the full record.
    """

    def write(self, message) -> None:
        record = message.record
        extra = dict(record["extra"])
        extra["message"] = record["message"]
        extra["level"] = record["level"].name
        extra["logger"] = extra.get("logger_name") or record["name"]
        extra["timestamp"] = record["time"].isoformat()
        extra["function"] = record["function"]
        extra["line"] = record["line"]
        sys.stderr.write(json.dumps(extra, default=str) + "\n")


def setup_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Configure the root loguru logger and silence noisy third-party loggers."""
    logger.remove()
    log_level = getattr(logging, level.upper(), logging.INFO)

    if fmt == "console":
        logger.add(
            sys.stderr,
            level=log_level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{line}</cyan> - "
                "<level>{message}</level>"
            ),
            backtrace=False,
            diagnose=False,
            enqueue=True,
        )
    else:
        logger.add(_JsonSink(), level=log_level, enqueue=True)

    # Route Python stdlib loggers into loguru so we get one consistent stream.
    logging.basicConfig(handlers=[_InterceptHandler()], level=log_level, force=True)

    for noisy in ("httpx", "chromadb", "sentence_transformers", "urllib3", "trafilatura"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class _InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).bind(logger_name=record.name).log(
            level, record.getMessage()
        )


def get_logger(name: str):
    return logger.bind(logger_name=name)
