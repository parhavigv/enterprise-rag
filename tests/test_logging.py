"""Regression tests for the JSON log sink.

loguru 0.7+ treats a callable ``format=`` as a format-string factory and
re-applies ``str.format_map`` to whatever it returns, which breaks JSON
(braces). Serialisation must therefore happen in a callable sink instead.
"""

from __future__ import annotations

import json
import logging
import time


def _read_json_lines(err_text: str) -> list[dict]:
    return [json.loads(line) for line in err_text.splitlines() if line.startswith("{")]


def test_json_sink_outputs_valid_records(capsys):
    from app.core.logging import get_logger, setup_logging

    setup_logging(level="DEBUG", fmt="json")
    get_logger("my.app").info("hello {}", 42)
    # Records routed from stdlib must not crash the sink either.
    logging.getLogger("uvicorn.access").info("stdlib routed")

    records: list[dict] = []
    deadline = time.time() + 5
    while not records and time.time() < deadline:
        records = _read_json_lines(capsys.readouterr().err)
        time.sleep(0.05)
    assert records, "expected JSON log lines"

    mine = next(r for r in records if r.get("message") == "hello 42")
    assert mine["logger"] == "my.app"
    assert mine["level"] == "INFO"
    assert mine["timestamp"]

    stdlib = next(r for r in records if r.get("message") == "stdlib routed")
    assert stdlib["logger"] == "uvicorn.access"


def test_json_sink_no_format_map_error(capsys):
    """The KeyError: '\"logger_name\"' from format_map must not recur."""
    from app.core.logging import get_logger, setup_logging

    setup_logging(level="DEBUG", fmt="json")
    get_logger("boom.app").info("should be fine")

    err = capsys.readouterr().err
    time.sleep(0.2)  # allow the enqueue thread to flush
    err += capsys.readouterr().err
    assert "Logging error in Loguru Handler" not in err
    assert "KeyError" not in err
