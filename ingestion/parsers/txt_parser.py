"""Plain-text parser (supports .txt, .md, .csv, .json as raw text)."""

from __future__ import annotations

from pathlib import Path

from llama_index.core import Document

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".csv", ".json", ".log"}


def parse_txt(path: str) -> list[Document]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Text file not found: {path}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{p.suffix}'. Expected one of {SUPPORTED_EXTENSIONS}."
        )
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise ValueError(f"Failed to read text file '{path}': {e}") from e

    if not text.strip():
        raise ValueError(f"Text file '{path}' is empty.")

    doc = Document(
        text=text,
        metadata={
            "format": "txt",
            "source": str(path),
            "file_path": str(path),
        },
    )
    return [doc]
