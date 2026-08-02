from __future__ import annotations

from pathlib import Path

from llama_index.core import Document
from llama_index.readers.file import PDFReader

SUPPORTED_EXTENSIONS = {".pdf"}


def parse_pdf(path: str) -> list[Document]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{p.suffix}'. Expected one of {SUPPORTED_EXTENSIONS}."
        )
    try:
        reader = PDFReader()
        docs = reader.load_data(file=p)
    except Exception as e:  # noqa: BLE001 - surface parser errors with context
        raise ValueError(f"Failed to parse PDF '{path}': {e}") from e

    if not docs:
        raise ValueError(f"PDF '{path}' yielded no parseable content.")

    for i, doc in enumerate(docs):
        doc.metadata["format"] = "pdf"
        doc.metadata["source"] = str(path)
        doc.metadata["page_number"] = i + 1
        doc.metadata["file_path"] = str(path)
    return docs
