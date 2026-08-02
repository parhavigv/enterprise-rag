from __future__ import annotations

from pathlib import Path

import docx
from llama_index.core import Document

SUPPORTED_EXTENSIONS = {".docx"}


def parse_docx(path: str) -> list[Document]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"DOCX file not found: {path}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{p.suffix}'. Expected one of {SUPPORTED_EXTENSIONS}."
        )
    try:
        doc = docx.Document(str(p))
        full_text = "\n".join(para.text for para in doc.paragraphs if para.text.strip())
        table_text = "\n".join(
            cell.text
            for table in doc.tables
            for row in table.rows
            for cell in row.cells
            if cell.text.strip()
        )
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Failed to parse DOCX '{path}': {e}") from e

    body = "\n".join(part for part in (full_text, table_text) if part.strip())
    if not body.strip():
        raise ValueError(f"DOCX '{path}' yielded no parseable content.")

    document = Document(
        text=body,
        metadata={
            "format": "docx",
            "source": str(path),
            "file_path": str(path),
        },
    )
    return [document]
