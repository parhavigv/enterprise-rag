from __future__ import annotations

from pathlib import Path

from llama_index.core import Document
from llama_index.readers.file import PDFReader

from ingestion.parsers.pdf_images import extract_images_ocr

SUPPORTED_EXTENSIONS = {".pdf"}

OCR_MARKER = "[OCR]"


def _merge_ocr_text(docs: list[Document], ocr_by_page: dict[int, list[str]]) -> list[Document]:
    """Append OCR-recovered text to each page document (page_number -> index)."""
    for i, doc in enumerate(docs):
        page_texts = ocr_by_page.get(i + 1)
        if not page_texts:
            continue
        recovered = " ".join(page_texts).strip()
        if not recovered:
            continue
        if doc.text:
            doc.text = f"{doc.text}\n\n{OCR_MARKER} {recovered}".strip()
        else:
            doc.text = f"{OCR_MARKER} {recovered}"
        doc.metadata["ocr"] = "true"
    return docs


def parse_pdf(path: str, *, ocr: bool = True) -> list[Document]:
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

    if ocr:
        try:
            ocr_by_page = extract_images_ocr(str(path))
            _merge_ocr_text(docs, ocr_by_page)
        except Exception as e:  # noqa: BLE001 - OCR must never break parsing
            from app.core.logging import get_logger

            get_logger(__name__).warning("OCR pass skipped for '{}': {}", path, e)

    for i, doc in enumerate(docs):
        doc.metadata["format"] = "pdf"
        doc.metadata["source"] = str(path)
        doc.metadata["page_number"] = i + 1
        doc.metadata["file_path"] = str(path)
    return docs
