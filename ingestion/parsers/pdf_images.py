"""Best-effort OCR of images embedded in PDFs.

Uses PyMuPDF to extract page images and RapidOCR (ONNX) to read text from
them. Everything is optional and failure-tolerant: if either library is
missing, or an individual page fails, the pipeline logs and continues with
whatever text was recovered. This keeps the ingestion path robust for
scanned documents without coupling parsing to an OCR engine.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)

_MAX_IMAGE_BYTES = 15 * 1024 * 1024  # skip absurdly large embedded images
_MIN_SIDE_PX = 96  # ignore tiny icons / decorative graphics


def _get_fitz() -> Any | None:
    """Return the PyMuPDF module, or None when unavailable."""
    try:
        import fitz  # type: ignore[import-not-found]

        return fitz
    except Exception as e:  # noqa: BLE001
        logger.warning("PyMuPDF not available - skipping PDF image OCR: {}", e)
        return None


def _load_ocr() -> Callable[[bytes], Any] | None:
    """Return an ``ocr(bytearray) -> result`` callable, or None when unavailable.

    Prefers RapidOCR (pip-installable, no system deps); falls back to
    pytesseract when tesseract is on the PATH.
    """
    try:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore[import-not-found]

        engine = RapidOCR()

        def _run_rapid(image_bytes: bytes) -> Any:
            return engine(image_bytes)

        return _run_rapid
    except Exception as e:  # noqa: BLE001
        logger.warning("RapidOCR unavailable: {}", e)

    try:
        import pytesseract  # type: ignore[import-not-found]
        from PIL import Image

        def _run_tesseract(image_bytes: bytes) -> Any:
            return pytesseract.image_to_string(Image.open(io.BytesIO(image_bytes)))

        return _run_tesseract
    except Exception as e:  # noqa: BLE001
        logger.warning("pytesseract unavailable: {}", e)

    logger.warning("No OCR engine available - embedded PDF images will not be read.")
    return None


def _result_to_text(result: Any) -> str:
    """Normalise RapidOCR / pytesseract results into plain text."""
    if not result:
        return ""
    if isinstance(result, str):
        return result
    # RapidOCR returns (result, elapsed) with result = [[box, text, conf], ...].
    if isinstance(result, list | tuple) and result and isinstance(result[0], list | tuple):
        lines = []
        for entry in result[0]:
            if isinstance(entry, list | tuple) and len(entry) >= 2:
                lines.append(str(entry[1]))
            elif isinstance(entry, dict):
                lines.append(str(entry.get("text", "")))
        return "\n".join(line for line in lines if line.strip())
    return ""


def extract_images_ocr(
    pdf_path: str,
    *,
    max_images_per_page: int = 3,
    min_side_px: int = _MIN_SIDE_PX,
) -> dict[int, list[str]]:
    """Extract and OCR images per PDF page.

    Returns ``{page_number: [ocr_text, ...]}`` for pages that yielded readable
    text. Returns ``{}`` when OCR is unavailable or the PDF cannot be opened.
    """
    fitz = _get_fitz()
    ocr = _load_ocr()
    if fitz is None or ocr is None:
        return {}

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not open PDF '{}' for OCR: {}", pdf_path, e)
        return {}

    text_by_page: dict[int, list[str]] = {}
    try:
        for page_no in range(len(doc)):
            page = doc[page_no]
            page_texts: list[str] = []
            try:
                for img_info in page.get_images(full=True) or []:
                    if len(page_texts) >= max_images_per_page:
                        break
                    try:
                        xref = img_info[0]
                        base = doc.extract_image(xref)
                    except Exception as e:  # noqa: BLE001
                        logger.debug("Image extract failed on page {}: {}", page_no + 1, e)
                        continue
                    if len(base.get("image", b"")) > _MAX_IMAGE_BYTES:
                        continue
                    if base.get("width", 0) < min_side_px and base.get("height", 0) < min_side_px:
                        continue
                    try:
                        text = _result_to_text(ocr(base["image"]))
                    except Exception as e:  # noqa: BLE001
                        logger.debug("OCR failed on page {}: {}", page_no + 1, e)
                        continue
                    if text.strip():
                        page_texts.append(text.strip())
            except Exception as e:  # noqa: BLE001
                logger.warning("Page {} OCR pass failed: {}", page_no + 1, e)
            if page_texts:
                text_by_page[page_no + 1] = page_texts
    finally:
        doc.close()

    if text_by_page:
        logger.info("OCR recovered text on {} page(s) of '{}'", len(text_by_page), pdf_path)
    return text_by_page
