"""OCR extraction + merge unit tests (deps are faked)."""

from __future__ import annotations

from types import SimpleNamespace

from llama_index.core import Document

from ingestion.parsers import pdf_images, pdf_parser


class _FakePage:
    def __init__(self, images, page_no: int) -> None:
        self._images = images
        self._page_no = page_no

    def get_images(self, full=True):
        return self._images


class _FakeDoc:
    def __init__(self, pages: list[_FakePage]) -> None:
        self._pages = pages
        self._closed = False

    def __len__(self) -> int:
        return len(self._pages)

    def __getitem__(self, idx: int) -> _FakePage:
        return self._pages[idx]

    def extract_image(self, xref: int):
        return {
            "image": b"PNGDATA-" + str(xref).encode(),
            "width": 400,
            "height": 200,
        }

    def close(self) -> None:
        self._closed = True


class _FakeFitz:
    @staticmethod
    def open(path: str) -> _FakeDoc:
        return _FakeDoc([_FakePage([(101,), (102,)], 1), _FakePage([], 2)])


def _fake_ocr(image_bytes: bytes):
    if b"101" in image_bytes:
        return ([[[0, 0, 10, 10], "BUSINESS ANALYST", 0.99]], 1.0)
    if b"102" in image_bytes:
        return (None, 0.2)
    return ([[[0, 0, 10, 10], "OTHER", 0.95]], 0.5)


def test_extract_images_ocr_skipped_when_pymupdf_missing(monkeypatch):
    monkeypatch.setattr(pdf_images, "_get_fitz", lambda: None)
    monkeypatch.setattr(pdf_images, "_load_ocr", lambda: _fake_ocr)
    assert pdf_images.extract_images_ocr("x.pdf") == {}


def test_extract_images_ocr_skipped_when_engine_missing(monkeypatch):
    monkeypatch.setattr(pdf_images, "_get_fitz", _FakeFitz)
    monkeypatch.setattr(pdf_images, "_load_ocr", lambda: None)
    assert pdf_images.extract_images_ocr("x.pdf") == {}


def test_extract_images_ocr_returns_per_page_text(monkeypatch):
    monkeypatch.setattr(pdf_images, "_get_fitz", _FakeFitz)
    monkeypatch.setattr(pdf_images, "_load_ocr", lambda: _fake_ocr)
    out = pdf_images.extract_images_ocr("guide.pdf")
    assert out == {1: ["BUSINESS ANALYST"]}


def test_extract_images_ocr_closes_document(monkeypatch):
    opened = []

    def _open(path):
        doc = _FakeDoc([_FakePage([], 1)])
        opened.append(doc)
        return doc

    monkeypatch.setattr(pdf_images, "_get_fitz", lambda: SimpleNamespace(open=_open))
    monkeypatch.setattr(pdf_images, "_load_ocr", lambda: _fake_ocr)
    pdf_images.extract_images_ocr("guide.pdf")
    assert opened[0]._closed is True


def test_result_to_text_handles_rapidocr_and_plain_string():
    assert pdf_images._result_to_text(([[[0, 0, 1, 1], "HELLO", 0.9]], 0.3)) == "HELLO"
    assert pdf_images._result_to_text("plain text") == "plain text"
    assert pdf_images._result_to_text(None) == ""
    assert pdf_images._result_to_text((None, 0.3)) == ""


def test_merge_ocr_text_appends_and_flags():
    docs = [
        Document(text="typed text", metadata={"page_number": 1}),
        Document(text="", metadata={}),
    ]
    pdf_parser._merge_ocr_text(docs, {1: ["RECOVERED WORDS"]})
    assert "typed text" in docs[0].text
    assert "[OCR] RECOVERED WORDS" in docs[0].text
    assert docs[0].metadata["ocr"] == "true"
    assert docs[1].metadata.get("ocr") is None


def test_merge_ocr_text_skips_empty_recovery():
    docs = [Document(text="only", metadata={})]
    pdf_parser._merge_ocr_text(docs, {1: ["   "]})
    assert docs[0].text == "only"
