from ingestion.parsers.docx_parser import parse_docx
from ingestion.parsers.pdf_parser import parse_pdf
from ingestion.parsers.url_parser import parse_url

PARSERS = {
    "pdf": parse_pdf,
    "docx": parse_docx,
    "url": parse_url,
}

__all__ = ["PARSERS", "parse_pdf", "parse_docx", "parse_url"]
