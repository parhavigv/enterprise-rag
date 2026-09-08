from ingestion.parsers.docx_parser import parse_docx
from ingestion.parsers.pdf_parser import parse_pdf
from ingestion.parsers.txt_parser import parse_txt
from ingestion.parsers.url_parser import parse_url
from ingestion.parsers.xlsx_parser import parse_xlsx

PARSERS = {
    "pdf": parse_pdf,
    "docx": parse_docx,
    "txt": parse_txt,
    "url": parse_url,
    "xlsx": parse_xlsx,
}

__all__ = ["PARSERS", "parse_pdf", "parse_docx", "parse_txt", "parse_url", "parse_xlsx"]
