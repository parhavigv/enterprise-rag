import logging
from typing import Optional
from ingestion.parsers.pdf_parser import parse_pdf
from ingestion.parsers.docx_parser import parse_docx
from ingestion.parsers.url_parser import parse_url
from ingestion.chunkers.text_chunker import chunk_text
from ingestion.embedders.ollama_embedder import embed_chunks
from ingestion.store import save_chunks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ingest_file(path: str) -> int:
    """
    Full pipeline for a local file (PDF or DOCX).

    Args:
        path: absolute or relative path to the file

    Returns:
        number of chunks stored
    """
    path_lower = path.lower()

    if path_lower.endswith(".pdf"):
        logger.info(f"Parsing PDF: {path}")
        text = parse_pdf(path)
        doc_type = "pdf"

    elif path_lower.endswith(".docx"):
        logger.info(f"Parsing DOCX: {path}")
        text = parse_docx(path)
        doc_type = "docx"

    else:
        raise ValueError(f"Unsupported file type: {path}")

    if not text or not text.strip():
        logger.warning(f"No text extracted from {path}")
        return 0

    chunks = chunk_text(text, source=path, doc_type=doc_type)
    logger.info(f"Created {len(chunks)} chunks from {path}")

    chunks = embed_chunks(chunks)
    logger.info(f"Embedded {len(chunks)} chunks")

    saved = save_chunks(chunks)
    logger.info(f"Stored {saved} chunks in ChromaDB")

    return saved


def ingest_url(url: str) -> int:
    """
    Full pipeline for a web URL.

    Args:
        url: fully qualified URL to scrape

    Returns:
        number of chunks stored
    """
    logger.info(f"Parsing URL: {url}")
    text = parse_url(url)

    if not text or not text.strip():
        logger.warning(f"No text extracted from {url}")
        return 0

    chunks = chunk_text(text, source=url, doc_type="url")
    logger.info(f"Created {len(chunks)} chunks from {url}")

    chunks = embed_chunks(chunks)
    logger.info(f"Embedded {len(chunks)} chunks")

    saved = save_chunks(chunks)
    logger.info(f"Stored {saved} chunks in ChromaDB")

    return saved