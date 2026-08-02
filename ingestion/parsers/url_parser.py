from __future__ import annotations

import trafilatura
from httpx import Client, HTTPStatusError, RequestError, Response
from llama_index.core import Document

DEFAULT_TIMEOUT = 15.0
MAX_BYTES = 10 * 1024 * 1024  # 10 MB guard against pathological pages
_USER_AGENT = (
    "Mozilla/5.0 (compatible; EnterpriseRAG/1.0; +https://github.com/parhavigv/enterprise-rag)"
)


def _fetch(url: str) -> str:
    headers = {"User-Agent": _USER_AGENT}
    with Client(follow_redirects=True, timeout=DEFAULT_TIMEOUT, headers=headers) as client:
        resp: Response = client.get(url)
    try:
        resp.raise_for_status()
    except HTTPStatusError as e:
        raise ValueError(f"Failed to fetch URL {url}: HTTP {resp.status_code}") from e
    return resp.text


def parse_url(url: str) -> list[Document]:
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"URL must start with http:// or https:// - got '{url[:40]}'")
    try:
        html = _fetch(url)
    except RequestError as e:
        raise ValueError(f"Failed to fetch URL {url}: {e}") from e

    if len(html) > MAX_BYTES:
        raise ValueError(f"URL {url} exceeds the {MAX_BYTES} byte size guard.")

    text = trafilatura.extract(html, include_links=False, include_tables=True)
    if not text or not text.strip():
        raise ValueError(f"trafilatura could not extract content from {url}")

    document = Document(
        text=text.strip(),
        metadata={
            "format": "url",
            "source": url,
            "url": url,
        },
    )
    return [document]
