import trafilatura
import requests
from llama_index.core import Document

def parse_url(url: str) -> list[Document]:
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        html = response.text
    except requests.RequestException as e:
        raise ValueError(f"Failed to fetch URL {url}: {e}")

    text = trafilatura.extract(html)
    if not text:
        raise ValueError(f"trafilatura could not extract content from {url}")

    document = Document(
        text=text,
        metadata={
            'format': 'url',
            'source': url,
            'url': url,
        }
    )
    return [document]
