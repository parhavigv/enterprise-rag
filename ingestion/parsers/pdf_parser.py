from pathlib import Path
from llama_index.core import Document
from llama_index.readers.file import PDFReader

def parse_pdf(path: str) -> list[Document]:
    reader = PDFReader()
    docs = reader.load_data(file=Path(path))
    for i, doc in enumerate(docs):
        doc.metadata['format'] = 'pdf'
        doc.metadata['source'] = str(path)
        doc.metadata['page_number'] = i + 1
    return docs
