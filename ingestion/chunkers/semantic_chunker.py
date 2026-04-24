# ingestion/chunkers/semantic_chunker.py
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
from llama_index.core.schema import TextNode
from ingestion.chunkers.config import CHUNK_CONFIGS, DEFAULT_CHUNK_CONFIG


def chunk_documents(docs: list[Document], config: str = DEFAULT_CHUNK_CONFIG) -> list[TextNode]:
    cfg = CHUNK_CONFIGS[config]
    parser = SentenceSplitter(
        chunk_size=cfg['chunk_size'],
        chunk_overlap=cfg['chunk_overlap']
    )
    nodes = parser.get_nodes_from_documents(docs)
    print(f'Chunked {len(docs)} docs into {len(nodes)} nodes using {config} config')
    return nodes
