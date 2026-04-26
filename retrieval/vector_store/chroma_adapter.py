import chromadb
from chromadb.config import Settings
from llama_index.core.schema import TextNode
from typing import List


class ChromaAdapter:
    def __init__(
        self,
        path: str = './chroma_data',
        collection: str = 'enterprise_rag'
    ):
        self.client = chromadb.PersistentClient(
            path=path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.col = self.client.get_or_create_collection(
            name=collection,
            metadata={'hnsw:space': 'cosine'}
        )
        print(f'ChromaAdapter ready -> collection: {collection} | path: {path}')

    def upsert(self, nodes: List[TextNode]) -> None:
        ids        = [n.node_id for n in nodes]
        embeddings = [n.embedding for n in nodes]
        documents  = [n.text for n in nodes]
        metadatas  = [n.metadata for n in nodes]
        self.col.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        print(f'Upserted {len(nodes)} nodes -> collection now has {self.col.count()} entries')

    def query(
        self,
        embedding: List[float],
        top_k: int = 20
    ) -> dict:
        return self.col.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )

    def count(self) -> int:
        return self.col.count()

    def reset(self) -> None:
        self.client.delete_collection(self.col.name)
        self.col = self.client.get_or_create_collection(
            name=self.col.name,
            metadata={'hnsw:space': 'cosine'}
        )
        print('Collection reset - all embeddings cleared')