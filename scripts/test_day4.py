from ingestion.parsers.docx_parser import parse_docx
from ingestion.chunkers.semantic_chunker import chunk_documents
from ingestion.embedders.nomic_embedder import embed_documents
from retrieval.vector_store import ChromaAdapter

# Stage 1 - Parse
docs = parse_docx('data/raw/sample.docx')
print(f'Parsed: {len(docs)} document(s)')

# Stage 2 - Chunk
nodes = chunk_documents(docs, config='512T')
print(f'Chunked into: {len(nodes)} nodes')

# Stage 3 - Embed
texts = [n.text for n in nodes]
embeddings = embed_documents(texts)
for node, emb in zip(nodes, embeddings):
    node.embedding = emb
print(f'Embedded: {len(embeddings)} vectors (dim={len(embeddings[0])})')

# Stage 4 - Upsert into ChromaDB
adapter = ChromaAdapter(path='./chroma_data', collection='enterprise_rag')
adapter.upsert(nodes)
print(f'ChromaDB count after upsert: {adapter.count()}')

# Stage 5 - Query smoke test
query_text = 'What is the repository structure?'
query_emb = embed_documents([query_text])[0]
results = adapter.query(query_emb, top_k=3)

print('\n--- Top 3 results ---')
for i, (doc, dist) in enumerate(zip(results['documents'][0], results['distances'][0])):
    print(f'[{i+1}] distance={dist:.4f} | preview: {doc[:100]}')