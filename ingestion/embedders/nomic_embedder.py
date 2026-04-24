# ingestion/embedders/nomic_embedder.py
import httpx
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434') + '/api/embeddings'
MODEL = os.getenv('EMBED_MODEL', 'nomic-embed-text')
BATCH_SIZE = int(os.getenv('EMBED_BATCH_SIZE', 32))


async def embed_batch(texts: list[str], retries: int = 3) -> list[list[float]]:
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                tasks = [
                    client.post(OLLAMA_URL, json={'model': MODEL, 'prompt': t})
                    for t in texts
                ]
                responses = await asyncio.gather(*tasks)
                return [r.json()['embedding'] for r in responses]
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f'Embed attempt {attempt+1} failed: {e}. Retrying in {wait}s...')
            await asyncio.sleep(wait)


def embed_documents(texts: list[str]) -> list[list[float]]:
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        print(f'Embedding batch {i//BATCH_SIZE + 1} ({len(batch)} chunks)...')
        embeddings = asyncio.run(embed_batch(batch))
        all_embeddings.extend(embeddings)
    return all_embeddings
