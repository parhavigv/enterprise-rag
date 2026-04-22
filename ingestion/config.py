from dotenv import load_dotenv
import os

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", 32))
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_data")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "enterprise_rag")
BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH", "./bm25_index.pkl")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")