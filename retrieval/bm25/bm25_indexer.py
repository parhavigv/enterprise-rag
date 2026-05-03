from __future__ import annotations

import hashlib
import logging
import os
import pickle
import re
import tempfile
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

DEFAULT_INDEX_PATH = Path(os.getenv("BM25_INDEX_PATH", "./bm25_index.pkl"))
_PUNCTUATION_RE = re.compile(r"[^\w\s]")


def _default_tokeniser(text: str) -> List[str]:
    text = text.lower()
    text = _PUNCTUATION_RE.sub(" ", text)
    return text.split()


class _BM25Payload:
    __slots__ = ("index", "doc_ids", "corpus_hash", "created_at")

    def __init__(self, index, doc_ids, corpus_hash):
        self.index = index
        self.doc_ids = doc_ids
        self.corpus_hash = corpus_hash
        self.created_at = time.time()


class BM25Indexer:
    def __init__(self, index_path=DEFAULT_INDEX_PATH, tokeniser=None):
        self._index_path = Path(index_path)
        self._tokeniser = tokeniser or _default_tokeniser
        self._payload: Optional[_BM25Payload] = None

    def build(self, nodes, force: bool = False) -> "BM25Indexer":
        if not nodes:
            raise ValueError("Cannot build BM25 index from an empty node list.")

        texts = [n.text for n in nodes]
        doc_ids = [n.node_id for n in nodes]
        corpus_hash = _hash_corpus(texts)

        if not force and self._index_path.exists():
            try:
                existing = _load_payload(self._index_path)
                if existing.corpus_hash == corpus_hash:
                    logger.info("BM25 corpus unchanged - skipping rebuild.")
                    self._payload = existing
                    return self
            except Exception:
                pass

        logger.info("Building BM25 index over %d chunks ...", len(texts))
        t0 = time.perf_counter()

        tokenised = [self._tokeniser(t) for t in texts]
        index = BM25Okapi(tokenised)
        payload = _BM25Payload(index=index, doc_ids=doc_ids, corpus_hash=corpus_hash)

        _atomic_save(payload, self._index_path)
        self._payload = payload

        elapsed = time.perf_counter() - t0
        print(f"BM25 index built in {elapsed:.2f}s  |  {len(texts)} chunks  |  saved -> {self._index_path}")
        return self

    def query(self, query_text: str, top_k: int = 20) -> List[Tuple[str, float]]:
        if self._payload is None:
            raise RuntimeError("BM25 index not loaded. Call BM25Indexer.load() or .build() first.")

        tokenised_query = self._tokeniser(query_text)
        scores = self._payload.index.get_scores(tokenised_query).tolist()
        scored = sorted(zip(self._payload.doc_ids, scores), key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    @classmethod
    def load(cls, index_path=DEFAULT_INDEX_PATH, tokeniser=None) -> "BM25Indexer":
        indexer = cls(index_path=index_path, tokeniser=tokeniser)
        indexer._payload = _load_payload(Path(index_path))
        print(f"BM25 index loaded  |  {len(indexer._payload.doc_ids)} chunks")
        return indexer

    def count(self) -> int:
        if self._payload is None:
            return 0
        return len(self._payload.doc_ids)

    @property
    def index_path(self) -> Path:
        return self._index_path


def _hash_corpus(texts: List[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
    return h.hexdigest()


def _atomic_save(payload: _BM25Payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise


def _load_payload(path: Path) -> _BM25Payload:
    if not path.exists():
        raise FileNotFoundError(
            f"BM25 index not found at '{path}'. Run ingestion with --rebuild-bm25 to build it."
        )
    with open(path, "rb") as f:
        return pickle.load(f)
