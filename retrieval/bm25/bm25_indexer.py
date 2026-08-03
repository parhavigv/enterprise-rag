from __future__ import annotations

import hashlib
import logging
import os
import pickle
import re
import tempfile
import time
from pathlib import Path

from rank_bm25 import BM25Okapi

from retrieval.types import RetrievedDocument

logger = logging.getLogger(__name__)

DEFAULT_INDEX_PATH = Path(os.getenv("BM25_INDEX_PATH", "./bm25_index.pkl"))
_PUNCTUATION_RE = re.compile(r"[^\w\s]")


def _default_tokeniser(text: str) -> list[str]:
    text = text.lower()
    text = _PUNCTUATION_RE.sub(" ", text)
    return text.split()


def _source_matches(actual: object, expected: str) -> bool:
    """Exact (case-insensitive) match on the chunk's metadata source."""
    return str(actual or "").lower() == str(expected).lower()


class _BM25Payload:
    __slots__ = ("index", "doc_ids", "doc_texts", "doc_metas", "corpus_hash", "created_at")

    def __init__(self, index, doc_ids, doc_texts, doc_metas, corpus_hash):
        self.index = index
        self.doc_ids = doc_ids
        self.doc_texts = doc_texts
        self.doc_metas = doc_metas
        self.corpus_hash = corpus_hash
        self.created_at = time.time()


class BM25Indexer:
    def __init__(self, index_path=DEFAULT_INDEX_PATH, tokeniser=None):
        self._index_path = Path(index_path)
        self._tokeniser = tokeniser or _default_tokeniser
        self._payload: _BM25Payload | None = None

    def build(self, nodes, force: bool = False) -> BM25Indexer:
        if not nodes:
            raise ValueError("Cannot build BM25 index from an empty node list.")

        texts = [n.text for n in nodes]
        doc_ids = [n.node_id for n in nodes]
        doc_metas = [dict(getattr(n, "metadata", {}) or {}) for n in nodes]
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
        payload = _BM25Payload(
            index=index,
            doc_ids=doc_ids,
            doc_texts=dict(zip(doc_ids, texts, strict=False)),
            doc_metas=dict(zip(doc_ids, doc_metas, strict=False)),
            corpus_hash=corpus_hash,
        )

        _atomic_save(payload, self._index_path)
        self._payload = payload

        elapsed = time.perf_counter() - t0
        print(
            f"BM25 index built in {elapsed:.2f}s  |  {len(texts)} chunks  |  "
            f"saved -> {self._index_path}"
        )
        return self

    def add(self, nodes) -> BM25Indexer:
        """Merge ``nodes`` into the persisted index and rebuild.

        Incremental ingestion must never lose previously indexed documents:
        the existing corpus on disk is loaded, new node ids are appended
        (deduplicated), and the merged index is rebuilt and atomically saved.
        """
        new_texts = [n.text for n in nodes]
        new_ids = [n.node_id for n in nodes]
        new_metas = [dict(getattr(n, "metadata", {}) or {}) for n in nodes]
        if not new_texts:
            raise ValueError("Cannot merge an empty node list into the BM25 index.")

        existing_ids: list[str] = []
        existing_texts: list[str] = []
        existing_metas: list[dict] = []
        try:
            payload = _load_payload(self._index_path)
            existing_ids = list(payload.doc_ids)
            existing_texts = [payload.doc_texts[i] for i in existing_ids]
            existing_metas = [payload.doc_metas.get(i, {}) for i in existing_ids]
        except FileNotFoundError:
            pass

        seen = set(existing_ids)
        for node_id, text, meta in zip(new_ids, new_texts, new_metas, strict=False):
            if node_id not in seen:
                seen.add(node_id)
                existing_ids.append(node_id)
                existing_texts.append(text)
                existing_metas.append(meta)

        logger.info(
            "Merging %d chunks into BM25 corpus (%d total) ...",
            len(new_ids),
            len(existing_ids),
        )
        t0 = time.perf_counter()
        tokenised = [self._tokeniser(t) for t in existing_texts]
        payload = _BM25Payload(
            index=BM25Okapi(tokenised),
            doc_ids=existing_ids,
            doc_texts=dict(zip(existing_ids, existing_texts, strict=False)),
            doc_metas=dict(zip(existing_ids, existing_metas, strict=False)),
            corpus_hash=_hash_corpus(existing_texts),
        )
        _atomic_save(payload, self._index_path)
        self._payload = payload
        elapsed = time.perf_counter() - t0
        print(f"BM25 index rebuilt in {elapsed:.2f}s  |  {len(existing_ids)} chunks")
        return self

    def query(self, query_text: str, top_k: int = 20) -> list[tuple[str, float]]:
        if self._payload is None:
            raise RuntimeError("BM25 index not loaded. Call BM25Indexer.load() or .build() first.")

        tokenised_query = self._tokeniser(query_text)
        scores = self._payload.index.get_scores(tokenised_query).tolist()
        scored = sorted(
            zip(self._payload.doc_ids, scores, strict=False),
            key=lambda x: x[1],
            reverse=True,
        )
        return scored[:top_k]

    def query_documents(
        self,
        query_text: str,
        top_k: int = 20,
        source: str | None = None,
    ) -> list[RetrievedDocument]:
        """Query and hydrate sparse hits into ``RetrievedDocument`` objects.

        When ``source`` is given, only chunks whose ``metadata["source"]``
        matches are returned (exact, case-insensitive match). A larger
        candidate pool is scored so filtering cannot starve the top-k.
        """
        if source:
            pool = max(100, top_k * 10)
            results = self.query(query_text, top_k=pool)
        else:
            results = self.query(query_text, top_k=top_k)

        out: list[RetrievedDocument] = []
        for doc_id, score in results:
            if doc_id not in self._payload.doc_texts:
                continue
            meta = self._payload.doc_metas.get(doc_id, {})
            if source and not _source_matches(meta.get("source"), source):
                continue
            out.append(
                RetrievedDocument(
                    node_id=doc_id,
                    text=self._payload.doc_texts[doc_id],
                    score=float(score),
                    metadata=meta,
                    source="sparse",
                )
            )
            if len(out) >= top_k:
                break
        return out

    def document_text(self, doc_id: str) -> str | None:
        if self._payload is None:
            return None
        return self._payload.doc_texts.get(doc_id)

    @classmethod
    def load(cls, index_path=DEFAULT_INDEX_PATH, tokeniser=None) -> BM25Indexer:
        indexer = cls(index_path=index_path, tokeniser=tokeniser)
        payload = _load_payload(Path(index_path))
        if not hasattr(payload, "doc_texts") or not hasattr(payload, "doc_metas"):
            raise RuntimeError(
                f"BM25 index at '{index_path}' is an outdated format. "
                "Rebuild it with ingestion --rebuild-bm25 to upgrade."
            )
        indexer._payload = payload
        print(f"BM25 index loaded  |  {len(indexer._payload.doc_ids)} chunks")
        return indexer

    def count(self) -> int:
        if self._payload is None:
            return 0
        return len(self._payload.doc_ids)

    @property
    def index_path(self) -> Path:
        return self._index_path


def _hash_corpus(texts: list[str]) -> str:
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
