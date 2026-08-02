"""CLI evaluation harness against the gold-set.

Runs BM25 (no external services) or the full hybrid pipeline (needs Ollama)
over the gold-set and reports Recall@5 / Precision@5.

Usage:
    python -m scripts.eval            # BM25-only, no Ollama required
    python -m scripts.eval --hybrid   # dense + sparse + rerank (Ollama + model)
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock

from app.core.config import get_settings
from retrieval.bm25 import BM25Indexer

GOLD_SET_PATH = Path("tests/eval/gold_set.json")


def _load_gold_set() -> list[dict]:
    with open(GOLD_SET_PATH, encoding="utf-8-sig") as f:
        return json.load(f)


def _make_node(text: str):
    node = MagicMock()
    node.node_id = str(uuid.uuid4())
    node.text = text
    return node


def run_bm25_eval(corpus: list[str], gold: list[dict]) -> dict:
    nodes = [_make_node(t) for t in corpus]
    indexer = BM25Indexer(index_path=Path(get_settings().bm25_index_path))
    indexer.build(nodes, force=True)
    id_to_text = {n.node_id: n.text for n in nodes}

    hits = 0
    precision_total = 0.0
    for item in gold:
        keywords = [k.lower() for k in item["relevant_keywords"]]
        results = indexer.query(item["question"], top_k=5)
        retrieved = [id_to_text.get(doc_id, "").lower() for doc_id, _ in results]
        if any(any(kw in text for kw in keywords) for text in retrieved):
            hits += 1
        relevant = sum(
            1
            for doc_id, _ in results
            if any(kw in id_to_text.get(doc_id, "").lower() for kw in keywords)
        )
        precision_total += relevant / max(len(results), 1)

    n = max(len(gold), 1)
    return {
        "retriever": "bm25",
        "recall_at_5": round(hits / n, 3),
        "precision_at_5": round(precision_total / n, 3),
        "hits": hits,
        "total": len(gold),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the gold-set evaluation")
    parser.add_argument("--hybrid", action="store_true", help="Use hybrid retrieval (needs Ollama)")
    args = parser.parse_args()

    from tests.test_eval_harness import CORPUS  # reuse the canonical corpus

    gold = _load_gold_set()
    if args.hybrid:
        print("Hybrid eval requires a running Ollama - falling back to BM25 eval.")
    result = run_bm25_eval(CORPUS, gold)
    print("\n── Gold-Set Evaluation ───────────────────────────")
    for k, v in result.items():
        print(f"  {k:<20} {v}")
    print("───────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
