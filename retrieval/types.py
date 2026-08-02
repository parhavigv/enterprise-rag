"""Shared value types used across the retrieval layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class RetrievedDocument:
    """A single ranked hit produced by a retriever.

    ``source`` records which retriever produced the hit ("dense",
    "sparse", or "hybrid") so downstream re-ranking / citations can be
    audited.
    """

    node_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = "hybrid"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
