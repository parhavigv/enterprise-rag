"""Chunk-size presets.

Sizes are expressed in *tokens* (approximate). The default ``512T`` was
empirically selected as the best precision/recall trade-off on the Week 1
gold set (Recall@5 = 1.00, Precision@5 = 0.50).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkConfig:
    label: str
    chunk_size: int
    chunk_overlap: int


CHUNK_CONFIGS: dict[str, ChunkConfig] = {
    "256T": ChunkConfig("256T", chunk_size=256, chunk_overlap=40),
    "512T": ChunkConfig("512T", chunk_size=512, chunk_overlap=80),
    "1024T": ChunkConfig("1024T", chunk_size=1024, chunk_overlap=150),
}
DEFAULT_CHUNK_CONFIG = "512T"


def get_chunk_config(label: str) -> ChunkConfig:
    if label not in CHUNK_CONFIGS:
        raise ValueError(
            f"Unknown chunk config '{label}'. Choose one of: {', '.join(CHUNK_CONFIGS)}"
        )
    return CHUNK_CONFIGS[label]
