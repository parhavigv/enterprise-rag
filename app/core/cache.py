"""Thread-safe in-memory TTL cache for query responses."""

from __future__ import annotations

import threading
import time


class TTLCache:
    def __init__(self, ttl_seconds: int = 600, max_entries: int = 512) -> None:
        self._ttl = ttl_seconds
        self._max_entries = max(1, max_entries)
        self._data: dict[str, tuple[float, object]] = {}
        self._lock = threading.Lock()

    def get(self, key: str):
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            expires_at, value = item
            if time.monotonic() > expires_at:
                del self._data[key]
                return None
            return value

    def set(self, key: str, value: object) -> None:
        with self._lock:
            if len(self._data) >= self._max_entries:
                self._evict_expired()
            if len(self._data) >= self._max_entries:
                # Simple FIFO eviction under pressure.
                oldest = min(self._data, key=lambda k: self._data[k][0])
                del self._data[oldest]
            self._data[key] = (time.monotonic() + self._ttl, value)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            self._evict_expired()
            return len(self._data)

    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired = [k for k, (exp, _) in self._data.items() if now > exp]
        for k in expired:
            del self._data[k]
