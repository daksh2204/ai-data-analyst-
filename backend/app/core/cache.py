"""Thread-safe LRU cache with per-entry TTL, used for query results and answers."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any, Generic, TypeVar

from app.core.metrics import metrics

V = TypeVar("V")


def make_key(*parts: Any) -> str:
    raw = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


class TTLCache(Generic[V]):
    def __init__(self, name: str, max_entries: int = 512, ttl_seconds: float = 900) -> None:
        self.name = name
        self.max_entries = max_entries
        self.ttl = ttl_seconds
        self._data: OrderedDict[str, tuple[float, V]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> V | None:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                metrics.inc("cache_miss", cache=self.name)
                return None
            expires, value = item
            if expires < time.monotonic():
                del self._data[key]
                metrics.inc("cache_miss", cache=self.name)
                return None
            self._data.move_to_end(key)
            metrics.inc("cache_hit", cache=self.name)
            return value

    def set(self, key: str, value: V) -> None:
        with self._lock:
            self._data[key] = (time.monotonic() + self.ttl, value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        return len(self._data)
