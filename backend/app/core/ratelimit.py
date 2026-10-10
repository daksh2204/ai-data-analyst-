"""In-memory sliding-window rate limiter for expensive endpoints (LLM calls cost money).

Keyed by client IP (or the access token when auth is on). For several instances behind a
load balancer, move the window to Redis; the interface stays the same.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from app.core.errors import LimitExceededError
from app.core.metrics import metrics


class RateLimiter:
    def __init__(self, limit: int, window_seconds: float = 60.0) -> None:
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        if self.limit <= 0:
            return
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                retry = int(self.window - (now - hits[0])) + 1
                metrics.inc("rate_limited")
                raise LimitExceededError(
                    f"Too many questions in a short time ({self.limit} per minute). Try again in {retry}s.",
                    details={"retry_after_seconds": retry},
                )
            hits.append(now)
            if len(self._hits) > 10_000:  # bound memory under many distinct clients
                for k in [k for k, v in self._hits.items() if not v][:5_000]:
                    del self._hits[k]
