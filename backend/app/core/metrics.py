"""Tiny in-process metrics registry (counters + latency summaries).

Exposed as JSON on ``/api/metrics`` and in Prometheus text format on
``/api/metrics?format=prometheus``. Deliberately dependency-free; swap for
``prometheus_client`` or OpenTelemetry in a multi-instance deployment.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


def _key(name: str, labels: dict[str, str] | None) -> tuple[str, tuple[tuple[str, str], ...]]:
    return name, tuple(sorted((labels or {}).items()))


@dataclass
class _Summary:
    count: int = 0
    total: float = 0.0
    max: float = 0.0
    recent: list[float] = field(default_factory=list)

    def observe(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.max = max(self.max, value)
        self.recent.append(value)
        if len(self.recent) > 500:
            del self.recent[: len(self.recent) - 500]

    def percentile(self, q: float) -> float:
        if not self.recent:
            return 0.0
        data = sorted(self.recent)
        idx = min(len(data) - 1, round(q * (len(data) - 1)))
        return data[idx]


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self._summaries: dict[tuple[str, tuple[tuple[str, str], ...]], _Summary] = defaultdict(_Summary)
        self.started_at = time.time()

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        with self._lock:
            self._counters[_key(name, labels)] += value

    def observe(self, name: str, value: float, **labels: str) -> None:
        with self._lock:
            self._summaries[_key(name, labels)].observe(value)

    @contextmanager
    def timer(self, name: str, **labels: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, time.perf_counter() - start, **labels)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            counters = [
                {"name": n, "labels": dict(lbl), "value": v} for (n, lbl), v in sorted(self._counters.items())
            ]
            summaries = [
                {
                    "name": n,
                    "labels": dict(lbl),
                    "count": s.count,
                    "avg": round(s.total / s.count, 4) if s.count else 0.0,
                    "p50": round(s.percentile(0.5), 4),
                    "p95": round(s.percentile(0.95), 4),
                    "max": round(s.max, 4),
                }
                for (n, lbl), s in sorted(self._summaries.items())
            ]
        return {"uptime_seconds": round(time.time() - self.started_at, 1), "counters": counters, "summaries": summaries}

    def prometheus(self) -> str:
        lines: list[str] = []
        snap = self.snapshot()

        def fmt_labels(labels: dict[str, str]) -> str:
            if not labels:
                return ""
            inner = ",".join(f'{k}="{v}"' for k, v in labels.items())
            return "{" + inner + "}"

        for c in snap["counters"]:  # type: ignore[union-attr]
            lines.append(f"{c['name']}_total{fmt_labels(c['labels'])} {c['value']}")
        for s in snap["summaries"]:  # type: ignore[union-attr]
            lbl = fmt_labels(s["labels"])
            lines.append(f"{s['name']}_count{lbl} {s['count']}")
            lines.append(f"{s['name']}_avg_seconds{lbl} {s['avg']}")
            lines.append(f"{s['name']}_p95_seconds{lbl} {s['p95']}")
        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._summaries.clear()


metrics = Metrics()
