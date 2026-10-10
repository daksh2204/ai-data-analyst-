"""Structured logging with a per-request correlation id."""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import UTC, datetime

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
session_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("session_id", default="-")

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        record.session_id = session_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
            "session_id": getattr(record, "session_id", "-"),
        }
        extras = {k: v for k, v in record.__dict__.items() if k not in _RESERVED and k not in payload}
        payload.update(extras)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = (
            f"{datetime.fromtimestamp(record.created).strftime('%H:%M:%S')} "
            f"{record.levelname:<7} {record.name}: {record.getMessage()}"
        )
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in _RESERVED and k not in {"request_id", "session_id"}
        }
        rid = getattr(record, "request_id", "-")
        if rid != "-":
            extras = {"rid": rid, **extras}
        if extras:
            base += "  " + " ".join(f"{k}={v}" for k, v in extras.items())
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging(level: str = "INFO", fmt: str = "text") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_ContextFilter())
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for noisy in ("httpx", "httpcore", "openai", "anthropic", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
