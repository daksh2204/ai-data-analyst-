"""Per-session DuckDB engine with a read-only SQL guard and query timeouts.

Safety model for LLM-written SQL:

* exactly one statement, and it must be a ``SELECT`` (incl. ``WITH``/``FROM``-first);
* the database is in-memory and only holds copies of the uploaded tables;
* ``enable_external_access`` is off (no file system, HTTP, or Python-object scans) and the
  configuration is locked, so a query cannot re-enable it;
* every query runs under a wall-clock timeout (``connection.interrupt()``) and results
  are capped at ``max_rows``.
"""

from __future__ import annotations

import datetime as dt
import decimal
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from app.core.errors import QueryError, QueryTimeoutError, UnsafeCodeError
from app.core.logging import get_logger
from app.core.metrics import metrics

log = get_logger(__name__)


@dataclass
class QueryResult:
    sql: str
    df: pd.DataFrame
    row_count: int
    truncated: bool
    elapsed_ms: float
    id: str = field(default_factory=lambda: f"r{uuid.uuid4().hex[:6]}")

    @property
    def columns(self) -> list[str]:
        return [str(c) for c in self.df.columns]


def _py(v: Any) -> Any:
    if isinstance(v, decimal.Decimal):
        return float(v)
    if isinstance(v, (dt.datetime, dt.date)):
        return v
    if isinstance(v, (np.generic,)):
        return v.item()
    return v


class SessionDatabase:
    """Wraps one in-memory DuckDB connection. Thread-safe via a lock."""

    def __init__(self, *, timeout_seconds: float = 15.0, max_rows: int = 5000) -> None:
        self._con = duckdb.connect(database=":memory:")
        self._lock = threading.RLock()
        self.timeout_seconds = timeout_seconds
        self.max_rows = max_rows
        self._con.execute("SET threads TO 2")
        self._con.execute("SET memory_limit = '1GB'")
        self._con.execute("SET enable_external_access = false")
        self._con.execute("SET lock_configuration = true")

    # ------------------------------------------------------------------ tables
    def register_table(self, name: str, df: pd.DataFrame, date_columns: list[str] | None = None) -> None:
        tmp = f"__load_{uuid.uuid4().hex[:8]}"
        with self._lock:
            self._con.register(tmp, df)
            try:
                select_list = ", ".join(
                    f'CAST("{c}" AS DATE) AS "{c}"' if c in (date_columns or []) else f'"{c}"' for c in df.columns
                )
                self._con.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT {select_list} FROM "{tmp}"')
            finally:
                self._con.unregister(tmp)

    def drop_table(self, name: str) -> None:
        with self._lock:
            self._con.execute(f'DROP TABLE IF EXISTS "{name}"')

    def tables(self) -> list[str]:
        with self._lock:
            return [r[0] for r in self._con.execute("SELECT table_name FROM information_schema.tables").fetchall()]

    # ------------------------------------------------------------------ queries
    @staticmethod
    def validate_sql(sql: str) -> str:
        cleaned = sql.strip().rstrip(";").strip()
        if not cleaned:
            raise QueryError("The SQL query is empty.")
        try:
            statements = duckdb.extract_statements(cleaned)
        except duckdb.Error as exc:
            raise QueryError(f"SQL syntax error: {exc}") from exc
        if len(statements) != 1:
            raise UnsafeCodeError("Send exactly one SQL statement per call.")
        if statements[0].type != duckdb.StatementType.SELECT:
            kind = str(statements[0].type).split(".")[-1]
            raise UnsafeCodeError(
                f"Only read-only SELECT queries are allowed (got {kind}). The data cannot be modified."
            )
        return cleaned

    def execute(self, sql: str, *, max_rows: int | None = None, timeout: float | None = None) -> QueryResult:
        cleaned = self.validate_sql(sql)
        limit = max_rows or self.max_rows
        deadline = timeout or self.timeout_seconds
        with self._lock:
            cur = self._con.cursor()
            timed_out = threading.Event()

            def _interrupt() -> None:
                timed_out.set()
                cur.interrupt()

            timer = threading.Timer(deadline, _interrupt)
            start = time.perf_counter()
            timer.start()
            try:
                cur.execute(cleaned)
                columns = [d[0] for d in (cur.description or [])]
                rows = cur.fetchmany(limit + 1)
            except duckdb.InterruptException as exc:
                metrics.inc("sql_queries", status="timeout")
                raise QueryTimeoutError(
                    f"The query took longer than {deadline:.0f}s and was stopped. Simplify it or aggregate first."
                ) from exc
            except duckdb.Error as exc:
                if timed_out.is_set():
                    metrics.inc("sql_queries", status="timeout")
                    raise QueryTimeoutError(f"The query took longer than {deadline:.0f}s and was stopped.") from exc
                metrics.inc("sql_queries", status="error")
                raise QueryError(_clean_duckdb_error(exc)) from exc
            finally:
                timer.cancel()
                cur.close()
        elapsed = (time.perf_counter() - start) * 1000
        truncated = len(rows) > limit
        rows = rows[:limit]
        df = pd.DataFrame([[_py(v) for v in r] for r in rows], columns=_dedupe(columns))
        metrics.inc("sql_queries", status="ok")
        metrics.observe("sql_query_seconds", elapsed / 1000)
        return QueryResult(sql=cleaned, df=df, row_count=len(df), truncated=truncated, elapsed_ms=round(elapsed, 1))

    def close(self) -> None:
        with self._lock:
            try:
                self._con.close()
            except duckdb.Error:  # pragma: no cover
                pass


def _dedupe(columns: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for c in columns:
        if c in seen:
            seen[c] += 1
            out.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            out.append(c)
    return out


def _clean_duckdb_error(exc: Exception) -> str:
    msg = str(exc).strip()
    # Keep the useful first lines (DuckDB appends a caret diagram).
    lines = [ln for ln in msg.splitlines() if ln.strip()]
    return " ".join(lines[:4])[:600]
