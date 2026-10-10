"""In-memory analysis sessions: uploaded tables, DuckDB engine, results and chat history.

Sessions expire after ``session_ttl_minutes`` of inactivity and the store is capped at
``max_sessions`` (least recently used is evicted). State is intentionally in-memory: an
analysis session is a scratchpad. For multi-instance deployments, back this with Redis
plus object storage for the uploaded files (see README, "Scaling").
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.core.errors import LimitExceededError, NotFoundError
from app.core.logging import get_logger
from app.core.metrics import metrics
from app.data.engine import SessionDatabase
from app.data.ingestion import LoadedTable
from app.data.profiling import TableProfile, profile_table
from app.data.relationships import Relationship, detect_relationships

log = get_logger(__name__)


@dataclass
class ResultSet:
    """A tabular result produced by a tool (SQL or pandas), referenceable by id."""

    id: str
    df: pd.DataFrame
    source: str  # "sql" | "pandas"
    code: str
    title: str | None = None


@dataclass
class TableEntry:
    loaded: LoadedTable
    profile: TableProfile

    @property
    def name(self) -> str:
        return self.loaded.name

    @property
    def df(self) -> pd.DataFrame:
        return self.loaded.df


@dataclass
class Session:
    id: str
    db: SessionDatabase
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    tables: dict[str, TableEntry] = field(default_factory=dict)
    relationships: list[Relationship] = field(default_factory=list)
    results: OrderedDict[str, ResultSet] = field(default_factory=OrderedDict)
    history: list[Any] = field(default_factory=list)  # list[agent.memory.Turn]
    messages: list[dict[str, Any]] = field(default_factory=list)  # UI transcript
    chat_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _counter: int = 0

    MAX_RESULTS = 200

    def touch(self) -> None:
        self.last_active = time.time()

    # --------------------------------------------------------------- tables
    def add_table(self, loaded: LoadedTable) -> TableEntry:
        profile = profile_table(loaded.name, loaded.df, loaded.original_columns)
        date_cols = [c.name for c in profile.columns if c.dtype == "date"]
        self.db.register_table(loaded.name, loaded.df, date_columns=date_cols)
        entry = TableEntry(loaded=loaded, profile=profile)
        self.tables[loaded.name] = entry
        self._refresh_relationships()
        return entry

    def replace_table_df(self, name: str, df: pd.DataFrame) -> TableEntry:
        entry = self.get_table(name)
        entry.loaded.df = df.reset_index(drop=True)
        entry.profile = profile_table(name, entry.loaded.df, entry.loaded.original_columns)
        date_cols = [c.name for c in entry.profile.columns if c.dtype == "date"]
        self.db.register_table(name, entry.loaded.df, date_columns=date_cols)
        self._refresh_relationships()
        return entry

    def remove_table(self, name: str) -> None:
        self.get_table(name)
        self.db.drop_table(name)
        del self.tables[name]
        self._refresh_relationships()

    def get_table(self, name: str) -> TableEntry:
        if name not in self.tables:
            raise NotFoundError(f"There is no table called '{name}'. Available: {', '.join(self.tables) or 'none'}.")
        return self.tables[name]

    def _refresh_relationships(self) -> None:
        self.relationships = detect_relationships(
            {n: t.df for n, t in self.tables.items()}, {n: t.profile for n, t in self.tables.items()}
        )

    @property
    def primary_table(self) -> TableEntry | None:
        if not self.tables:
            return None
        return max(self.tables.values(), key=lambda t: (len(t.profile.by_role("measure")) > 0, t.profile.row_count))

    def fingerprint(self) -> str:
        return "|".join(f"{n}:{t.loaded.fingerprint}:{t.profile.row_count}" for n, t in sorted(self.tables.items()))

    # --------------------------------------------------------------- results
    def next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}{self._counter}"

    def store_result(self, df: pd.DataFrame, source: str, code: str, title: str | None = None) -> ResultSet:
        rs = ResultSet(id=self.next_id("r"), df=df, source=source, code=code, title=title)
        self.results[rs.id] = rs
        while len(self.results) > self.MAX_RESULTS:
            self.results.popitem(last=False)
        return rs

    def get_result(self, result_id: str) -> ResultSet:
        rs = self.results.get(result_id)
        if rs is None:
            known = ", ".join(list(self.results)[-8:]) or "none yet"
            raise NotFoundError(f"Unknown result_id '{result_id}'. Recent result ids: {known}.")
        return rs

    def reset_conversation(self) -> None:
        self.history.clear()
        self.messages.clear()
        self.results.clear()

    def close(self) -> None:
        self.db.close()


class SessionStore:
    def __init__(self, *, ttl_minutes: int, max_sessions: int, query_timeout: float, query_max_rows: int) -> None:
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        self._lock = threading.Lock()
        self.ttl_seconds = ttl_minutes * 60
        self.max_sessions = max_sessions
        self.query_timeout = query_timeout
        self.query_max_rows = query_max_rows

    def create(self) -> Session:
        self.evict_expired()
        with self._lock:
            if len(self._sessions) >= self.max_sessions:
                _, oldest = self._sessions.popitem(last=False)
                oldest.close()
                log.info("session evicted (capacity)", extra={"evicted": oldest.id})
            sid = uuid.uuid4().hex
            session = Session(
                id=sid, db=SessionDatabase(timeout_seconds=self.query_timeout, max_rows=self.query_max_rows)
            )
            self._sessions[sid] = session
        metrics.inc("sessions_created")
        return session

    def get(self, session_id: str) -> Session:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or time.time() - session.last_active > self.ttl_seconds:
                if session is not None:
                    self._sessions.pop(session_id, None)
                    session.close()
                raise NotFoundError("This session has expired or doesn't exist. Start a new session.")
            self._sessions.move_to_end(session_id)
            session.touch()
            return session

    def delete(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            raise NotFoundError("Session not found.")
        session.close()

    def evict_expired(self) -> int:
        now = time.time()
        with self._lock:
            expired = [sid for sid, s in self._sessions.items() if now - s.last_active > self.ttl_seconds]
            for sid in expired:
                self._sessions.pop(sid).close()
        if expired:
            log.info("sessions expired", extra={"count": len(expired)})
        return len(expired)

    def __len__(self) -> int:
        return len(self._sessions)

    def check_file_limit(self, session: Session, incoming: int, limit: int) -> None:
        if len(session.tables) + incoming > limit:
            raise LimitExceededError(f"A session can hold up to {limit} files. Remove one before adding more.")
