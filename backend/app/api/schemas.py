"""Request models and serialisers for API responses."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.analysis.dashboard import suggest_questions
from app.sessions.store import Session, TableEntry


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="The question to ask about the data.")


class FixRequest(BaseModel):
    fix: Literal["drop_duplicates", "normalize_labels"]
    column: str | None = None


class QueryRequest(BaseModel):
    sql: str = Field(..., min_length=1, max_length=20_000)


def dataset_summary(entry: TableEntry) -> dict[str, Any]:
    p = entry.profile
    return {
        "name": entry.name,
        "filename": entry.loaded.filename,
        "rows": p.row_count,
        "columns": p.column_count,
        "size_bytes": entry.loaded.size_bytes,
        "encoding": entry.loaded.encoding,
        "delimiter": entry.loaded.delimiter,
        "quality_score": p.quality_score,
        "issue_count": len(p.issues),
        "notes": [n.to_dict() for n in entry.loaded.notes],
        "column_summary": [{"name": c.name, "dtype": c.dtype, "role": c.role} for c in p.columns],
    }


def session_state(session: Session, provider: str, model: str) -> dict[str, Any]:
    return {
        "id": session.id,
        "created_at": session.created_at,
        "datasets": [dataset_summary(t) for t in session.tables.values()],
        "relationships": [r.to_dict() for r in session.relationships],
        "messages": session.messages,
        "suggestions": suggest_questions([t.profile for t in session.tables.values()]),
        "provider": provider,
        "model": model,
    }
