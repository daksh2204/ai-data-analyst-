"""Streaming chat endpoint (Server-Sent Events over POST)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.agent.agent import AnalystAgent
from app.api.deps import AppState, chat_rate_limit, get_session, get_state, require_auth
from app.api.schemas import ChatRequest
from app.core.errors import LimitExceededError, ValidationError
from app.core.logging import get_logger
from app.sessions.store import Session

log = get_logger(__name__)
router = APIRouter(prefix="/api/sessions/{session_id}", tags=["chat"], dependencies=[Depends(require_auth)])


def sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, default=str, allow_nan=False)}\n\n"


@router.post("/chat", dependencies=[Depends(chat_rate_limit)])
async def chat(body: ChatRequest, session: Session = Depends(get_session), state: AppState = Depends(get_state)) -> StreamingResponse:
    """Ask a question. Streams events: start, notice, token, thought, step_start, step_end, artifact, final, error."""
    if not session.tables:
        raise ValidationError("Upload a CSV file (or load the sample data) before asking questions.")
    if session.chat_lock.locked():
        raise LimitExceededError("Still working on the previous question in this session.")

    agent = AnalystAgent(
        session,
        state.provider,
        state.settings,
        fallback=state.offline,
        answer_cache=state.answer_cache,
        tool_cache=state.tool_cache,
    )

    async def events() -> AsyncIterator[str]:
        async with session.chat_lock:
            try:
                async for event in agent.run(body.message.strip()):
                    yield sse(_sanitize(event))
            except Exception as exc:  # pragma: no cover - last-resort guard; details are logged
                log.exception("chat turn crashed")
                yield sse({"type": "error", "code": "internal_error", "message": f"Something went wrong: {type(exc).__name__}."})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


def _sanitize(obj: Any) -> Any:
    """Replace NaN/inf (invalid JSON) with None, recursively."""
    if isinstance(obj, float):
        return None if obj != obj or obj in (float("inf"), float("-inf")) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj
