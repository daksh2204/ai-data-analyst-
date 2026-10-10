"""Session lifecycle."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Response, status

from app.api.deps import AppState, get_session, get_state, require_auth
from app.api.schemas import session_state
from app.sessions.store import Session

router = APIRouter(prefix="/api/sessions", tags=["sessions"], dependencies=[Depends(require_auth)])


@router.post("", status_code=status.HTTP_201_CREATED)
def create_session(state: AppState = Depends(get_state)) -> dict[str, Any]:
    session = state.sessions.create()
    return session_state(session, state.provider.name, state.provider.model)


@router.get("/{session_id}")
def read_session(session: Session = Depends(get_session), state: AppState = Depends(get_state)) -> dict[str, Any]:
    return session_state(session, state.provider.name, state.provider.model)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(session_id: str, state: AppState = Depends(get_state)) -> Response:
    state.sessions.delete(session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{session_id}/reset")
def reset_conversation(session: Session = Depends(get_session), state: AppState = Depends(get_state)) -> dict[str, Any]:
    """Clear the conversation but keep the uploaded data."""
    session.reset_conversation()
    return session_state(session, state.provider.name, state.provider.model)
