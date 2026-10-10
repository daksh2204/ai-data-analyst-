"""Shared application state and FastAPI dependencies."""

from __future__ import annotations

import hmac
from dataclasses import dataclass
from typing import Any

from fastapi import Depends, Request

from app.agent.llm.base import LLMProvider
from app.agent.llm.factory import create_provider
from app.agent.llm.offline import OfflineProvider
from app.config import Settings
from app.core.cache import TTLCache
from app.core.errors import AuthError
from app.core.logging import session_id_var
from app.core.ratelimit import RateLimiter
from app.sessions.store import Session, SessionStore


@dataclass
class AppState:
    settings: Settings
    sessions: SessionStore
    provider: LLMProvider
    offline: OfflineProvider
    answer_cache: TTLCache[Any]
    tool_cache: TTLCache[Any]
    chat_limiter: RateLimiter

    @classmethod
    def build(cls, settings: Settings) -> AppState:
        return cls(
            settings=settings,
            sessions=SessionStore(
                ttl_minutes=settings.session_ttl_minutes,
                max_sessions=settings.max_sessions,
                query_timeout=settings.query_timeout_seconds,
                query_max_rows=settings.query_max_rows,
            ),
            provider=create_provider(settings),
            offline=OfflineProvider(),
            answer_cache=TTLCache("answers", settings.cache_max_entries, settings.cache_ttl_seconds),
            tool_cache=TTLCache("queries", settings.cache_max_entries, settings.cache_ttl_seconds),
            chat_limiter=RateLimiter(settings.chat_rate_limit_per_minute),
        )


def get_state(request: Request) -> AppState:
    return request.app.state.app_state  # type: ignore[no-any-return]


def require_auth(request: Request, state: AppState = Depends(get_state)) -> None:
    token = state.settings.app_access_token
    if not token:
        return
    header = request.headers.get("authorization", "")
    supplied = header[7:] if header.lower().startswith("bearer ") else request.query_params.get("token", "")
    if not supplied or not hmac.compare_digest(supplied.encode(), token.encode()):
        raise AuthError("This workspace needs an access token.")


def get_session(session_id: str, state: AppState = Depends(get_state)) -> Session:
    session = state.sessions.get(session_id)
    session_id_var.set(session_id[:8])
    return session


def chat_rate_limit(request: Request, state: AppState = Depends(get_state)) -> None:
    client = request.headers.get("authorization") or (request.client.host if request.client else "unknown")
    state.chat_limiter.check(client)
