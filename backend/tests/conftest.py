"""Shared fixtures: an app wired to a scripted LLM, sessions preloaded with the sample data."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.agent.llm.base import AgentContext, LLMEvent, Message, ToolCall, ToolSpec, Usage
from app.config import Settings
from app.core.errors import LLMError
from app.data.ingestion import load_csv
from app.main import create_app
from app.sessions.store import Session, SessionStore

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "sample_data"
SAMPLE_FILES = ("sales_orders.csv", "customers.csv", "products.csv", "regional_targets.csv")


# --------------------------------------------------------------------------- scripted LLM


Step = Callable[[list[Message]], list[LLMEvent]] | list[LLMEvent]


def text(t: str) -> LLMEvent:
    return LLMEvent(type="text", text=t)


def call(name: str, call_id: str | None = None, **arguments: Any) -> LLMEvent:
    return LLMEvent(type="tool_call", tool_call=ToolCall(id=call_id or f"call_{name}", name=name, arguments=arguments))


class ScriptedProvider:
    """A fake LLM that replays a fixed script of rounds and records every request it receives.

    Each round is a list of events (text chunks and/or tool calls) or a callable that builds
    them from the messages so far, which lets a test react to tool results.
    """

    name = "scripted"
    model = "scripted-model"

    def __init__(self, rounds: list[Step], *, fail_with: str | None = None) -> None:
        self.rounds = list(rounds)
        self.requests: list[dict[str, Any]] = []
        self.fail_with = fail_with

    async def stream(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        context: AgentContext | None = None,
    ) -> AsyncIterator[LLMEvent]:
        self.requests.append({"system": system, "messages": list(messages), "tools": [t.name for t in tools]})
        if self.fail_with:
            raise LLMError(self.fail_with)
        if not self.rounds:
            events: list[LLMEvent] = [text("Done.")]
        else:
            step = self.rounds.pop(0)
            events = step(messages) if callable(step) else step
        for ev in events:
            yield ev
        yield LLMEvent(type="usage", usage=Usage(input_tokens=100, output_tokens=20))
        yield LLMEvent(type="done", stop_reason="tool_use" if any(e.type == "tool_call" for e in events) else "end_turn")


def last_tool_result(messages: list[Message]) -> dict[str, Any]:
    tool_msgs = [m for m in messages if m.role == "tool"]
    return json.loads(tool_msgs[-1].content)


# --------------------------------------------------------------------------- data fixtures


def read_sample(name: str) -> bytes:
    return (SAMPLES / name).read_bytes()


@pytest.fixture
def settings() -> Settings:
    return Settings(
        llm_provider="offline",
        environment="test",
        app_access_token=None,
        sample_data_dir=SAMPLES,
        frontend_dist=ROOT / "does-not-exist",
    )


@pytest.fixture
def store() -> SessionStore:
    return SessionStore(ttl_minutes=30, max_sessions=10, query_timeout=10, query_max_rows=5000)


@pytest.fixture
def session(store: SessionStore) -> Session:
    s = store.create()
    for fname in SAMPLE_FILES:
        loaded = load_csv(
            read_sample(fname),
            fname,
            existing_tables=set(s.tables),
            max_bytes=50 * 1024 * 1024,
            max_rows=2_000_000,
            max_columns=500,
        )
        s.add_table(loaded)
    return s


@pytest.fixture
def client(settings: Settings) -> TestClient:
    return TestClient(create_app(settings))


@pytest.fixture
def api_session(client: TestClient) -> str:
    sid = client.post("/api/sessions").json()["id"]
    r = client.post(f"/api/sessions/{sid}/datasets/samples")
    assert r.status_code == 200, r.text
    return sid


def sse_events(client: TestClient, sid: str, message: str) -> list[dict[str, Any]]:
    with client.stream("POST", f"/api/sessions/{sid}/chat", json={"message": message}) as resp:
        assert resp.status_code == 200, resp.read()
        return [json.loads(line[6:]) for line in resp.iter_lines() if line.startswith("data: ")]
