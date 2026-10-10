"""Provider-neutral LLM types and the provider interface.

Every provider adapter converts these neutral messages to its own wire format and
streams back :class:`LLMEvent` objects, so the agent loop never sees provider details.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Role = Literal["user", "assistant", "tool"]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None  # tool name, for tool results
    is_error: bool = False
    # Provider-invisible metadata (e.g. the offline planner's structured plan).
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cached_input_tokens += other.cached_input_tokens


@dataclass
class LLMEvent:
    type: Literal["text", "tool_call", "usage", "meta", "done"]
    text: str = ""
    tool_call: ToolCall | None = None
    usage: Usage | None = None
    stop_reason: str | None = None
    meta: dict[str, Any] | None = None


@dataclass
class AgentContext:
    """Structured context some providers (the offline planner) use instead of prose."""

    session: Any
    question: str


class LLMProvider(Protocol):
    name: str
    model: str

    def stream(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        context: AgentContext | None = None,
    ) -> AsyncIterator[LLMEvent]: ...
