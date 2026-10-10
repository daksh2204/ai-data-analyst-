"""Conversation memory with compaction.

Recent turns are replayed verbatim (including tool calls and results, so the model can
build on earlier queries); older turns are reduced to question + answer + the SQL used.
A character budget bounds the total so long sessions never overflow the context window.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.agent.llm.base import Message

TOOL_RESULT_CHARS = 2500
OLD_ANSWER_CHARS = 1200
TOTAL_BUDGET_CHARS = 90_000


@dataclass
class Turn:
    question: str
    messages: list[Message]  # assistant tool-call messages, tool results, final assistant answer
    answer: str
    code: list[dict[str, str]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f" …[truncated {len(text) - limit} chars]"


def _verbatim(turn: Turn) -> list[Message]:
    out = [Message(role="user", content=turn.question)]
    for m in turn.messages:
        if m.role == "tool":
            out.append(
                Message(
                    role="tool",
                    content=_truncate(m.content, TOOL_RESULT_CHARS),
                    tool_call_id=m.tool_call_id,
                    name=m.name,
                    is_error=m.is_error,
                )
            )
        else:
            out.append(m)
    return out


def _compacted(turn: Turn) -> list[Message]:
    sql = [c["code"] for c in turn.code if c.get("language") == "sql"][:2]
    note = ""
    if sql:
        note = "\n\n(Queries used: " + " | ".join(" ".join(s.split())[:400] for s in sql) + ")"
    meta = next((m.meta for m in reversed(turn.messages) if m.role == "assistant" and m.meta), {})
    return [
        Message(role="user", content=turn.question),
        Message(role="assistant", content=_truncate(turn.answer, OLD_ANSWER_CHARS) + note, meta=meta),
    ]


def build_history(turns: list[Turn], keep_verbatim: int) -> list[Message]:
    messages: list[list[Message]] = []
    for i, turn in enumerate(turns):
        recent = i >= len(turns) - keep_verbatim
        messages.append(_verbatim(turn) if recent else _compacted(turn))

    def size(chunks: list[list[Message]]) -> int:
        return sum(len(m.content) + sum(len(str(tc.arguments)) for tc in m.tool_calls) for c in chunks for m in c)

    # Drop oldest turns until we fit the budget (always keep the latest turn).
    while len(messages) > 1 and size(messages) > TOTAL_BUDGET_CHARS:
        messages.pop(0)
    return [m for chunk in messages for m in chunk]
