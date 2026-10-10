"""The analyst agent: a streaming tool-calling loop over a session's data.

Event stream (each item is a JSON-serialisable dict with a ``type``):

``start``      turn id, provider, model
``notice``     non-fatal information (e.g. switched to offline mode)
``token``      a chunk of streamed answer text
``thought``    text the model wrote before calling tools (moved into the reasoning trace)
``step_start`` a tool call began (tool, purpose, input)
``step_end``   a tool call finished (status, summary, preview table, code, timing)
``artifact``   a chart / table / anomaly report / forecast / quality report to render
``final``      the complete assistant message (answer + artifacts + steps + usage)
``error``      a fatal error for this turn
"""

from __future__ import annotations

import asyncio
import re
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from app.agent.llm.base import AgentContext, LLMProvider, Message, ToolCall, Usage
from app.agent.memory import Turn, build_history
from app.agent.prompts import FINAL_ANSWER_NUDGE, build_system_prompt
from app.agent.tools import TOOL_SPECS, ToolExecutor
from app.config import Settings
from app.core.cache import TTLCache, make_key
from app.core.errors import LLMError
from app.core.logging import get_logger
from app.core.metrics import metrics
from app.sessions.store import Session

log = get_logger(__name__)


def normalise_question(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower()).rstrip("?.! ")


def _step_input(call: ToolCall) -> dict[str, Any]:
    args = {k: v for k, v in call.arguments.items() if k != "purpose"}
    return args


class AnalystAgent:
    def __init__(
        self,
        session: Session,
        provider: LLMProvider,
        settings: Settings,
        *,
        fallback: LLMProvider | None,
        answer_cache: TTLCache[Any],
        tool_cache: TTLCache[Any],
    ) -> None:
        self.session = session
        self.provider = provider
        self.settings = settings
        self.fallback = fallback
        self.answer_cache = answer_cache
        self.executor = ToolExecutor(
            session,
            max_rows_llm=settings.tool_result_max_rows,
            pandas_timeout=settings.pandas_timeout_seconds,
            cache=tool_cache,
        )

    def _cache_key(self, question: str, provider: LLMProvider) -> str:
        history = [(t.question, t.answer[:200]) for t in self.session.history[-4:]]
        return make_key("answer", self.session.fingerprint(), provider.name, provider.model, normalise_question(question), history)

    async def run(self, question: str) -> AsyncIterator[dict[str, Any]]:
        turn_id = uuid.uuid4().hex[:12]
        started = time.perf_counter()
        provider = self.provider
        yield {"type": "start", "turn_id": turn_id, "provider": provider.name, "model": provider.model}

        key = self._cache_key(question, provider)
        cached = self.answer_cache.get(key)
        if cached is not None:
            async for ev in self._replay(cached, question, turn_id, started):
                yield ev
            return

        system = build_system_prompt(self.session)
        base_messages = build_history(self.session.history, self.settings.agent_history_turns)
        messages: list[Message] = [*base_messages, Message(role="user", content=question)]
        turn_messages: list[Message] = []
        steps: list[dict[str, Any]] = []
        artifacts: list[dict[str, Any]] = []
        code_blocks: list[dict[str, str]] = []
        usage = Usage()
        answer = ""
        had_errors = False
        context = AgentContext(session=self.session, question=question)
        max_steps = self.settings.agent_max_steps
        llm_calls = 0

        step_no = 0
        while step_no <= max_steps:
            final_round = step_no == max_steps
            if final_round:
                nudge = Message(role="user", content=FINAL_ANSWER_NUDGE)
                messages.append(nudge)
            text_parts: list[str] = []
            calls: list[ToolCall] = []
            meta: dict[str, Any] = {}
            try:
                t0 = time.perf_counter()
                async for ev in provider.stream(
                    system=system, messages=messages, tools=[] if final_round else TOOL_SPECS, context=context
                ):
                    if ev.type == "text" and ev.text:
                        text_parts.append(ev.text)
                        yield {"type": "token", "text": ev.text}
                    elif ev.type == "tool_call" and ev.tool_call:
                        calls.append(ev.tool_call)
                    elif ev.type == "usage" and ev.usage:
                        usage.add(ev.usage)
                    elif ev.type == "meta" and ev.meta:
                        meta = ev.meta
                llm_calls += 1
                metrics.observe("llm_call_seconds", time.perf_counter() - t0, provider=provider.name)
                metrics.inc("llm_calls", provider=provider.name, status="ok")
            except LLMError as exc:
                metrics.inc("llm_calls", provider=provider.name, status="error")
                log.warning("llm call failed", extra={"provider": provider.name, "error": exc.message})
                can_fallback = (
                    self.fallback is not None
                    and self.settings.llm_fallback_to_offline
                    and provider is not self.fallback
                    and not steps
                )
                if can_fallback:
                    assert self.fallback is not None
                    provider = self.fallback
                    if text_parts:
                        yield {"type": "thought", "text": "".join(text_parts)}
                    yield {
                        "type": "notice",
                        "level": "warning",
                        "message": f"{exc.message} Answering with the offline analysis engine instead.",
                    }
                    messages = [*base_messages, Message(role="user", content=question)]
                    continue
                had_errors = True
                if steps:
                    answer = (
                        f"The language model stopped responding mid-analysis ({exc.message}). "
                        "The results computed so far are shown below."
                    )
                    break
                yield {"type": "error", "code": exc.code, "message": exc.message}
                return

            text = "".join(text_parts)
            assistant = Message(role="assistant", content=text, tool_calls=calls, meta=meta)
            messages.append(assistant)
            turn_messages.append(assistant)
            if not calls:
                answer = text
                break
            if text.strip():
                steps.append({"id": f"note_{len(steps)}", "kind": "note", "text": text.strip()})
                yield {"type": "thought", "text": text}

            for call in calls:
                step: dict[str, Any] = {
                    "id": call.id,
                    "kind": "tool",
                    "tool": call.name,
                    "purpose": str(call.arguments.get("purpose") or ""),
                    "input": _step_input(call),
                    "status": "running",
                }
                yield {"type": "step_start", "step": dict(step)}
                t0 = time.perf_counter()
                outcome = await asyncio.to_thread(self.executor.execute, call.name, call.arguments)
                elapsed = round((time.perf_counter() - t0) * 1000, 1)
                metrics.inc("tool_calls", tool=call.name, status="error" if outcome.is_error else "ok")
                metrics.observe("tool_call_seconds", elapsed / 1000, tool=call.name)
                step.update(
                    status="error" if outcome.is_error else "ok",
                    summary=outcome.summary,
                    preview=outcome.preview,
                    code=outcome.code,
                    elapsed_ms=elapsed,
                    result_id=outcome.result_id,
                )
                steps.append(step)
                yield {"type": "step_end", "step": step}
                for art in outcome.artifacts:
                    artifacts.append(art)
                    yield {"type": "artifact", "artifact": art}
                if outcome.code and not outcome.is_error and outcome.code not in code_blocks:
                    code_blocks.append(outcome.code)
                tool_msg = Message(
                    role="tool",
                    content=outcome.content,
                    tool_call_id=call.id,
                    name=call.name,
                    is_error=outcome.is_error,
                )
                messages.append(tool_msg)
                turn_messages.append(tool_msg)
            step_no += 1

        if not answer.strip():
            answer = "I wasn't able to produce an answer for that. Try rephrasing, or name the table and columns to use."
            had_errors = True

        elapsed_ms = round((time.perf_counter() - started) * 1000)
        message = {
            "id": turn_id,
            "role": "assistant",
            "question": question,
            "answer": answer,
            "artifacts": artifacts,
            "steps": steps,
            "code": code_blocks,
            "provider": provider.name,
            "model": provider.model,
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cached_input_tokens": usage.cached_input_tokens,
                "llm_calls": llm_calls,
            },
            "elapsed_ms": elapsed_ms,
            "cached": False,
            "created_at": time.time(),
        }
        self._record(question, message, turn_messages, code_blocks)
        metrics.observe("chat_turn_seconds", elapsed_ms / 1000, provider=provider.name)
        metrics.inc("tokens_input", usage.input_tokens, provider=provider.name)
        metrics.inc("tokens_output", usage.output_tokens, provider=provider.name)
        log.info(
            "chat turn complete",
            extra={
                "turn": turn_id, "provider": provider.name, "steps": sum(1 for s in steps if s.get("kind") == "tool"),
                "elapsed_ms": elapsed_ms, "in_tokens": usage.input_tokens, "out_tokens": usage.output_tokens,
            },
        )
        if not had_errors and not any(s.get("status") == "error" for s in steps):
            # Keyed on the context *before* this turn (the key computed at the start), so asking the
            # same question in the same context again is a hit. Answers produced by the offline
            # fallback are not cached, so the real LLM answers once it is reachable again.
            if provider is self.provider:
                self.answer_cache.set(key, {**message, "turn_messages": turn_messages})
        yield {"type": "final", "message": message}

    def _record(self, question: str, message: dict[str, Any], turn_messages: list[Message], code: list[dict[str, str]]) -> None:
        self.session.messages.append({"id": f"u_{message['id']}", "role": "user", "content": question, "created_at": time.time()})
        self.session.messages.append(message)
        self.session.history.append(Turn(question=question, messages=turn_messages, answer=message["answer"], code=code))

    async def _replay(self, cached: dict[str, Any], question: str, turn_id: str, started: float) -> AsyncIterator[dict[str, Any]]:
        metrics.inc("chat_cache_hits")
        turn_messages = cached.get("turn_messages", [])
        message = {k: v for k, v in cached.items() if k != "turn_messages"}
        message.update(id=turn_id, cached=True, elapsed_ms=round((time.perf_counter() - started) * 1000), created_at=time.time())
        for step in message["steps"]:
            if step.get("kind") == "tool":
                yield {"type": "step_end", "step": step}
        for art in message["artifacts"]:
            yield {"type": "artifact", "artifact": art}
        answer = message["answer"]
        for i in range(0, len(answer), 80):
            yield {"type": "token", "text": answer[i : i + 80]}
            await asyncio.sleep(0)
        self._record(question, message, turn_messages, message.get("code", []))
        yield {"type": "final", "message": message}
