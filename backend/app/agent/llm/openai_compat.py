"""Adapter for any provider speaking the OpenAI Chat Completions protocol.

Covers OpenAI, Google Gemini (OpenAI-compatible endpoint), Groq, OpenRouter and Ollama.
Tool-call arguments arrive as streamed JSON fragments keyed by index; they are
accumulated and parsed once the stream ends.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

import openai
from openai import AsyncOpenAI

from app.agent.llm.base import AgentContext, LLMEvent, Message, ToolCall, ToolSpec, Usage
from app.core.errors import LLMError
from app.core.logging import get_logger

log = get_logger(__name__)

# Reasoning-model families that reject a custom temperature.
_NO_TEMPERATURE_PREFIXES = ("gpt-5", "o1", "o3", "o4", "gpt-6")


def parse_arguments(raw: str) -> dict[str, Any]:
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {"_raw": raw}
    except json.JSONDecodeError:
        # Some models emit two JSON objects back to back or trailing junk; keep the first object.
        try:
            value, _ = json.JSONDecoder().raw_decode(raw)
            return value if isinstance(value, dict) else {"_raw": raw}
        except json.JSONDecodeError:
            return {"_raw": raw, "_error": "Arguments were not valid JSON."}


def to_openai_messages(system: str, messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for m in messages:
        if m.role == "user":
            out.append({"role": "user", "content": m.content})
        elif m.role == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": m.content or None}
            if m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in m.tool_calls
                ]
            if msg["content"] is None and not m.tool_calls:
                msg["content"] = ""
            out.append(msg)
        elif m.role == "tool":
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
    return out


def to_openai_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
        for t in tools
    ]


class OpenAICompatibleProvider:
    def __init__(
        self,
        *,
        name: str,
        model: str,
        api_key: str | None,
        base_url: str | None,
        temperature: float,
        max_tokens: int,
        timeout: float,
        max_retries: int,
    ) -> None:
        self.name = name
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = AsyncOpenAI(
            api_key=api_key or "missing", base_url=base_url, timeout=timeout, max_retries=max_retries
        )

    def _request_kwargs(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": to_openai_messages(system, messages),
            "stream": True,
        }
        if tools:
            kwargs["tools"] = to_openai_tools(tools)
            kwargs["tool_choice"] = "auto"
        if self.name == "openai":
            kwargs["max_completion_tokens"] = self.max_tokens
            kwargs["stream_options"] = {"include_usage": True}
            if not self.model.startswith(_NO_TEMPERATURE_PREFIXES):
                kwargs["temperature"] = self.temperature
        else:
            kwargs["max_tokens"] = self.max_tokens
            kwargs["temperature"] = self.temperature
            if self.name in ("groq", "openrouter"):
                kwargs["stream_options"] = {"include_usage": True}
        return kwargs

    async def stream(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        context: AgentContext | None = None,
    ) -> AsyncIterator[LLMEvent]:
        kwargs = self._request_kwargs(system, messages, tools)
        partial: dict[int, dict[str, str]] = {}
        usage = Usage()
        stop_reason: str | None = None
        try:
            stream = await self._client.chat.completions.create(**kwargs)
            async for chunk in stream:
                if getattr(chunk, "usage", None):
                    u = chunk.usage
                    usage.input_tokens = u.prompt_tokens or 0
                    usage.output_tokens = u.completion_tokens or 0
                    details = getattr(u, "prompt_tokens_details", None)
                    usage.cached_input_tokens = getattr(details, "cached_tokens", 0) or 0
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                if delta is not None:
                    if delta.content:
                        yield LLMEvent(type="text", text=delta.content)
                    for i, tc in enumerate(delta.tool_calls or []):
                        idx = tc.index if tc.index is not None else i
                        slot = partial.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                        if tc.id:
                            slot["id"] = tc.id
                        if tc.function is not None:
                            if tc.function.name:
                                slot["name"] += tc.function.name
                            if tc.function.arguments:
                                slot["arguments"] += tc.function.arguments
                if choice.finish_reason:
                    stop_reason = choice.finish_reason
        except openai.AuthenticationError as exc:
            raise LLMError(f"{self.name} rejected the API key (401). Check your {self.name.upper()}_API_KEY.") from exc
        except openai.RateLimitError as exc:
            raise LLMError(f"{self.name} rate limit or quota exceeded (429). Try again shortly.") from exc
        except openai.NotFoundError as exc:
            raise LLMError(f"{self.name} doesn't recognise model '{self.model}'. Set LLM_MODEL to a valid model.") from exc
        except openai.APITimeoutError as exc:
            raise LLMError(f"{self.name} timed out.") from exc
        except openai.APIConnectionError as exc:
            raise LLMError(f"Couldn't reach {self.name}: {exc}") from exc
        except openai.APIStatusError as exc:
            raise LLMError(f"{self.name} returned an error ({exc.status_code}): {str(exc)[:300]}") from exc

        for idx in sorted(partial):
            slot = partial[idx]
            if not slot["name"]:
                continue
            yield LLMEvent(
                type="tool_call",
                tool_call=ToolCall(
                    id=slot["id"] or f"call_{uuid.uuid4().hex[:10]}",
                    name=slot["name"],
                    arguments=parse_arguments(slot["arguments"]),
                ),
            )
        yield LLMEvent(type="usage", usage=usage)
        yield LLMEvent(type="done", stop_reason=stop_reason)
