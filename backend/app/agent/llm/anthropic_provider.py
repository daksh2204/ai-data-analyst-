"""Adapter for Anthropic's Messages API (native tool use, streaming, prompt caching)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import anthropic
from anthropic import AsyncAnthropic

from app.agent.llm.base import AgentContext, LLMEvent, Message, ToolCall, ToolSpec, Usage
from app.core.errors import LLMError


def to_anthropic_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert neutral messages; consecutive tool results are grouped into one user turn."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "user":
            out.append({"role": "user", "content": [{"type": "text", "text": m.content or " "}]})
        elif m.role == "assistant":
            blocks: list[dict[str, Any]] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            for tc in m.tool_calls:
                blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments})
            if not blocks:
                blocks.append({"type": "text", "text": " "})
            out.append({"role": "assistant", "content": blocks})
        elif m.role == "tool":
            block = {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content, "is_error": m.is_error}
            if out and out[-1]["role"] == "user" and all(b.get("type") == "tool_result" for b in out[-1]["content"]):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
    # The API requires alternation; merge accidental consecutive user turns.
    merged: list[dict[str, Any]] = []
    for msg in out:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"].extend(msg["content"])
        else:
            merged.append(msg)
    return merged


def to_anthropic_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    out = [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools]
    if out:
        out[-1]["cache_control"] = {"type": "ephemeral"}  # cache the (static) tool block
    return out


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, *, model: str, api_key: str | None, max_tokens: int, timeout: float, max_retries: int) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._client = AsyncAnthropic(api_key=api_key or "missing", timeout=timeout, max_retries=max_retries)

    async def stream(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        context: AgentContext | None = None,
    ) -> AsyncIterator[LLMEvent]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": to_anthropic_messages(messages),
        }
        if tools:
            kwargs["tools"] = to_anthropic_tools(tools)
        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    if event.type == "text" and event.text:
                        yield LLMEvent(type="text", text=event.text)
                final = await stream.get_final_message()
        except anthropic.AuthenticationError as exc:
            raise LLMError("Anthropic rejected the API key (401). Check ANTHROPIC_API_KEY.") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("Anthropic rate limit exceeded (429). Try again shortly.") from exc
        except anthropic.NotFoundError as exc:
            raise LLMError(f"Anthropic doesn't recognise model '{self.model}'. Set LLM_MODEL to a valid model.") from exc
        except anthropic.APITimeoutError as exc:
            raise LLMError("Anthropic timed out.") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"Couldn't reach Anthropic: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic returned an error ({exc.status_code}): {str(exc)[:300]}") from exc

        for block in final.content:
            if block.type == "tool_use":
                args = block.input if isinstance(block.input, dict) else {"_raw": str(block.input)}
                yield LLMEvent(type="tool_call", tool_call=ToolCall(id=block.id, name=block.name, arguments=args))
        u = final.usage
        yield LLMEvent(
            type="usage",
            usage=Usage(
                input_tokens=(u.input_tokens or 0) + (getattr(u, "cache_read_input_tokens", 0) or 0)
                + (getattr(u, "cache_creation_input_tokens", 0) or 0),
                output_tokens=u.output_tokens or 0,
                cached_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            ),
        )
        yield LLMEvent(type="done", stop_reason=final.stop_reason)
