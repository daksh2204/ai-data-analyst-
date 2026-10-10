"""Contract tests: the real provider adapters against recorded-format streaming responses.

The official SDKs run unmodified; only the HTTP transport is replaced, so these tests prove
that streamed text, streamed tool-call arguments and usage are parsed exactly as the
OpenAI-compatible and Anthropic wire formats deliver them, end to end through the agent.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import httpx2  # the Anthropic SDK ships on httpx2; the OpenAI SDK still uses httpx
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from app.agent.agent import AnalystAgent
from app.agent.llm.anthropic_provider import AnthropicProvider
from app.agent.llm.openai_compat import OpenAICompatibleProvider
from app.config import Settings
from app.core.cache import TTLCache
from app.sessions.store import Session

SQL = "SELECT region, SUM(revenue) AS revenue FROM sales_orders GROUP BY 1 ORDER BY 2 DESC LIMIT 1"


def sse(events: list[tuple[str | None, dict[str, Any] | str]]) -> bytes:
    out = []
    for name, data in events:
        payload = data if isinstance(data, str) else json.dumps(data)
        out.append((f"event: {name}\n" if name else "") + f"data: {payload}\n\n")
    return "".join(out).encode()


async def run_agent(session: Session, provider: Any) -> dict[str, Any]:
    agent = AnalystAgent(session, provider, Settings(llm_provider="offline"), fallback=None,
                         answer_cache=TTLCache("a", 4, 60), tool_cache=TTLCache("q", 4, 60))
    events = [e async for e in agent.run("Which region generated the highest revenue?")]
    assert events[-1]["type"] == "final", events[-1]
    return events[-1]["message"]


# --------------------------------------------------------------------------- OpenAI-compatible


def openai_chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
    return {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


async def test_openai_compatible_stream_with_tool_calls(session: Session):
    requests: list[dict[str, Any]] = []
    args = json.dumps({"query": SQL, "purpose": "Rank regions by revenue"})

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            events: list[tuple[str | None, Any]] = [
                (None, openai_chunk({"role": "assistant", "content": "Checking regions. "})),
                (None, openai_chunk({"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                                     "function": {"name": "run_sql", "arguments": ""}}]})),
                (None, openai_chunk({"tool_calls": [{"index": 0, "function": {"arguments": args[:25]}}]})),
                (None, openai_chunk({"tool_calls": [{"index": 0, "function": {"arguments": args[25:]}}]})),
                (None, openai_chunk({}, "tool_calls")),
                (None, {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "m", "choices": [],
                        "usage": {"prompt_tokens": 900, "completion_tokens": 40, "total_tokens": 940}}),
                (None, "[DONE]"),
            ]
        else:
            tool_msg = next(m for m in body["messages"] if m["role"] == "tool")
            region = json.loads(tool_msg["content"])["rows"][0][0]
            events = [
                (None, openai_chunk({"role": "assistant", "content": f"**{region}** "})),
                (None, openai_chunk({"content": "generated the most revenue."})),
                (None, openai_chunk({}, "stop")),
                (None, "[DONE]"),
            ]
        return httpx.Response(200, content=sse(events), headers={"content-type": "text/event-stream"})

    provider = OpenAICompatibleProvider(name="openai", model="gpt-test", api_key="k", base_url=None,
                                        temperature=0.1, max_tokens=500, timeout=10, max_retries=0)
    provider._client = AsyncOpenAI(api_key="k", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    message = await run_agent(session, provider)

    assert message["answer"] == "**East** generated the most revenue."
    step = next(s for s in message["steps"] if s["kind"] == "tool")
    assert step["tool"] == "run_sql" and step["status"] == "ok" and step["purpose"] == "Rank regions by revenue"
    assert message["usage"]["input_tokens"] == 900
    first = requests[0]
    assert first["stream"] is True and first["tools"][0]["type"] == "function"
    assert first["messages"][0]["role"] == "system"
    # The second request replays the assistant tool call and the tool result with matching ids.
    second = requests[1]["messages"]
    assert second[-2]["tool_calls"][0]["id"] == "call_1"
    assert second[-1]["role"] == "tool" and second[-1]["tool_call_id"] == "call_1"


# --------------------------------------------------------------------------- Anthropic


def anthropic_tool_round(args: str) -> bytes:
    return sse([
        ("message_start", {"type": "message_start", "message": {
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-test", "content": [],
            "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 1200, "output_tokens": 1, "cache_read_input_tokens": 800}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Let me check. "}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("content_block_start", {"type": "content_block_start", "index": 1, "content_block": {
            "type": "tool_use", "id": "toolu_1", "name": "run_sql", "input": {}}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": args[:30]}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": args[30:]}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 1}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                           "usage": {"output_tokens": 60}}),
        ("message_stop", {"type": "message_stop"}),
    ])


def anthropic_text_round(text: str) -> bytes:
    return sse([
        ("message_start", {"type": "message_start", "message": {
            "id": "msg_2", "type": "message", "role": "assistant", "model": "claude-test", "content": [],
            "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 1500, "output_tokens": 1}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                           "usage": {"output_tokens": 30}}),
        ("message_stop", {"type": "message_stop"}),
    ])


async def test_anthropic_stream_with_tool_use(session: Session):
    requests: list[dict[str, Any]] = []
    args = json.dumps({"query": SQL, "purpose": "Rank regions by revenue"})

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            content = anthropic_tool_round(args)
        else:
            result_block = body["messages"][-1]["content"][0]
            assert result_block["type"] == "tool_result" and result_block["tool_use_id"] == "toolu_1"
            region = json.loads(result_block["content"])["rows"][0][0]
            content = anthropic_text_round(f"**{region}** leads on revenue.")
        return httpx2.Response(200, content=content, headers={"content-type": "text/event-stream"})

    provider = AnthropicProvider(model="claude-test", api_key="k", max_tokens=500, timeout=10, max_retries=0)
    provider._client = AsyncAnthropic(api_key="k", http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))
    message = await run_agent(session, provider)

    assert message["answer"] == "**East** leads on revenue."
    step = next(s for s in message["steps"] if s["kind"] == "tool")
    assert step["status"] == "ok" and step["input"]["query"] == SQL
    assert message["usage"]["cached_input_tokens"] == 800
    first = requests[0]
    assert first["stream"] is True
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert {t["name"] for t in first["tools"]} >= {"run_sql", "create_chart", "detect_anomalies", "forecast"}


async def test_provider_http_errors_become_readable_llm_errors(session: Session):
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(401, json={"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}})

    provider = AnthropicProvider(model="claude-test", api_key="bad", max_tokens=100, timeout=5, max_retries=0)
    provider._client = AsyncAnthropic(api_key="bad", max_retries=0,
                                      http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))
    agent = AnalystAgent(session, provider, Settings(llm_provider="offline", llm_fallback_to_offline=False),
                         fallback=None, answer_cache=TTLCache("a", 4, 60), tool_cache=TTLCache("q", 4, 60))
    events = [e async for e in agent.run("hi")]
    assert events[-1]["type"] == "error"
    assert "ANTHROPIC_API_KEY" in events[-1]["message"]
