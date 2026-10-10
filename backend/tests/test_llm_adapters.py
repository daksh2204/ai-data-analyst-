"""Provider adapters: message conversion, argument parsing, provider selection."""

from __future__ import annotations

from app.agent.llm.anthropic_provider import to_anthropic_messages, to_anthropic_tools
from app.agent.llm.base import Message, ToolCall
from app.agent.llm.factory import create_provider
from app.agent.llm.openai_compat import parse_arguments, to_openai_messages
from app.agent.memory import Turn, build_history
from app.agent.tools import TOOL_SPECS
from app.config import Settings

CONVERSATION = [
    Message(role="user", content="Top region?"),
    Message(
        role="assistant",
        content="Checking.",
        tool_calls=[
            ToolCall(id="t1", name="run_sql", arguments={"query": "SELECT 1"}),
            ToolCall(id="t2", name="run_sql", arguments={"query": "SELECT 2"}),
        ],
    ),
    Message(role="tool", content='{"rows": [[1]]}', tool_call_id="t1", name="run_sql"),
    Message(role="tool", content='{"error": "x"}', tool_call_id="t2", name="run_sql", is_error=True),
    Message(role="assistant", content="East."),
]


def test_anthropic_conversion_groups_parallel_tool_results():
    out = to_anthropic_messages(CONVERSATION)
    assert [m["role"] for m in out] == ["user", "assistant", "user", "assistant"]
    assert [b["type"] for b in out[1]["content"]] == ["text", "tool_use", "tool_use"]
    results = out[2]["content"]
    assert [b["tool_use_id"] for b in results] == ["t1", "t2"]
    assert results[1]["is_error"] is True
    tools = to_anthropic_tools(TOOL_SPECS)
    assert tools[-1]["cache_control"] == {"type": "ephemeral"}  # prompt caching on the static tool block


def test_openai_conversion_keeps_tool_call_ids():
    out = to_openai_messages("system prompt", CONVERSATION)
    assert out[0] == {"role": "system", "content": "system prompt"}
    assert out[2]["tool_calls"][1]["function"]["name"] == "run_sql"
    assert [m.get("tool_call_id") for m in out if m["role"] == "tool"] == ["t1", "t2"]


def test_argument_parsing_is_forgiving():
    assert parse_arguments('{"a": 1}') == {"a": 1}
    assert parse_arguments("") == {}
    assert parse_arguments('{"a": 1}{"b": 2}') == {"a": 1}
    assert "_error" in parse_arguments("{not json")


def test_provider_auto_selection():
    assert Settings(llm_provider="auto", _env_file=None).resolved_provider() == "offline"
    s = Settings(llm_provider="auto", gemini_api_key="g", _env_file=None)
    assert s.resolved_provider() == "gemini"
    assert s.base_url_for("gemini").startswith("https://generativelanguage.googleapis.com")
    s = Settings(llm_provider="auto", anthropic_api_key="a", openai_api_key="o", _env_file=None)
    assert s.resolved_provider() == "anthropic"
    p = create_provider(Settings(llm_provider="groq", groq_api_key="k", llm_model="m", _env_file=None))
    assert (p.name, p.model) == ("groq", "m")


def test_memory_compacts_old_turns_and_respects_the_budget():
    turns = []
    for i in range(10):
        msgs = [
            Message(role="assistant", content="", tool_calls=[ToolCall(id=f"t{i}", name="run_sql", arguments={"query": f"SELECT {i}"})]),
            Message(role="tool", content="x" * 10_000, tool_call_id=f"t{i}", name="run_sql"),
            Message(role="assistant", content=f"answer {i}"),
        ]
        turns.append(Turn(question=f"q{i}", messages=msgs, answer=f"answer {i}", code=[{"language": "sql", "code": f"SELECT {i}"}]))
    history = build_history(turns, keep_verbatim=2)
    # Old turns: question + compact answer (with the SQL used); recent turns verbatim but truncated.
    assert history[0].content == "q0"
    assert "SELECT 0" in history[1].content
    tool_msgs = [m for m in history if m.role == "tool"]
    assert len(tool_msgs) == 2 and all(len(m.content) < 3000 for m in tool_msgs)
    assert history[-1].content == "answer 9"
