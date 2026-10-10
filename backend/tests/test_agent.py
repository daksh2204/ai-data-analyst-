"""The tool-calling agent loop, driven by a scripted LLM so every branch is deterministic."""

from __future__ import annotations

from typing import Any

import pytest

from app.agent.agent import AnalystAgent
from app.agent.llm.offline import OfflineProvider
from app.agent.tools import TOOL_SPECS, ToolExecutor
from app.config import Settings
from app.core.cache import TTLCache
from app.sessions.store import Session
from tests.conftest import ScriptedProvider, call, last_tool_result, text


def make_agent(session: Session, provider: Any, *, fallback: Any = None, **overrides: Any) -> AnalystAgent:
    settings = Settings(llm_provider="offline", **overrides)
    return AnalystAgent(
        session,
        provider,
        settings,
        fallback=fallback,
        answer_cache=TTLCache("answers", 32, 60),
        tool_cache=TTLCache("queries", 32, 60),
    )


async def collect(agent: AnalystAgent, question: str) -> list[dict[str, Any]]:
    return [ev async for ev in agent.run(question)]


REGION_SQL = "SELECT region, SUM(revenue) AS revenue FROM sales_orders GROUP BY 1 ORDER BY 2 DESC"


async def test_happy_path_query_chart_answer(session: Session):
    provider = ScriptedProvider([
        [text("Let me total revenue by region."),
         call("run_sql", "c1", query=REGION_SQL, purpose="Total revenue per region", display=True, title="Revenue by region")],
        lambda msgs: [call("create_chart", "c2", chart_type="bar", result_id=last_tool_result(msgs)["result_id"],
                           x="region", y=["revenue"], title="Revenue by region", purpose="Compare regions")],
        [text("**East** generated the most revenue.\n\n**How I got this**\n- Summed revenue by region.")],
    ])
    events = await collect(make_agent(session, provider), "Which region generated the highest revenue?")
    types = [e["type"] for e in events]
    assert types[0] == "start" and types[-1] == "final"
    assert types.count("step_start") == 2 and types.count("step_end") == 2
    assert "token" in types and "thought" in types

    final = events[-1]["message"]
    assert final["answer"].startswith("**East**")
    tool_steps = [s for s in final["steps"] if s["kind"] == "tool"]
    assert [s["tool"] for s in tool_steps] == ["run_sql", "create_chart"]
    assert all(s["status"] == "ok" for s in tool_steps)
    assert tool_steps[0]["purpose"] == "Total revenue per region"
    assert {a["kind"] for a in final["artifacts"]} == {"table", "chart"}
    assert final["code"][0]["language"] == "sql"
    assert final["usage"]["input_tokens"] == 300 and final["usage"]["llm_calls"] == 3

    # The SQL result the model saw is real data.
    sql_result = [m for m in provider.requests[1]["messages"] if m.role == "tool"][-1]
    assert '"East"' in sql_result.content
    # Tools are offered to the model, and the schema is in the system prompt.
    assert set(provider.requests[0]["tools"]) == {t.name for t in TOOL_SPECS}
    assert "sales_orders" in provider.requests[0]["system"]
    assert "sales_orders.customer_id = customers.customer_id" in provider.requests[0]["system"]


async def test_sql_errors_are_fed_back_so_the_model_can_self_correct(session: Session):
    provider = ScriptedProvider([
        [call("run_sql", "bad", query="SELECT regon, SUM(revenue) FROM sales_orders GROUP BY 1", purpose="try")],
        [call("run_sql", "good", query=REGION_SQL, purpose="fixed column name")],
        [text("East leads.")],
    ])
    events = await collect(make_agent(session, provider), "Top region?")
    final = events[-1]["message"]
    steps = [s for s in final["steps"] if s["kind"] == "tool"]
    assert [s["status"] for s in steps] == ["error", "ok"]
    error_msg = [m for m in provider.requests[1]["messages"] if m.role == "tool"][-1]
    assert error_msg.is_error and "regon" in error_msg.content


async def test_unsafe_sql_is_refused_as_a_tool_error(session: Session):
    provider = ScriptedProvider([
        [call("run_sql", "x", query="DROP TABLE sales_orders", purpose="oops")],
        [text("I can't modify data.")],
    ])
    events = await collect(make_agent(session, provider), "delete everything")
    assert events[-1]["message"]["steps"][0]["status"] == "error"
    assert "sales_orders" in session.db.tables()


async def test_conversation_context_carries_into_follow_ups(session: Session):
    agent_provider = ScriptedProvider([
        [call("run_sql", "c1", query=REGION_SQL, purpose="by region")],
        [text("East is the top region.")],
        [text("Broken down by month, East peaks in November.")],
    ])
    agent = make_agent(session, agent_provider)
    await collect(agent, "Which region generated the highest revenue?")
    await collect(make_agent(session, agent_provider), "Now break that down by month")
    follow_up_request = agent_provider.requests[-1]["messages"]
    contents = [m.content for m in follow_up_request]
    assert "Which region generated the highest revenue?" in contents
    assert "East is the top region." in contents
    assert any(m.role == "tool" for m in follow_up_request)  # previous tool results are replayed
    assert contents[-1] == "Now break that down by month"
    assert len(session.history) == 2


async def test_step_budget_forces_a_final_answer(session: Session):
    looping = [[call("run_sql", f"c{i}", query="SELECT 1 AS one", purpose="again")] for i in range(10)]
    provider = ScriptedProvider([*looping[:2], [text("Final answer from what I have.")]])
    events = await collect(make_agent(session, provider, agent_max_steps=2), "loop forever")
    final = events[-1]["message"]
    assert final["answer"] == "Final answer from what I have."
    assert provider.requests[-1]["tools"] == []  # no tools offered on the last round


async def test_llm_outage_falls_back_to_the_offline_engine(session: Session):
    broken = ScriptedProvider([], fail_with="Provider returned 503.")
    events = await collect(make_agent(session, broken, fallback=OfflineProvider()), "Which region generated the highest revenue?")
    notice = next(e for e in events if e["type"] == "notice")
    assert "offline" in notice["message"]
    final = events[-1]["message"]
    assert final["provider"] == "offline"
    assert "East" in final["answer"]


async def test_llm_outage_without_fallback_reports_an_error(session: Session):
    broken = ScriptedProvider([], fail_with="Invalid API key.")
    events = await collect(make_agent(session, broken, llm_fallback_to_offline=False, fallback=OfflineProvider()), "hi")
    assert events[-1]["type"] == "error"
    assert "Invalid API key" in events[-1]["message"]


async def test_identical_questions_are_answered_from_cache(session: Session):
    provider = ScriptedProvider([[call("run_sql", "c1", query=REGION_SQL, purpose="p")], [text("East.")]])
    cache = TTLCache("answers", 32, 60)
    settings = Settings(llm_provider="offline")
    tool_cache = TTLCache("q", 32, 60)
    a1 = AnalystAgent(session, provider, settings, fallback=None, answer_cache=cache, tool_cache=tool_cache)
    await collect(a1, "Top region?")
    session.history.clear()  # same context as the first ask
    a2 = AnalystAgent(session, provider, settings, fallback=None, answer_cache=cache, tool_cache=tool_cache)
    events = await collect(a2, "top region")
    assert events[-1]["message"]["cached"] is True
    assert len(provider.requests) == 2  # no new LLM calls


def test_tool_executor_handles_bad_calls(session: Session):
    ex = ToolExecutor(session, max_rows_llm=10, pandas_timeout=5, cache=TTLCache("q", 8, 60))
    assert ex.execute("drop_database", {}).is_error
    assert ex.execute("run_sql", {"_raw": "{not json"}).is_error
    assert ex.execute("create_chart", {"chart_type": "bar", "result_id": "r999", "x": "a", "title": "t"}).is_error
    out = ex.execute("profile_table", {"table": "customers", "purpose": "check"})
    assert not out.is_error and out.artifacts[0]["kind"] == "quality"
    out = ex.execute("run_pandas", {"code": "result = sales_orders.groupby('region')['revenue'].sum()", "purpose": "p"})
    assert not out.is_error and out.result_id
    out = ex.execute("forecast", {"table": "sales_orders", "date_column": "order_date", "value_column": "revenue",
                                  "periods": 3, "frequency": "M", "where": "region = 'West'", "purpose": "p"})
    assert not out.is_error and out.artifacts[0]["kind"] == "forecast"


# --------------------------------------------------------------------------- offline engine accuracy


def ground_truth(session: Session, sql: str) -> list[tuple]:
    return list(session.db.execute(sql).df.itertuples(index=False, name=None))


@pytest.mark.parametrize(
    ("question", "sql", "expect_in_answer"),
    [
        ("Which region generated the highest revenue?",
         "SELECT region FROM sales_orders GROUP BY 1 ORDER BY SUM(revenue) DESC LIMIT 1", None),
        ("What are the top five customers?",
         "SELECT c.customer_name FROM sales_orders o JOIN customers c USING (customer_id) "
         "GROUP BY 1 ORDER BY SUM(o.revenue) DESC LIMIT 1", None),
        ("Which products are underperforming?", "SELECT 'Legacy CRM Desktop'", None),
        ("Show monthly sales trends.", "SELECT 'year over year'", None),
        ("Detect anomalies in the dataset.", "SELECT '2024-11-29'", None),
        ("Which regions missed their revenue targets?", "SELECT 'of target'", None),
    ],
)
async def test_offline_engine_answers_the_assignment_questions(session: Session, question, sql, expect_in_answer):
    expected = str(ground_truth(session, sql)[0][0])
    events = await collect(make_agent(session, OfflineProvider()), question)
    final = events[-1]["message"]
    assert expected.lower() in final["answer"].lower(), final["answer"]
    assert "How I got this" in final["answer"]
    assert all(s["status"] == "ok" for s in final["steps"] if s["kind"] == "tool")


async def test_offline_engine_generates_runnable_sql_and_pandas_for_the_previous_question(session: Session):
    agent = make_agent(session, OfflineProvider())
    await collect(agent, "What are the top five customers?")
    events = await collect(make_agent(session, OfflineProvider()), "Generate SQL and pandas for this analysis.")
    answer = events[-1]["message"]["answer"]
    assert "```sql" in answer and "```python" in answer
    sql = answer.split("```sql\n", 1)[1].split("```", 1)[0]
    top_sql = session.db.execute(sql).df
    assert len(top_sql) == 5
    from app.analysis.pandas_sandbox import result_to_frame, run_pandas

    code = answer.split("```python\n", 1)[1].split("```", 1)[0]
    frames = {n: t.df for n, t in session.tables.items()}
    top_pd = result_to_frame(run_pandas(code, frames).value)
    assert top_pd is not None
    assert list(top_pd.iloc[:, 0]) == list(top_sql.iloc[:, 0])


async def test_offline_engine_treats_a_new_question_with_code_request_as_new(session: Session):
    await collect(make_agent(session, OfflineProvider()), "Which products are underperforming?")
    events = await collect(make_agent(session, OfflineProvider()), "What are the top five customers? Show the SQL.")
    answer = events[-1]["message"]["answer"]
    assert "customer" in answer.lower() and "```sql" in answer
