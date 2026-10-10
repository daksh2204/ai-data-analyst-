"""The two places LLM-written code runs: the read-only SQL guard and the pandas sandbox."""

from __future__ import annotations

import pandas as pd
import pytest

from app.analysis.pandas_sandbox import result_to_frame, run_pandas
from app.core.errors import QueryError, QueryTimeoutError, UnsafeCodeError
from app.data.engine import SessionDatabase


@pytest.fixture
def db() -> SessionDatabase:
    d = SessionDatabase(timeout_seconds=2, max_rows=100)
    d.register_table("sales", pd.DataFrame({"region": ["A", "B", "A"], "revenue": [10.0, 20.0, 5.0]}))
    return d


# --------------------------------------------------------------------------- SQL guard


def test_select_and_cte_queries_run(db: SessionDatabase):
    res = db.execute("SELECT region, SUM(revenue) AS r FROM sales GROUP BY 1 ORDER BY 2 DESC")
    assert res.df.to_dict("records") == [{"region": "B", "r": 20.0}, {"region": "A", "r": 15.0}]
    res = db.execute("WITH t AS (SELECT * FROM sales) SELECT COUNT(*) AS n FROM t;")
    assert res.df["n"].iloc[0] == 3


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE sales",
        "DELETE FROM sales",
        "INSERT INTO sales VALUES ('C', 1)",
        "UPDATE sales SET revenue = 0",
        "CREATE TABLE x AS SELECT 1",
        "ATTACH 'other.db' AS o",
        "COPY sales TO 'out.csv'",
        "SET enable_external_access = true",
        "INSTALL httpfs",
        "SELECT 1; DROP TABLE sales",
    ],
)
def test_non_select_statements_are_rejected(db: SessionDatabase, sql: str):
    with pytest.raises(UnsafeCodeError):
        db.execute(sql)
    assert db.execute("SELECT COUNT(*) AS n FROM sales").df["n"].iloc[0] == 3  # data untouched


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM read_csv('/etc/passwd')",
        "SELECT * FROM '/etc/hostname'",
        "SELECT * FROM read_text('/etc/passwd')",
    ],
)
def test_file_system_access_is_blocked(db: SessionDatabase, sql: str):
    with pytest.raises(QueryError):
        db.execute(sql)


def test_sql_errors_are_readable_for_self_correction(db: SessionDatabase):
    with pytest.raises(QueryError) as exc:
        db.execute("SELECT revenu FROM sales")
    assert "revenu" in exc.value.message


def test_results_are_capped(db: SessionDatabase):
    res = db.execute("SELECT * FROM range(1000)")
    assert res.row_count == 100 and res.truncated


def test_long_queries_time_out():
    d = SessionDatabase(timeout_seconds=0.5, max_rows=10)
    with pytest.raises(QueryTimeoutError):
        d.execute("SELECT COUNT(*) FROM range(1000000000) a, range(1000000) b")


# --------------------------------------------------------------------------- pandas sandbox

FRAMES = {"sales": pd.DataFrame({"region": ["A", "B", "A"], "revenue": [10.0, 20.0, 5.0]})}


def test_vectorised_pandas_runs_and_returns_a_frame():
    out = run_pandas('result = sales.groupby("region")["revenue"].sum().sort_values()', FRAMES)
    frame = result_to_frame(out.value)
    assert frame is not None
    assert frame.to_dict("records") == [{"region": "A", "revenue": 15.0}, {"region": "B", "revenue": 20.0}]


def test_trailing_expression_becomes_result():
    out = run_pandas("sales['revenue'].mean()", FRAMES)
    assert out.value == pytest.approx(35 / 3)


@pytest.mark.parametrize(
    "code",
    [
        "import os\nresult = 1",
        "from subprocess import run\nresult = 1",
        "result = open('/etc/passwd').read()",
        "result = __import__('os')",
        "result = sales.__class__",
        "result = ().__class__.__bases__[0].__subclasses__()",
        "result = getattr(sales, 'to_csv')",
        "sales.to_csv('/tmp/leak.csv')\nresult = 1",
        "result = pd.read_csv('/etc/passwd')",
        "result = sales.query('revenue > 5')",
        "result = eval('1+1')",
        "while True:\n    pass",
        "result = pd.io.common",
    ],
)
def test_unsafe_code_is_rejected_before_running(code: str):
    with pytest.raises(UnsafeCodeError):
        run_pandas(code, FRAMES)


def test_runaway_loops_are_stopped():
    with pytest.raises((QueryTimeoutError, UnsafeCodeError)):
        run_pandas("total = 0\nfor i in range(10**6):\n    for j in range(10**6):\n        total += 1\nresult = total", FRAMES, timeout=1)


def test_session_data_cannot_be_mutated():
    run_pandas("sales['revenue'] = 0\nresult = 1", FRAMES)
    assert FRAMES["sales"]["revenue"].sum() == 35


def test_runtime_errors_are_reported_back():
    with pytest.raises(QueryError, match="KeyError"):
        run_pandas("result = sales['missing']", FRAMES)
    with pytest.raises(QueryError, match="result"):
        run_pandas("x = 1", FRAMES)
