"""The analyst's tools: JSON-schema specs for the LLM plus their implementations.

Every tool takes a ``purpose`` argument — one sentence on why the step is needed. It is
shown to the user as the step's reasoning, which makes the agent's work auditable
without exposing raw chain-of-thought.

Tool results returned to the LLM are compact JSON (rows as arrays, capped length). The
full results stay server-side and are referenced by ``result_id``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.agent.llm.base import ToolSpec
from app.analysis.anomalies import detect_anomalies
from app.analysis.charts import CHART_TYPES, ChartSpec, _clean, build_chart, pretty
from app.analysis.forecasting import forecast_series
from app.analysis.pandas_sandbox import result_to_frame, run_pandas
from app.core.cache import TTLCache, make_key
from app.core.errors import AppError, NotFoundError, ToolError
from app.sessions.store import Session

PURPOSE = {
    "type": "string",
    "description": "One short sentence explaining why this step is needed. Shown to the user as your reasoning.",
}

TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        name="run_sql",
        description=(
            "Run one read-only DuckDB SQL SELECT query over the uploaded tables. Use it for aggregations, "
            "filters, joins, rankings and time bucketing (date_trunc('month', col)). Returns columns, up to 40 "
            "rows and a result_id that create_chart can reuse. Quote identifiers with double quotes if needed."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A single DuckDB SELECT statement."},
                "purpose": PURPOSE,
                "display": {
                    "type": "boolean",
                    "description": "Show this result to the user as a table in the answer (use for top-N lists, comparisons).",
                },
                "title": {"type": "string", "description": "Title for the displayed table."},
            },
            "required": ["query", "purpose"],
        },
    ),
    ToolSpec(
        name="run_pandas",
        description=(
            "Run sandboxed pandas code when SQL is awkward (correlations, pivots, statistical tests, reshaping). "
            "Each table is available as a DataFrame named after it (also in the dict `dfs`); `pd` and `np` are "
            "imported. Assign the output to `result` (DataFrame, Series, scalar or dict). Imports, file access, "
            "while-loops, eval/query and dunder access are blocked."
        ),
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python/pandas code that assigns `result`."},
                "purpose": PURPOSE,
                "display": {"type": "boolean", "description": "Show the result to the user as a table."},
                "title": {"type": "string", "description": "Title for the displayed table."},
            },
            "required": ["code", "purpose"],
        },
    ),
    ToolSpec(
        name="create_chart",
        description=(
            "Create a chart shown to the user. Source data is either a previous result (result_id) or a new SQL "
            "query (sql). Pick the form by the data's job: line/area for change over time, bar or horizontal_bar "
            "for comparing categories (horizontal for long labels or ranked lists), stacked_bar/grouped_bar for a "
            "category split by a second dimension, pie/donut only for part-of-whole with <= 6 slices, scatter for "
            "the relationship between two measures, histogram/box for distributions, heatmap for a matrix."
        ),
        parameters={
            "type": "object",
            "properties": {
                "chart_type": {"type": "string", "enum": list(CHART_TYPES)},
                "result_id": {"type": "string", "description": "id returned by run_sql/run_pandas."},
                "sql": {"type": "string", "description": "Alternative to result_id: a SELECT returning the chart data."},
                "x": {"type": "string", "description": "Column for the x axis / categories / pie labels."},
                "y": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Value column(s). For heatmaps from long data: [row_column, value_column].",
                },
                "color": {"type": "string", "description": "Optional column that splits the data into series."},
                "title": {"type": "string", "description": "Short chart title in sentence case."},
                "x_label": {"type": "string"},
                "y_label": {"type": "string"},
                "sort": {"type": "string", "enum": ["asc", "desc"]},
                "top_n": {"type": "integer", "description": "Keep only the first N categories."},
                "purpose": PURPOSE,
            },
            "required": ["chart_type", "x", "title", "purpose"],
        },
    ),
    ToolSpec(
        name="detect_anomalies",
        description=(
            "Find and explain anomalies in a table: robust outliers per numeric column (scored within product "
            "groups for amounts), negative values where only positives make sense, and unusual days/weeks in the "
            "main measure over time. Returns each anomaly with a plain-language reason. Also renders a chart."
        ),
        parameters={
            "type": "object",
            "properties": {
                "table": {"type": "string"},
                "columns": {"type": "array", "items": {"type": "string"}, "description": "Numeric columns to check (default: all measures)."},
                "date_column": {"type": "string", "description": "Date column for time-series checks (default: auto)."},
                "value_column": {"type": "string", "description": "Measure to track over time (default: primary measure)."},
                "group_by": {"type": "string", "description": "Score values within these groups instead of globally."},
                "sensitivity": {"type": "number", "description": "Robust z threshold; 3.5 default, lower finds more."},
                "purpose": PURPOSE,
            },
            "required": ["table", "purpose"],
        },
    ),
    ToolSpec(
        name="forecast",
        description=(
            "Forecast a measure over time with Holt-Winters exponential smoothing (seasonality when there are two "
            "full seasons). Returns forecast values with 80%/95% intervals, a backtest error and a chart."
        ),
        parameters={
            "type": "object",
            "properties": {
                "table": {"type": "string"},
                "date_column": {"type": "string"},
                "value_column": {"type": "string", "description": "Measure to forecast; omit to forecast row counts."},
                "periods": {"type": "integer", "description": "How many periods ahead (default 6)."},
                "frequency": {"type": "string", "enum": ["auto", "D", "W", "M", "Q"]},
                "where": {"type": "string", "description": "Optional SQL filter, e.g. region = 'West'."},
                "purpose": PURPOSE,
            },
            "required": ["table", "date_column", "purpose"],
        },
    ),
    ToolSpec(
        name="profile_table",
        description="Get the full column profile and data-quality report of a table (types, nulls, stats, issues).",
        parameters={
            "type": "object",
            "properties": {"table": {"type": "string"}, "purpose": PURPOSE},
            "required": ["table", "purpose"],
        },
    ),
]


@dataclass
class ToolOutcome:
    content: str
    summary: str
    is_error: bool = False
    preview: dict[str, Any] | None = None
    code: dict[str, str] | None = None
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    result_id: str | None = None


def _compact_value(v: Any) -> Any:
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return round(v, 4) if abs(v) < 1e6 else round(v, 1)
    return v


def frame_payload(df: pd.DataFrame, max_rows: int) -> dict[str, Any]:
    rows = [[_compact_value(v) for v in _clean(r)] for r in df.head(max_rows).itertuples(index=False, name=None)]
    return {"columns": [str(c) for c in df.columns], "rows": rows, "row_count": len(df), "truncated": len(df) > max_rows}


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


class ToolExecutor:
    """Executes tool calls against one session. Synchronous; run via ``asyncio.to_thread``."""

    def __init__(self, session: Session, *, max_rows_llm: int, pandas_timeout: float, cache: TTLCache[Any]) -> None:
        self.session = session
        self.max_rows_llm = max_rows_llm
        self.pandas_timeout = pandas_timeout
        self.cache = cache
        self._handlers: dict[str, Callable[[dict[str, Any]], ToolOutcome]] = {
            "run_sql": self.run_sql,
            "run_pandas": self.run_pandas,
            "create_chart": self.create_chart,
            "detect_anomalies": self.detect_anomalies,
            "forecast": self.forecast,
            "profile_table": self.profile_table,
        }

    def execute(self, name: str, args: dict[str, Any]) -> ToolOutcome:
        handler = self._handlers.get(name)
        if handler is None:
            return ToolOutcome(
                content=json.dumps({"error": f"Unknown tool '{name}'. Available: {', '.join(self._handlers)}."}),
                summary=f"Unknown tool {name}",
                is_error=True,
            )
        if "_raw" in args:
            return ToolOutcome(
                content=json.dumps({"error": "Tool arguments were not valid JSON. Resend the call with a JSON object."}),
                summary="Invalid arguments",
                is_error=True,
            )
        try:
            return handler(args)
        except AppError as exc:
            return ToolOutcome(
                content=json.dumps({"error": exc.message, "hint": "Fix the problem and try again."}),
                summary=exc.message,
                is_error=True,
                code=self._code_for(name, args),
            )
        except (KeyError, ValueError, TypeError) as exc:
            return ToolOutcome(
                content=json.dumps({"error": f"{type(exc).__name__}: {exc}"}),
                summary=str(exc),
                is_error=True,
                code=self._code_for(name, args),
            )

    @staticmethod
    def _code_for(name: str, args: dict[str, Any]) -> dict[str, str] | None:
        if name == "run_sql" and args.get("query"):
            return {"language": "sql", "code": str(args["query"])}
        if name == "run_pandas" and args.get("code"):
            return {"language": "python", "code": str(args["code"])}
        if name == "create_chart" and args.get("sql"):
            return {"language": "sql", "code": str(args["sql"])}
        return None

    # ------------------------------------------------------------------ SQL
    def _sql(self, query: str):
        key = make_key("sql", self.session.fingerprint(), query.strip())
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        result = self.session.db.execute(query)
        self.cache.set(key, result)
        return result

    def run_sql(self, args: dict[str, Any]) -> ToolOutcome:
        query = str(args.get("query", "")).strip()
        if not query:
            raise ToolError("`query` is required.")
        res = self._sql(query)
        rs = self.session.store_result(res.df, "sql", res.sql, args.get("title"))
        payload = {"result_id": rs.id, **frame_payload(res.df, self.max_rows_llm)}
        if res.truncated:
            payload["note"] = f"Result capped at {len(res.df):,} rows. Aggregate further if you need all of it."
        artifacts = []
        if args.get("display"):
            artifacts.append(self._table_artifact(rs.id, res.df, args.get("title") or "Result"))
        return ToolOutcome(
            content=json.dumps(payload, default=str),
            summary=f"{len(res.df):,} row(s) × {len(res.df.columns)} column(s) in {res.elapsed_ms:.0f} ms",
            preview=frame_payload(res.df, 8),
            code={"language": "sql", "code": res.sql},
            artifacts=artifacts,
            result_id=rs.id,
        )

    def _table_artifact(self, result_id: str, df: pd.DataFrame, title: str) -> dict[str, Any]:
        return {"kind": "table", "id": f"t_{result_id}", "title": title, "result_id": result_id, **frame_payload(df, 200)}

    # ------------------------------------------------------------------ pandas
    def run_pandas(self, args: dict[str, Any]) -> ToolOutcome:
        code = str(args.get("code", ""))
        if not code.strip():
            raise ToolError("`code` is required.")
        frames = {name: t.df for name, t in self.session.tables.items()}
        out = run_pandas(code, frames, timeout=self.pandas_timeout)
        frame = result_to_frame(out.value)
        code_block = {"language": "python", "code": code}
        if frame is not None:
            rs = self.session.store_result(frame, "pandas", code, args.get("title"))
            payload: dict[str, Any] = {"result_id": rs.id, **frame_payload(frame, self.max_rows_llm)}
            if out.stdout:
                payload["stdout"] = out.stdout[:2000]
            artifacts = [self._table_artifact(rs.id, frame, args.get("title") or "Result")] if args.get("display") else []
            return ToolOutcome(
                content=json.dumps(payload, default=str),
                summary=f"{len(frame):,} row(s) × {len(frame.columns)} column(s) in {out.elapsed_ms:.0f} ms",
                preview=frame_payload(frame, 8),
                code=code_block,
                artifacts=artifacts,
                result_id=rs.id,
            )
        value = out.value
        if isinstance(value, (np.generic,)):
            value = value.item()
        text = value if isinstance(value, (int, float, str, bool, type(None))) else repr(value)[:3000]
        payload = {"value": _compact_value(text) if isinstance(text, float) else text}
        if out.stdout:
            payload["stdout"] = out.stdout[:2000]
        return ToolOutcome(
            content=json.dumps(payload, default=str),
            summary=f"Returned {type(out.value).__name__} in {out.elapsed_ms:.0f} ms",
            code=code_block,
        )

    # ------------------------------------------------------------------ charts
    def create_chart(self, args: dict[str, Any]) -> ToolOutcome:
        code = None
        if args.get("result_id"):
            rs = self.session.get_result(str(args["result_id"]))
            df, source_id, code = rs.df, rs.id, {"language": rs.source if rs.source == "sql" else "python", "code": rs.code}
        elif args.get("sql"):
            res = self._sql(str(args["sql"]))
            rs = self.session.store_result(res.df, "sql", res.sql, args.get("title"))
            df, source_id, code = res.df, rs.id, {"language": "sql", "code": res.sql}
        else:
            raise ToolError("Provide either `result_id` (from run_sql/run_pandas) or `sql`.")
        y = args.get("y") or []
        if isinstance(y, str):
            y = [y]
        spec = ChartSpec(
            chart_type=str(args.get("chart_type", "bar")),
            x=args.get("x"),
            y=[str(v) for v in y],
            color=args.get("color"),
            title=str(args.get("title") or ""),
            x_label=args.get("x_label"),
            y_label=args.get("y_label"),
            sort=args.get("sort"),
            top_n=int(args["top_n"]) if args.get("top_n") else None,
        )
        built = build_chart(df, spec)
        chart_id = self.session.next_id("c")
        artifact = {
            "kind": "chart",
            "id": chart_id,
            "title": spec.title or pretty(spec.x or "chart"),
            "chart_type": spec.chart_type,
            "figure": built["figure"],
            "notes": built["notes"],
            "result_id": source_id,
        }
        info = {"chart_id": chart_id, "rendered": True, "points": len(df), "notes": built["notes"]}
        return ToolOutcome(
            content=json.dumps(info),
            summary=f"{spec.chart_type.replace('_', ' ').capitalize()} chart: {artifact['title']}",
            code=code if args.get("sql") else None,
            artifacts=[artifact],
        )

    # ------------------------------------------------------------------ anomalies
    def detect_anomalies(self, args: dict[str, Any]) -> ToolOutcome:
        entry = self.session.get_table(str(args.get("table", "")))
        profile = entry.profile
        measures = [c for c in (args.get("columns") or []) if c in entry.df.columns] or [
            c.name for c in profile.by_role("measure")
        ]
        date_col = args.get("date_column") or (profile.primary_time.name if profile.primary_time else None)
        value_col = args.get("value_column") or (profile.primary_measure.name if profile.primary_measure else None)
        context = [c.name for c in profile.columns if c.role in ("identifier", "time")] + [
            c.name for c in profile.by_role("dimension") if c.name.endswith("_name")
        ][:2]
        report = detect_anomalies(
            entry.df,
            entry.name,
            measures=measures,
            context_columns=context,
            date_column=date_col,
            primary_measure=value_col,
            group_by=args.get("group_by"),
            sensitivity=float(args.get("sensitivity") or 3.5),
        )
        data = report.to_dict()
        llm_payload = {
            "summary": report.summary,
            "counts": report.counts,
            "method": report.method,
            "anomalies": [
                {k: v for k, v in a.items() if k in ("kind", "column", "value", "expected", "score", "reason", "date", "record")}
                for a in data["anomalies"][:25]
            ],
        }
        artifact = {
            "kind": "anomalies",
            "id": self.session.next_id("a"),
            "title": f"Anomalies in {entry.name}",
            "summary": report.summary,
            "method": report.method,
            "counts": report.counts,
            "items": data["anomalies"],
            "figure": report.chart,
        }
        return ToolOutcome(
            content=json.dumps(llm_payload, default=str),
            summary=report.summary,
            artifacts=[artifact],
        )

    # ------------------------------------------------------------------ forecast
    def forecast(self, args: dict[str, Any]) -> ToolOutcome:
        entry = self.session.get_table(str(args.get("table", "")))
        df = entry.df
        where = (args.get("where") or "").strip()
        code = None
        if where:
            res = self.session.db.execute(f"SELECT * FROM {_q(entry.name)} WHERE {where}", max_rows=2_000_000)
            df = res.df
            code = {"language": "sql", "code": res.sql}
            if df.empty:
                raise ToolError("The filter matched no rows.")
        result = forecast_series(
            df,
            str(args.get("date_column")),
            args.get("value_column"),
            periods=int(args.get("periods") or 6),
            freq=str(args.get("frequency") or "auto"),
        )
        table_df = result.forecast.copy()
        table_df["period"] = table_df["period"].dt.date.astype(str)
        rs = self.session.store_result(table_df, "forecast", f"forecast({args})", "Forecast")
        summary = result.summary()
        label = pretty(args.get("value_column") or "rows")
        artifact = {
            "kind": "forecast",
            "id": self.session.next_id("f"),
            "title": f"{label} forecast" + (f" ({where})" if where else ""),
            "summary": summary,
            "figure": result.chart,
            "table": frame_payload(table_df, 60),
            "result_id": rs.id,
        }
        payload = {"result_id": rs.id, **summary, "forecast": frame_payload(table_df, 36)}
        return ToolOutcome(
            content=json.dumps(payload, default=str),
            summary=f"{result.method}; {len(table_df)} {summary['frequency']}(s) ahead",
            artifacts=[artifact],
            code=code,
            result_id=rs.id,
        )

    # ------------------------------------------------------------------ profile
    def profile_table(self, args: dict[str, Any]) -> ToolOutcome:
        name = str(args.get("table", ""))
        try:
            entry = self.session.get_table(name)
        except NotFoundError:
            raise
        p = entry.profile
        payload = {
            "table": p.name,
            "rows": p.row_count,
            "quality_score": p.quality_score,
            "duplicate_rows": p.duplicate_rows,
            "columns": [
                {
                    "name": c.name, "type": c.dtype, "role": c.role, "null_pct": c.null_pct,
                    "unique": c.unique_count, "stats": c.stats, "top_values": c.top_values[:5],
                }
                for c in p.columns
            ],
            "issues": [i.to_dict() for i in p.issues],
        }
        artifact = {
            "kind": "quality",
            "id": self.session.next_id("q"),
            "title": f"Data quality: {p.name}",
            "table": p.name,
            "score": p.quality_score,
            "issues": [i.to_dict() for i in p.issues],
            "rows": p.row_count,
            "columns": p.column_count,
        }
        return ToolOutcome(
            content=json.dumps(payload, default=str),
            summary=f"Quality score {p.quality_score}/100, {len(p.issues)} issue(s)",
            artifacts=[artifact],
        )
