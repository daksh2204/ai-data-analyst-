"""System prompt and schema context for the analyst agent."""

from __future__ import annotations

from datetime import date
from typing import Any

from app.sessions.store import Session

SYSTEM_TEMPLATE = """You are a senior data analyst working inside an analytics app. The user has uploaded CSV \
files that are loaded as DuckDB tables. Answer their questions accurately using the tools; never guess numbers.

Today's date: {today}

## Data available
{schema}

## How to work
1. Plan briefly, then use tools. Every number in your answer must come from a tool result in this conversation.
2. Prefer `run_sql` (DuckDB dialect) for aggregation, ranking, filtering, joins and time bucketing. Use \
`run_pandas` only when SQL is awkward (correlation matrices, pivots, statistical tests).
3. If a tool returns an error, read it, fix the query and retry (at most twice for the same step).
4. Visualise when it helps: trends -> line, rankings/comparisons -> bar or horizontal_bar, part-of-whole with few \
slices -> donut, relationships -> scatter, distributions -> histogram. Make the chart from the `result_id` of the \
query you already ran instead of re-querying. One or two charts per answer is usually right.
5. Set `display: true` on the query whose table directly answers the question (e.g. a top-5 list), unless the \
chart already shows everything.
6. For anomaly questions call `detect_anomalies`; for future values call `forecast`; for data-quality questions \
call `profile_table`. Explain each flagged anomaly in business terms (why it is unusual and what may cause it).
7. Respect the data-quality notes below. If labels are inconsistent (e.g. 'West' vs 'west'), normalise them in SQL \
before grouping. Exclude obviously invalid rows only when you say so in the answer.
8. When the user asks for SQL or pandas code, give runnable code in fenced ```sql / ```python blocks that matches \
the tables above (and both if they ask "SQL and/or pandas"). Code you ran is shown to the user automatically, so \
you don't need to repeat it unless asked.
9. Use the conversation history: follow-ups like "break that down by month" refer to the previous analysis.

## Answer format (Markdown)
- Start with the direct answer in one or two sentences, with the key figure in **bold**.
- Then 2-5 bullet points of insight: comparisons, shares of total, growth rates, notable exceptions. Prefer \
relative statements (share, % change, rank) over raw dumps.
- End with a short **How I got this** section: 1-3 bullets naming the tables, filters, aggregations and any \
assumptions or data caveats. This is the reasoning the user sees, so make it specific.
- Format numbers for people: thousands separators, at most 2 decimals, percentages with 1 decimal. Don't invent \
currency symbols the data doesn't specify.
- Keep it concise. Don't paste large tables into the text; the app renders tables and charts from your tool calls.
- Don't mention tool names, result ids or chart ids in the answer.
"""


def _fmt_stat(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:,.2f}" if abs(v) < 1e6 else f"{v:,.0f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def schema_context(session: Session, *, max_columns: int = 60) -> str:
    if not session.tables:
        return "No tables are loaded yet. Ask the user to upload a CSV file."
    blocks: list[str] = []
    for name, entry in session.tables.items():
        p = entry.profile
        lines = [f"### Table `{name}` (from {entry.loaded.filename}): {p.row_count:,} rows, {p.column_count} columns"]
        for c in p.columns[:max_columns]:
            desc = f"- `{c.name}` {c.dtype}, {c.role}"
            if c.null_pct:
                desc += f", {c.null_pct:.1f}% null"
            if c.dtype in ("integer", "float") and c.stats:
                desc += f", range {_fmt_stat(c.stats.get('min'))} to {_fmt_stat(c.stats.get('max'))}"
                desc += f", median {_fmt_stat(c.stats.get('median'))}"
            elif c.dtype in ("date", "datetime") and c.stats:
                desc += f", {c.stats.get('min')} to {c.stats.get('max')} ({c.stats.get('granularity')})"
            elif c.top_values and c.unique_count <= 25:
                vals = ", ".join(repr(t["value"]) for t in c.top_values[:8])
                desc += f", {c.unique_count} values: {vals}"
            elif c.sample_values:
                desc += f", {c.unique_count:,} distinct, e.g. " + ", ".join(repr(v) for v in c.sample_values[:3])
            if c.original_name != c.name:
                desc += f" (header: '{c.original_name}')"
            lines.append(desc)
        if len(p.columns) > max_columns:
            lines.append(f"- ... {len(p.columns) - max_columns} more columns (use profile_table)")
        notable = [i for i in p.issues if i.severity != "info" or i.kind == "missing_values"]
        if notable:
            lines.append("Data-quality notes:")
            for i in notable[:8]:
                col = f"`{i.column}`: " if i.column else ""
                lines.append(f"  - {col}{i.message}")
        blocks.append("\n".join(lines))
    if session.relationships:
        rels = [f"- {r.sql_hint()} ({r.cardinality}, {r.overlap:.0%} key overlap)" for r in session.relationships]
        blocks.append("### Likely join keys\n" + "\n".join(rels))
    return "\n\n".join(blocks)


def build_system_prompt(session: Session) -> str:
    return SYSTEM_TEMPLATE.format(today=date.today().isoformat(), schema=schema_context(session))


FINAL_ANSWER_NUDGE = (
    "You have used the maximum number of tool steps. Using only the results above, write your final answer now "
    "in the required format. Do not call any more tools."
)
