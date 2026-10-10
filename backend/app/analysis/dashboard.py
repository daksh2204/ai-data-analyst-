"""Deterministic dashboard and question suggestions built from the table profile.

No LLM is involved: the dashboard is instant, reproducible and free. Charts are chosen
by column role — the primary measure over time, its split by the most useful dimensions,
a share-of-total for a low-cardinality dimension and the measure's distribution.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.analysis.charts import ChartSpec, build_chart, pretty
from app.data.engine import SessionDatabase
from app.data.profiling import ColumnProfile, TableProfile


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _dimension_rank(c: ColumnProfile) -> float:
    """Prefer readable business dimensions with a handful of values."""
    score = 0.0
    if 2 <= c.unique_count <= 12:
        score += 3
    elif c.unique_count <= 40:
        score += 2
    if any(k in c.name for k in ("region", "category", "segment", "channel", "product", "country", "type", "department")):
        score += 2
    if c.name.endswith("_name"):
        score += 1
    score -= c.null_pct / 25
    return score


def pick_dimensions(profile: TableProfile, n: int = 3) -> list[ColumnProfile]:
    dims = [c for c in profile.by_role("dimension") if c.unique_count >= 2]
    return sorted(dims, key=_dimension_rank, reverse=True)[:n]


def _time_grain(span_days: int) -> str:
    if span_days > 730 * 2:
        return "quarter"
    if span_days > 120:
        return "month"
    if span_days > 21:
        return "week"
    return "day"


def build_dashboard(db: SessionDatabase, profile: TableProfile) -> dict[str, Any]:
    table = profile.name
    measure = profile.primary_measure
    time_col = profile.primary_time
    dims = pick_dimensions(profile)
    kpis: list[dict[str, Any]] = [{"label": "Rows", "value": profile.row_count, "format": "integer"}]
    charts: list[dict[str, Any]] = []

    if measure:
        m = _q(measure.name)
        stats = db.execute(f"SELECT SUM({m}) AS total, AVG({m}) AS avg FROM {_q(table)}").df.iloc[0]
        kpis.append({"label": f"Total {pretty(measure.name).lower()}", "value": float(stats["total"] or 0), "format": "compact"})
        kpis.append({"label": f"Average {pretty(measure.name).lower()} per row", "value": float(stats["avg"] or 0), "format": "compact"})
    ids = [c for c in profile.by_role("identifier") if c.name.endswith("_id")]
    entity = next((c for c in ids if "customer" in c.name), ids[0] if ids else None)
    if entity:
        kpis.append({"label": f"Distinct {pretty(entity.name[:-3]).lower()}s", "value": entity.unique_count, "format": "integer"})

    if time_col and measure:
        grain = _time_grain(int(time_col.stats.get("span_days", 0)))
        res = db.execute(
            f"SELECT date_trunc('{grain}', {_q(time_col.name)}) AS period, SUM({_q(measure.name)}) AS {_q(measure.name)} "
            f"FROM {_q(table)} WHERE {_q(time_col.name)} IS NOT NULL GROUP BY 1 ORDER BY 1"
        )
        if len(res.df) >= 2:
            # Period-over-period change of the last complete period, as a KPI.
            last, prev = res.df.iloc[-1][measure.name], res.df.iloc[-2][measure.name]
            if prev:
                kpis.append({
                    "label": f"Last {grain} vs previous",
                    "value": float((last - prev) / abs(prev)),
                    "format": "percent",
                    "detail": f"{pd.Timestamp(res.df.iloc[-1]['period']).strftime('%b %Y')}",
                })
            fig = build_chart(res.df, ChartSpec("line", x="period", y=[measure.name]))
            charts.append({
                "id": "trend",
                "title": f"{pretty(measure.name)} by {grain}",
                "chart_type": "line",
                "figure": fig["figure"],
                "sql": res.sql,
                "span": 2,
            })

    if measure:
        for d in dims[:2]:
            res = db.execute(
                f"SELECT {_q(d.name)}, SUM({_q(measure.name)}) AS {_q(measure.name)} FROM {_q(table)} "
                f"WHERE {_q(d.name)} IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT 12"
            )
            horizontal = res.df[d.name].astype(str).str.len().max() > 10 or len(res.df) > 6
            fig = build_chart(res.df, ChartSpec("horizontal_bar" if horizontal else "bar", x=d.name, y=[measure.name]))
            charts.append({
                "id": f"by_{d.name}",
                "title": f"{pretty(measure.name)} by {pretty(d.name).lower()}" + (" (top 12)" if d.unique_count > 12 else ""),
                "chart_type": "bar",
                "figure": fig["figure"],
                "sql": res.sql,
                "span": 1,
            })
        share_dim = next((d for d in dims if 2 <= d.unique_count <= 6), None)
        if share_dim:
            res = db.execute(
                f"SELECT {_q(share_dim.name)}, SUM({_q(measure.name)}) AS {_q(measure.name)} FROM {_q(table)} "
                f"WHERE {_q(share_dim.name)} IS NOT NULL GROUP BY 1 ORDER BY 2 DESC"
            )
            if (res.df[measure.name] >= 0).all():
                fig = build_chart(res.df, ChartSpec("donut", x=share_dim.name, y=[measure.name]))
                charts.append({
                    "id": f"share_{share_dim.name}",
                    "title": f"Share of {pretty(measure.name).lower()} by {pretty(share_dim.name).lower()}",
                    "chart_type": "donut",
                    "figure": fig["figure"],
                    "sql": res.sql,
                    "span": 1,
                })
        res = db.execute(f"SELECT {_q(measure.name)} FROM {_q(table)} WHERE {_q(measure.name)} IS NOT NULL", max_rows=20000)
        fig = build_chart(res.df, ChartSpec("histogram", x=measure.name))
        charts.append({
            "id": "distribution",
            "title": f"Distribution of {pretty(measure.name).lower()}",
            "chart_type": "histogram",
            "figure": fig["figure"],
            "sql": res.sql,
            "span": 1,
        })

    return {
        "table": table,
        "kpis": kpis,
        "charts": charts,
        "quality_score": profile.quality_score,
        "issues": [i.to_dict() for i in profile.issues[:6]],
    }


def suggest_questions(profiles: list[TableProfile]) -> list[str]:
    """Questions tailored to the uploaded data, shown as starting points."""
    if not profiles:
        return []
    main = max(profiles, key=lambda p: (p.primary_measure is not None, p.row_count))
    m = main.primary_measure
    t = main.primary_time
    dims = pick_dimensions(main, 3)
    out: list[str] = []
    mname = pretty(m.name).lower() if m else None
    if m and dims:
        out.append(f"Which {pretty(dims[0].name).lower()} generated the highest {mname}?")
    if m and t:
        out.append(f"Show monthly {mname} trends.")
    entities = [d.name.removesuffix("_name") for d in main.by_role("dimension") if d.name.endswith("_name")]
    # Entities whose readable names live in a lookup file (orders.customer_id -> customers.customer_name).
    lookup_names = {c.name for p in profiles if p is not main for c in p.columns if c.name.endswith("_name")}
    entities += [c.name.removesuffix("_id") for c in main.by_role("identifier")
                 if c.name.endswith("_id") and f"{c.name.removesuffix('_id')}_name" in lookup_names]
    if m and entities:
        prod = next((e for e in entities if "product" in e), entities[0])
        out.append(f"Which {pretty(prod).lower()}s are underperforming?")
        cust = next((e for e in entities if "customer" in e or "client" in e), None)
        if cust:
            out.append(f"What are the top five {pretty(cust).lower()}s?")
    out.append(f"Detect anomalies in {main.name}.")
    if m and t:
        out.append(f"Forecast {mname} for the next 6 months.")
    if len(profiles) > 1:
        out.append("How do these files relate, and what can I analyse across them?")
    out.append("Summarise the key business insights.")
    return out[:8]
