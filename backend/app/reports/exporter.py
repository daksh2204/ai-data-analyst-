"""Export a session's analysis as a self-contained HTML report or a Markdown file."""

from __future__ import annotations

import copy
import html
import json
from datetime import datetime
from typing import Any

import markdown as md

from app.sessions.store import Session

# Same roles and palette as the web client (light mode), so exported charts match.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ROLE_COLORS = {
    "baseline": "#8a8f98",
    "trend": "#52514e",
    "anomaly": "#d03b3b",
    "forecast": "#2a78d6",
    "band": "rgba(42,120,214,0.12)",
    "band-inner": "rgba(42,120,214,0.22)",
}
SEQUENTIAL = [[0, "#cde2fb"], [0.5, "#3987e5"], [1, "#0d366b"]]
DIVERGING = [[0, "#e34948"], [0.5, "#f0efec"], [1, "#2a78d6"]]
PLOTLY_CDN = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"


def themed_figure(figure: dict[str, Any]) -> dict[str, Any]:
    fig = copy.deepcopy(figure)
    for trace in fig.get("data", []):
        meta = trace.get("meta") or {}
        role = meta.get("role", "series")
        color = SERIES[int(meta.get("index", 0)) % len(SERIES)] if role == "series" else ROLE_COLORS.get(role)
        t = trace.get("type")
        if t == "pie":
            trace.setdefault("marker", {})["colors"] = SERIES
            trace["marker"]["line"] = {"color": "#ffffff", "width": 2}
        elif t == "heatmap":
            trace["colorscale"] = DIVERGING if role == "diverging" else SEQUENTIAL
        elif color:
            if t in ("bar", "histogram", "box"):
                trace.setdefault("marker", {})["color"] = color
            else:
                trace.setdefault("line", {})["color"] = color
                trace.setdefault("marker", {})["color"] = color
                if role in ("band", "band-inner"):
                    trace["fillcolor"] = color
    layout = fig.setdefault("layout", {})
    layout.update(
        {
            "font": {"family": "Inter, system-ui, sans-serif", "size": 12, "color": "#2b3640"},
            "paper_bgcolor": "#ffffff",
            "plot_bgcolor": "#ffffff",
            "height": 340,
        }
    )
    for axis in ("xaxis", "yaxis"):
        layout.setdefault(axis, {}).update({"gridcolor": "#e8ebef", "zerolinecolor": "#d5dbe1", "linecolor": "#d5dbe1"})
    return fig


def _table_html(columns: list[str], rows: list[list[Any]], limit: int = 50) -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape('' if v is None else _fmt(v))}</td>" for v in row) + "</tr>" for row in rows[:limit]
    )
    more = f"<p class='muted'>Showing {limit} of {len(rows)} rows.</p>" if len(rows) > limit else ""
    return f"<div class='table'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>{more}"


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:,.2f}"
    if isinstance(v, int) and not isinstance(v, bool):
        return f"{v:,}"
    return str(v)


def _turn_pairs(session: Session) -> list[dict[str, Any]]:
    return [m for m in session.messages if m.get("role") == "assistant"]


def render_html(session: Session) -> str:
    charts_js: list[str] = []
    sections: list[str] = []
    for n, msg in enumerate(_turn_pairs(session), start=1):
        parts = [f"<h2>{html.escape(msg['question'])}</h2>", f"<div class='answer'>{md.markdown(msg['answer'], extensions=['fenced_code', 'tables'])}</div>"]
        for art in msg.get("artifacts", []):
            kind = art.get("kind")
            fig = art.get("figure")
            if kind in ("chart", "anomalies", "forecast") and fig:
                div_id = f"chart_{n}_{art['id']}"
                parts.append(f"<figure><figcaption>{html.escape(art.get('title', ''))}</figcaption><div id='{div_id}' class='chart'></div></figure>")
                charts_js.append(
                    f"Plotly.newPlot('{div_id}', {json.dumps(themed_figure(fig)['data'], default=str)}, "
                    f"{json.dumps(themed_figure(fig)['layout'], default=str)}, {{displayModeBar: false, responsive: true}});"
                )
            if kind == "table":
                parts.append(f"<h3>{html.escape(art.get('title', 'Result'))}</h3>" + _table_html(art["columns"], art["rows"]))
            if kind == "anomalies":
                items = "".join(f"<li>{html.escape(a['reason'])}</li>" for a in art.get("items", [])[:20])
                parts.append(f"<h3>Flagged anomalies</h3><ul>{items}</ul><p class='muted'>{html.escape(art.get('method', ''))}</p>")
            if kind == "forecast" and art.get("table"):
                parts.append("<h3>Forecast values</h3>" + _table_html(art["table"]["columns"], art["table"]["rows"]))
            if kind == "quality":
                items = "".join(f"<li>{html.escape(i['message'])}</li>" for i in art.get("issues", []))
                parts.append(f"<h3>{html.escape(art['title'])}: {art['score']}/100</h3><ul>{items}</ul>")
        for code in msg.get("code", []):
            parts.append(f"<details><summary>{'SQL' if code['language'] == 'sql' else 'pandas'} used</summary><pre><code>{html.escape(code['code'])}</code></pre></details>")
        sections.append("<section>" + "\n".join(parts) + "</section>")

    datasets = "".join(
        f"<tr><td>{html.escape(t.name)}</td><td>{html.escape(t.loaded.filename)}</td><td>{t.profile.row_count:,}</td>"
        f"<td>{t.profile.column_count}</td><td>{t.profile.quality_score}/100</td></tr>"
        for t in session.tables.values()
    )
    generated = datetime.now().strftime("%d %B %Y, %H:%M")
    body = "\n".join(sections) or "<p class='muted'>No questions have been asked in this session yet.</p>"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Analysis report</title>
<script src="{PLOTLY_CDN}"></script>
<style>
  :root {{ --ink:#16232e; --muted:#5b6670; --rule:#dde3e8; --accent:#0e6b5c; --paper:#ffffff; --tint:#f4f6f8; }}
  body {{ margin:0; background:var(--tint); color:var(--ink); font:15px/1.6 Inter, system-ui, -apple-system, "Segoe UI", sans-serif; }}
  main {{ max-width:900px; margin:0 auto; padding:40px 24px 80px; }}
  header h1 {{ font-size:28px; margin:0 0 4px; letter-spacing:-0.01em; }}
  .muted {{ color:var(--muted); font-size:13px; }}
  section {{ background:var(--paper); border:1px solid var(--rule); border-radius:10px; padding:24px 28px; margin:24px 0; }}
  h2 {{ font-size:19px; margin:0 0 12px; }}
  h3 {{ font-size:14px; margin:20px 0 8px; }}
  .answer strong {{ color:var(--ink); }}
  figure {{ margin:16px 0; }} figcaption {{ font-weight:600; font-size:14px; margin-bottom:4px; }}
  .chart {{ width:100%; min-height:340px; }}
  .table {{ overflow-x:auto; }} table {{ border-collapse:collapse; width:100%; font-size:13px; font-variant-numeric:tabular-nums; }}
  th, td {{ text-align:left; padding:6px 10px; border-bottom:1px solid var(--rule); white-space:nowrap; }}
  th {{ color:var(--muted); font-weight:600; }}
  pre {{ background:#0f1b24; color:#e4ebf0; padding:14px; border-radius:8px; overflow-x:auto; font-size:12.5px; }}
  details summary {{ cursor:pointer; color:var(--accent); font-size:13px; margin-top:12px; }}
</style></head>
<body><main>
<header><h1>Analysis report</h1><p class="muted">Generated {generated}</p></header>
<section><h2>Datasets</h2><div class="table"><table><thead><tr><th>Table</th><th>File</th><th>Rows</th><th>Columns</th><th>Quality</th></tr></thead>
<tbody>{datasets}</tbody></table></div></section>
{body}
</main>
<script>{"".join(charts_js)}</script>
</body></html>"""


def render_markdown(session: Session) -> str:
    lines = ["# Analysis report", "", f"_Generated {datetime.now().strftime('%d %B %Y, %H:%M')}_", "", "## Datasets", ""]
    lines += ["| Table | File | Rows | Columns | Quality |", "|---|---|---:|---:|---:|"]
    for t in session.tables.values():
        lines.append(f"| {t.name} | {t.loaded.filename} | {t.profile.row_count:,} | {t.profile.column_count} | {t.profile.quality_score}/100 |")
    for msg in _turn_pairs(session):
        lines += ["", f"## {msg['question']}", "", msg["answer"], ""]
        for art in msg.get("artifacts", []):
            if art.get("kind") == "table":
                cols = art["columns"]
                lines += [f"**{art.get('title', 'Result')}**", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
                lines += ["| " + " | ".join(_fmt(v) if v is not None else "" for v in row) + " |" for row in art["rows"][:20]]
                lines.append("")
            elif art.get("kind") in ("chart", "forecast", "anomalies"):
                lines.append(f"_Chart: {art.get('title', '')} (see the HTML report for the interactive version)_")
        for code in msg.get("code", []):
            lines += ["", f"```{code['language']}", code["code"], "```"]
    return "\n".join(lines) + "\n"
