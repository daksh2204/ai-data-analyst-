"""Offline analysis engine: a deterministic planner that speaks the LLM tool protocol.

It lets the app run with no API key (and serves as a fallback when the LLM is down):
the agent loop, tools, streaming and UI are exactly the same — only the "brain" differs.
The planner maps a question to an intent (ranking, trend, underperformers, anomalies,
forecast, data quality, correlation, breakdown, summary, code request), resolves the
columns it refers to from the table profile, issues real tool calls and writes the
answer from the tool results. It handles the common business questions well; open-ended
reasoning needs a real LLM.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.agent.llm.base import AgentContext, LLMEvent, Message, ToolCall, ToolSpec
from app.analysis.charts import pretty
from app.analysis.dashboard import pick_dimensions
from app.data.profiling import ColumnProfile, TableProfile

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "twelve": 12, "fifteen": 15, "twenty": 20,
}
SYNONYMS: dict[str, tuple[str, ...]] = {
    "revenue": ("revenue", "sales", "turnover", "income", "amount", "total", "value", "earning", "spend"),
    "profit": ("profit", "margin", "earnings"),
    "quantity": ("quantity", "qty", "unit", "volume"),
    "customer": ("customer", "client", "buyer", "account", "company"),
    "product": ("product", "item", "sku", "article"),
    "region": ("region", "area", "territory", "geography", "location", "market"),
    "category": ("category", "categories", "type", "segment", "line"),
    "sales_rep": ("rep", "salesperson", "seller", "agent", "representative", "employee"),
    "channel": ("channel",),
}
INTENT_WORDS: dict[str, tuple[str, ...]] = {
    "anomalies": ("anomal", "outlier", "unusual", "suspicious", "odd ", "weird", "strange", "irregular", "fraud"),
    "forecast": ("forecast", "predict", "projection", "project ", "next month", "next quarter", "next year",
                 "future", "upcoming", "next 3", "next 6", "next 12", "next six", "next three", "next twelve"),
    "quality": ("quality", "missing", "null", "duplicate", "clean", "dirty", "data issue", "errors in", "problems with the data"),
    "correlation": ("correlat", "relationship between", "related to", "relationship of", "drive", "affect", "impact of"),
    "trend": ("trend", "over time", "monthly", "weekly", "daily", "quarterly", "yearly", "by month", "per month",
              "by week", "by quarter", "by year", "growth", "seasonal", "timeline", "evolution", "month over month",
              "each month", "every month"),
    "under": ("underperform", "worst", "lowest", "least", "bottom", "declin", "poor", "weak", "lagging",
              "struggling", "losing", "decreas", "drop"),
    "top": ("top", "highest", "best", "most", "largest", "biggest", "leading", "greatest", "maximum", "max "),
    "distribution": ("distribution", "histogram", "spread", "how are", "range of"),
    "share": ("share", "proportion", "percentage", "split", "mix", "breakdown of", "composition"),
    "count": ("how many", "number of", "count"),
    "summary": ("summar", "insight", "overview", "tell me about", "describe", "key findings", "highlights", "what can you"),
    "relate": ("relate", "join", "across", "combine", "link", "connect", "multi-file", "multiple files", "together"),
    "code": ("sql", "pandas", "code", "query for", "the query"),
    "target": ("target", "budget", "goal", "quota", "attainment", "on track", "missed", "beat ", "plan vs",
               "vs plan", "against plan"),
}
GRAINS = {"day": ("daily", " day", "per day"), "week": ("weekly", " week"), "month": ("monthly", "month"),
          "quarter": ("quarterly", "quarter"), "year": ("yearly", "annual", " year")}


def q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def fmt(v: Any, pct: bool = False) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "n/a"
    if pct:
        return f"{v * 100:+.1f}%"
    if isinstance(v, (int, float, np.number)) and not isinstance(v, bool):
        v = float(v)
        av = abs(v)
        if av >= 1e9:
            return f"{v / 1e9:,.2f}B"
        if av >= 1e6:
            return f"{v / 1e6:,.2f}M"
        if av >= 1e4:
            return f"{v / 1e3:,.1f}K"
        if av >= 1e3 or float(v).is_integer():
            return f"{v:,.0f}"
        return f"{v:,.2f}"
    return str(v)


def noun(col: str | None) -> str:
    """Human label for a column used as an entity: customer_name -> 'customer'."""
    if not col:
        return ""
    return pretty(col.removesuffix("_name").removesuffix("_id")).lower()


def plural(word: str) -> str:
    if word.endswith("y") and not word.endswith(("ay", "ey", "oy")):
        return word[:-1] + "ies"
    return word if word.endswith("s") else word + "s"


def _singular(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


@dataclass
class Spec:
    intent: str
    table: str
    measure: str | None = None
    agg: str = "sum"
    dimension: str | None = None
    breakdown: str | None = None
    time_col: str | None = None
    grain: str = "month"
    n: int | None = None
    order: str = "desc"
    year: int | None = None
    periods: int = 6
    question: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlannedCall:
    name: str
    args: dict[str, Any]
    uses_previous_result: bool = False  # inject result_id from the previous tool result


class Planner:
    def __init__(self, session: Any) -> None:
        self.session = session

    # ------------------------------------------------------------- resolution helpers
    def _table(self, text: str) -> tuple[Any, TableProfile]:
        """Use a table only when it is named explicitly; "products" alone names a column, not a file."""
        for name, entry in self.session.tables.items():
            explicit = (
                f"{name} table" in text or f"table {name}" in text or f"{name}.csv" in text
                or ("_" in name and name in text) or entry.loaded.filename.lower() in text
            )
            if explicit:
                return entry, entry.profile
        entry = self.session.primary_table
        return entry, entry.profile

    def find_join(self, entry: Any, text: str, words: set[str], exclude: set[str]) -> dict[str, Any] | None:
        """A dimension mentioned in the question that lives in a related table (many-to-one join)."""
        for r in self.session.relationships:
            if r.cardinality != "many-to-one" or r.left_table != entry.name:
                continue
            other = self.session.tables[r.right_table]
            dims = self._dimension(other.profile, text, words, exclude=set(entry.df.columns) | exclude)
            if dims:
                return {"table": r.right_table, "fk": r.left_column, "pk": r.right_column, "column": dims[0].name}
        return None

    @staticmethod
    def _words(text: str) -> set[str]:
        return {_singular(w) for w in re.findall(r"[a-z0-9_]+", text.lower())}

    def _mentions(self, col: ColumnProfile, text: str, words: set[str], *, skip_measure_words: bool = False) -> int:
        tokens = [t for t in col.name.lower().split("_") if t not in ("id", "name", "code", "no")]
        if skip_measure_words:
            # "sales" in "monthly sales" names the measure, not the sales_rep column.
            measure_words = {w for k in ("revenue", "profit", "quantity") for w in (k, *SYNONYMS[k])}
            tokens = [t for t in tokens if t not in measure_words and _singular(t) not in measure_words]
            if not tokens:
                return 0
        score = 0
        if col.name.lower().replace("_", " ") in text:
            score += 3
        for t in tokens:
            if _singular(t) in words:
                score += 2
            for concept, syns in SYNONYMS.items():
                if t == concept or t in syns or _singular(t) in syns:
                    if any(any(w.startswith(s) for w in words) for s in (concept, *syns)):
                        score += 1
        return score

    def _measure(self, p: TableProfile, text: str, words: set[str]) -> ColumnProfile | None:
        measures = p.by_role("measure")
        scored = sorted(((self._mentions(c, text, words), c) for c in measures), key=lambda x: -x[0])
        if scored and scored[0][0] >= 2:
            return scored[0][1]
        return p.primary_measure

    def _dimension(self, p: TableProfile, text: str, words: set[str], exclude: set[str]) -> list[ColumnProfile]:
        cands = [c for c in p.columns if c.role in ("dimension", "identifier") and c.name not in exclude]
        scored = [(self._mentions(c, text, words, skip_measure_words=True), c) for c in cands]
        scored = [(s + (1 if c.role == "dimension" else 0) + (1 if c.name.endswith("_name") else 0), c)
                  for s, c in scored if s >= 2]
        scored.sort(key=lambda x: -x[0])
        out: list[ColumnProfile] = []
        seen_concepts: set[str] = set()
        for _, c in scored:
            concept = c.name.lower().removesuffix("_id").removesuffix("_name")
            if concept in seen_concepts:
                continue
            seen_concepts.add(concept)
            out.append(c)
        return out

    @staticmethod
    def _n(text: str) -> int | None:
        m = re.search(r"\b(?:top|bottom|best|worst|first|last|lowest|highest)\s+(\d+|" + "|".join(NUMBER_WORDS) + r")\b", text)
        if m:
            v = m.group(1)
            return int(v) if v.isdigit() else NUMBER_WORDS[v]
        m = re.search(r"\b(\d+|" + "|".join(NUMBER_WORDS) + r")\s+(?:top|best|worst|biggest|largest|lowest)\b", text)
        if m:
            v = m.group(1)
            return int(v) if v.isdigit() else NUMBER_WORDS[v]
        return None

    @staticmethod
    def _grain(text: str, default: str = "month") -> str:
        for g, keys in GRAINS.items():
            if any(k in f" {text}" for k in keys):
                return g
        return default

    @staticmethod
    def _has(text: str, intent: str) -> bool:
        return any(k in f" {text} " for k in INTENT_WORDS[intent])

    def label_map(self, entry: Any, col: str) -> dict[str, str]:
        """Variant -> canonical spelling for columns flagged with inconsistent labels."""
        if not any(i.kind == "inconsistent_labels" and i.column == col for i in entry.profile.issues):
            return {}
        s = entry.df[col].dropna().astype(str)
        canon = s.groupby(s.str.strip().str.lower()).agg(lambda v: v.value_counts().index[0])
        return {v: canon[v.strip().lower()] for v in s.unique() if v != canon[v.strip().lower()]}

    def dim_sql(self, entry: Any, col: str) -> str:
        mapping = self.label_map(entry, col)
        if not mapping:
            return q(col)
        canon = sorted(set(mapping.values()))
        whens = " ".join(f"WHEN '{c.lower()}' THEN '{c}'" for c in canon)
        return f"CASE lower(trim({q(col)})) {whens} ELSE {q(col)} END"

    # ------------------------------------------------------------- planning
    def plan(self, question: str, prior: dict[str, Any] | None) -> tuple[Spec, list[PlannedCall]]:
        text = question.lower().strip()
        words = self._words(text)
        entry, p = self._table(text)
        has = lambda intent: self._has(text, intent)  # noqa: E731

        # Follow-ups that only change the code format or one parameter reuse the prior spec.
        own_subject = any(has(i) for i in ("anomalies", "forecast", "top", "under", "trend", "target", "share",
                                            "count", "correlation", "distribution", "quality"))
        if has("code") and prior and not own_subject:
            spec = Spec(**{**prior, "extra": {**prior.get("extra", {}), "code_request": True}})
            if spec.intent in ("summary", "quality", "relate", "correlation", "anomalies", "forecast"):
                spec.intent = "top" if spec.dimension else "trend" if spec.time_col else spec.intent
            spec.question = question
            return spec, self._calls_for(spec, entry)

        measure = self._measure(p, text, words)
        count_mode = has("count") and not (measure and self._mentions(measure, text, words) >= 2)
        dims = self._dimension(p, text, words, exclude={measure.name} if measure else set())
        join = None
        if len(self.session.tables) > 1 and (not dims or dims[0].role == "identifier"):
            join = self.find_join(entry, text, words, exclude={measure.name} if measure else set())
            # "revenue by product category": the lookup table's `category` beats the fact table's product_id.
            if join and dims and _singular(join["column"].split("_")[-1]) not in words:
                join = None
        time_col = p.primary_time.name if p.primary_time else None
        years = [int(y) for y in re.findall(r"\b(20\d{2}|19\d{2})\b", text)]
        spec = Spec(
            intent="summary",
            table=entry.name,
            measure=None if count_mode else (measure.name if measure else None),
            agg="count" if count_mode else ("avg" if re.search(r"\b(average|avg|mean|typical)\b", text) else "sum"),
            dimension=dims[0].name if dims else None,
            breakdown=dims[1].name if len(dims) > 1 else None,
            time_col=time_col,
            grain=self._grain(text),
            n=self._n(text),
            year=years[0] if len(years) == 1 else None,
            question=question,
        )
        if join:
            spec.dimension = join["column"]
            spec.breakdown = None
            spec.extra["join"] = join
        m = re.search(r"next\s+(\d+|" + "|".join(NUMBER_WORDS) + r")\s+(day|week|month|quarter)", text)
        if m:
            v = m.group(1)
            spec.periods = int(v) if v.isdigit() else NUMBER_WORDS[v]
            spec.grain = m.group(2)

        # A follow-up such as "now by month" or "what about profit?" inherits the prior spec.
        followup_marker = re.match(
            r"^(now|what about|how about|and|same|instead|break|split|also|only|just|then|ok|okay|what if)\b", text
        ) or re.search(r"\b(that|this|these|those|it)\b", text)
        standalone = any(has(i) for i in ("anomalies", "forecast", "quality", "summary", "under", "top", "correlation",
                                          "distribution", "relate"))
        if prior and followup_marker and not standalone and len(text.split()) <= 10:
            merged = {**prior}
            if merged.get("intent") not in ("top", "breakdown", "share", "trend", "under"):
                merged["intent"] = "breakdown"
                merged["measure"] = merged.get("measure") or (p.primary_measure.name if p.primary_measure else None)
            if dims:
                merged["dimension"] = dims[0].name
            if measure and self._mentions(measure, text, words) >= 2:
                merged["measure"] = measure.name
            if has("trend") or any(g in text for g in ("month", "week", "quarter", "year", "day")):
                merged["intent"] = "trend"
                merged["grain"] = spec.grain
                if dims:
                    merged["breakdown"] = dims[0].name
                    merged["dimension"] = prior.get("dimension")
                elif prior.get("dimension") and prior.get("intent") in ("top", "breakdown", "share"):
                    col = p.column(prior["dimension"])
                    if col and col.unique_count <= 8:
                        merged["breakdown"] = prior["dimension"]
                    elif col and prior.get("n"):
                        merged["breakdown"] = prior["dimension"]
                        merged["extra"] = {"top_filter": {"column": prior["dimension"], "n": prior["n"]}}
                merged["n"] = None
            merged["question"] = question
            merged.setdefault("extra", {})
            merged["extra"] = {k: v for k, v in merged["extra"].items() if k in ("top_filter", "join")}
            spec = Spec(**merged)
            return spec, self._calls_for(spec, entry)

        target = self.target_table(entry) if has("target") else None
        if has("anomalies"):
            spec.intent = "anomalies"
        elif target and spec.measure:
            spec.intent = "target"
            spec.extra["target"] = target
        elif has("count") and spec.agg == "count" and (
            spec.dimension is None or (p.column(spec.dimension) and p.column(spec.dimension).role == "identifier")
        ):
            spec.intent = "count"
        elif has("forecast") and time_col:
            spec.intent = "forecast"
        elif has("quality"):
            spec.intent = "quality"
        elif has("correlation") and len(p.by_role("measure")) >= 2:
            spec.intent = "correlation"
        elif has("relate") and len(self.session.tables) > 1:
            spec.intent = "relate"
        elif has("under") and (spec.dimension or self._default_entity(p)):
            spec.intent = "under"
            spec.dimension = spec.dimension or self._default_entity(p)
        elif has("trend") and time_col:
            spec.intent = "trend"
            if spec.dimension and not spec.breakdown:
                spec.breakdown, spec.dimension = spec.dimension, None
        elif has("distribution") and spec.measure:
            spec.intent = "distribution"
        elif has("top") and (spec.dimension or self._default_entity(p)):
            spec.intent = "top"
            spec.dimension = spec.dimension or self._default_entity(p)
        elif has("share") and spec.dimension:
            spec.intent = "share"
        elif has("count") and spec.dimension is None:
            spec.intent = "count"
        elif spec.dimension and (spec.measure or spec.agg == "count") and not has("summary"):
            spec.intent = "breakdown"
        else:
            spec.intent = "summary"
        if has("code"):
            spec.extra["code_request"] = True
        return spec, self._calls_for(spec, entry)

    def target_table(self, entry: Any) -> dict[str, Any] | None:
        """A second table holding targets/budgets keyed by dimensions the fact table shares."""
        pattern = re.compile(r"(target|budget|goal|quota|plan)", re.I)
        for name, other in self.session.tables.items():
            if name == entry.name:
                continue
            tcol = next((c for c in other.profile.by_role("measure") if pattern.search(c.name)), None)
            if tcol is None:
                continue
            keys = [c.name for c in other.profile.columns
                    if c.role == "dimension" and c.name in entry.df.columns]
            if not keys:
                continue
            ttime = other.profile.primary_time
            return {"table": name, "column": tcol.name, "key": keys[0],
                    "time": ttime.name if ttime else None, "grain": (ttime.stats.get("granularity") if ttime else None)}
        return None

    def target_sql(self, spec: Spec, entry: Any) -> str:
        tg = spec.extra["target"]
        key, m = q(tg["key"]), q(spec.measure or "")
        grain = {"daily": "day", "weekly": "week", "quarterly": "quarter", "yearly": "year"}.get(tg.get("grain") or "", "month")
        if tg.get("time") and spec.time_col:
            ft, tt = q(spec.time_col), q(tg["time"])
            year = f"\nWHERE year(t.period) = {spec.year}" if spec.year else ""
            return (
                f"WITH actual AS (\n"
                f"  SELECT {key}, date_trunc('{grain}', {ft}) AS period, SUM({m}) AS actual\n"
                f"  FROM {q(entry.name)} GROUP BY 1, 2\n),\n"
                f"target AS (\n"
                f"  SELECT {key}, date_trunc('{grain}', {tt}) AS period, SUM({q(tg['column'])}) AS target\n"
                f"  FROM {q(tg['table'])} GROUP BY 1, 2\n)\n"
                f"SELECT t.{key},\n"
                f"       ROUND(SUM(COALESCE(a.actual, 0)), 2) AS actual,\n"
                f"       ROUND(SUM(t.target), 2) AS target,\n"
                f"       ROUND(100.0 * SUM(COALESCE(a.actual, 0)) / NULLIF(SUM(t.target), 0), 1) AS attainment_pct,\n"
                f"       COUNT(*) FILTER (WHERE COALESCE(a.actual, 0) < t.target) AS periods_below_target,\n"
                f"       COUNT(*) AS periods\n"
                f"FROM target t\nLEFT JOIN actual a ON a.{key} = t.{key} AND a.period = t.period{year}\n"
                f"GROUP BY 1\nORDER BY attainment_pct ASC"
            )
        return (
            f"WITH actual AS (SELECT {key}, SUM({m}) AS actual FROM {q(entry.name)} GROUP BY 1),\n"
            f"target AS (SELECT {key}, SUM({q(tg['column'])}) AS target FROM {q(tg['table'])} GROUP BY 1)\n"
            f"SELECT t.{key}, ROUND(COALESCE(a.actual, 0), 2) AS actual, ROUND(t.target, 2) AS target,\n"
            f"       ROUND(100.0 * COALESCE(a.actual, 0) / NULLIF(t.target, 0), 1) AS attainment_pct\n"
            f"FROM target t LEFT JOIN actual a ON a.{key} = t.{key}\nORDER BY attainment_pct ASC"
        )

    @staticmethod
    def _default_entity(p: TableProfile) -> str | None:
        names = [c for c in p.by_role("dimension") if c.name.endswith("_name")]
        prod = next((c for c in names if "product" in c.name), None)
        if prod:
            return prod.name
        if names:
            return names[0].name
        dims = pick_dimensions(p, 1)
        return dims[0].name if dims else None

    # ------------------------------------------------------------- SQL builders
    def _agg(self, spec: Spec) -> tuple[str, str]:
        if spec.agg == "count" or not spec.measure:
            return "COUNT(*)", "row_count"
        fn = {"sum": "SUM", "avg": "AVG"}.get(spec.agg, "SUM")
        alias = spec.measure if spec.agg == "sum" else f"avg_{spec.measure}"
        expr = f"{fn}({q(spec.measure)})"
        return (f"ROUND({expr}, 2)" if spec.agg == "avg" else expr), alias

    def _where(self, spec: Spec) -> str:
        if spec.year and spec.time_col:
            return f"WHERE year({q(spec.time_col)}) = {spec.year}"
        return ""

    def group_sql(self, spec: Spec, entry: Any, *, limit: int | None, order: str) -> str:
        dim = spec.dimension
        assert dim
        agg, alias = self._agg(spec)
        where = self._where(spec)
        join = spec.extra.get("join")
        if join:
            dsql = f"d.{q(dim)}"
            from_sql = f"{q(entry.name)} f\nJOIN {q(join['table'])} d ON f.{q(join['fk'])} = d.{q(join['pk'])}"
            agg = agg.replace('("', '(f."') if spec.measure else agg
            null_filter = f"d.{q(dim)} IS NOT NULL"
        else:
            dsql = self.dim_sql(entry, dim)
            from_sql = q(entry.name)
            null_filter = f"{q(dim)} IS NOT NULL"
        where = f"{where} AND {null_filter}" if where else f"WHERE {null_filter}"
        share = f",\n       ROUND(100.0 * {agg} / SUM({agg}) OVER (), 1) AS pct_of_total" if spec.agg != "avg" else ""
        sql = (
            f"SELECT {dsql} AS {q(dim)},\n       {agg} AS {q(alias)}{share}\n"
            f"FROM {from_sql}\n{where}\nGROUP BY 1\nORDER BY 2 {order.upper()}"
        )
        if limit:
            sql += f"\nLIMIT {limit}"
        return sql

    def trend_sql(self, spec: Spec, entry: Any) -> str:
        agg, alias = self._agg(spec)
        t = q(spec.time_col or "")
        sel = [f"date_trunc('{spec.grain}', {t}) AS period"]
        group = "1"
        if spec.breakdown:
            sel.append(f"{self.dim_sql(entry, spec.breakdown)} AS {q(spec.breakdown)}")
            group = "1, 2"
        where = self._where(spec) or f"WHERE {t} IS NOT NULL"
        top = spec.extra.get("top_filter")
        if top:
            m = q(spec.measure) if spec.measure else "1"
            where += (f"\n  AND {q(top['column'])} IN (SELECT {q(top['column'])} FROM {q(entry.name)} "
                      f"GROUP BY 1 ORDER BY SUM({m}) DESC LIMIT {int(top['n'])})")
        return (
            f"SELECT {', '.join(sel)},\n       {agg} AS {q(alias)}\nFROM {q(entry.name)}\n{where}\n"
            f"GROUP BY {group}\nORDER BY {group}"
        )

    def under_sql(self, spec: Spec, entry: Any) -> str:
        dim = spec.dimension or ""
        join = spec.extra.get("join")
        if join:
            m = f"f.{q(spec.measure)}" if spec.measure else "1"
            t = f"f.{q(spec.time_col or '')}"
            dsql = f"d.{q(dim)}"
            from_sql = (f"{q(entry.name)} f\n  JOIN {q(join['table'])} d ON f.{q(join['fk'])} = d.{q(join['pk'])}"
                        f"\n  CROSS JOIN bounds")
        else:
            m = q(spec.measure) if spec.measure else "1"
            t = q(spec.time_col or "")
            dsql = self.dim_sql(entry, dim)
            from_sql = f"{q(entry.name)}, bounds"
        bounds_t = q(spec.time_col or "")
        return (
            f"WITH bounds AS (SELECT max({bounds_t}) AS max_d FROM {q(entry.name)}),\n"
            f"by_period AS (\n"
            f"  SELECT {dsql} AS {q(dim)},\n"
            f"         SUM(CASE WHEN {t} > max_d - INTERVAL 12 MONTH THEN {m} ELSE 0 END) AS last_12m,\n"
            f"         SUM(CASE WHEN {t} <= max_d - INTERVAL 12 MONTH AND {t} > max_d - INTERVAL 24 MONTH "
            f"THEN {m} ELSE 0 END) AS prior_12m\n"
            f"  FROM {from_sql}\n  WHERE {dsql} IS NOT NULL\n  GROUP BY 1\n)\n"
            f"SELECT {q(dim)}, ROUND(last_12m, 2) AS last_12m, ROUND(prior_12m, 2) AS prior_12m,\n"
            f"       ROUND(100.0 * (last_12m - prior_12m) / NULLIF(prior_12m, 0), 1) AS growth_pct,\n"
            f"       ROUND(100.0 * last_12m / SUM(last_12m) OVER (), 2) AS share_pct\n"
            f"FROM by_period\nORDER BY growth_pct ASC NULLS LAST, last_12m ASC\nLIMIT {spec.n or 10}"
        )

    def pandas_for(self, spec: Spec, entry: Any) -> str:
        t = entry.name
        dim, m = spec.dimension, spec.measure
        norm = ""
        target = dim if spec.intent != "trend" else spec.breakdown
        if target and self.label_map(entry, target):
            norm = f'.assign({target}=lambda d: d["{target}"].str.strip().str.title())'
        yr = f'[{t}["{spec.time_col}"].dt.year == {spec.year}]' if spec.year else ""
        agg = "mean" if spec.agg == "avg" else "sum"
        join = spec.extra.get("join")
        if join and spec.intent != "trend":
            # Bring the readable name in from the lookup table first.
            jt = join["table"]
            on = f'left_on="{join["fk"]}", right_on="{join["pk"]}"' if join["fk"] != join["pk"] else f'on="{join["fk"]}"'
            norm = f'.merge({jt}[["{join["pk"]}", "{dim}"]], {on}, how="left")' + norm
        if spec.intent == "trend":
            freq = {"day": "D", "week": "W", "month": "M", "quarter": "Q", "year": "Y"}[spec.grain]
            keys = '["period", "' + spec.breakdown + '"]' if spec.breakdown else '"period"'
            value = f'["{m}"].{agg}()' if m else ".size().rename(\"row_count\")"
            return (
                f"result = (\n    {t}{yr}{norm}\n"
                f'    .assign(period=lambda d: d["{spec.time_col}"].dt.to_period("{freq}").dt.to_timestamp())\n'
                f"    .groupby({keys}){value}\n    .reset_index()\n)"
            )
        if spec.intent == "under":
            base = f"data = {t}{norm}\n" if norm else ""
            src = "data" if norm else t
            return (
                f'{base}max_d = {src}["{spec.time_col}"].max()\n'
                f'last = {src}[{src}["{spec.time_col}"] > max_d - pd.DateOffset(months=12)]\n'
                f'prior = {src}[({src}["{spec.time_col}"] <= max_d - pd.DateOffset(months=12)) & '
                f'({src}["{spec.time_col}"] > max_d - pd.DateOffset(months=24))]\n'
                f'result = pd.DataFrame({{\n    "last_12m": last.groupby("{dim}")["{m}"].sum(),\n'
                f'    "prior_12m": prior.groupby("{dim}")["{m}"].sum(),\n}}).fillna(0)\n'
                f'result["growth_pct"] = (100 * (result["last_12m"] / result["prior_12m"] - 1)).round(1)\n'
                f'result = result.sort_values("growth_pct").head({spec.n or 10}).reset_index()'
            )
        value = f'["{m}"].{agg}()' if m and spec.agg != "count" else ".size().rename(\"row_count\")"
        sort_col = m if m and spec.agg != "count" else "row_count"
        asc = "True" if spec.order == "asc" else "False"
        head = f"\n    .head({spec.n})" if spec.n else ""
        return (
            f"result = (\n    {t}{yr}{norm}\n    .groupby(\"{dim}\"){value}\n    .reset_index()\n"
            f"    .sort_values(\"{sort_col}\", ascending={asc}){head}\n)"
        )

    # ------------------------------------------------------------- calls per intent
    def name_join(self, entry: Any, dim: str | None) -> dict[str, Any] | None:
        """customer_id -> customers.customer_name, when a many-to-one lookup table holds readable names."""
        if not dim or not dim.endswith("_id"):
            return None
        stem = dim[: -len("_id")]
        for r in self.session.relationships:
            if r.cardinality != "many-to-one" or r.left_table != entry.name or r.left_column != dim:
                continue
            other = self.session.tables[r.right_table]
            for cand in (f"{stem}_name", "name", f"{stem}_title"):
                if cand in other.df.columns and cand not in entry.df.columns:
                    return {"table": r.right_table, "fk": r.left_column, "pk": r.right_column, "column": cand}
        return None

    def _calls_for(self, spec: Spec, entry: Any) -> list[PlannedCall]:
        p = entry.profile
        if spec.intent in ("top", "breakdown", "share", "under") and not spec.extra.get("join"):
            if (nj := self.name_join(entry, spec.dimension)) is not None:
                spec.extra["join"] = nj
                spec.dimension = nj["column"]
        purpose_dim = noun(spec.dimension)
        measure_label = AnswerWriter.measure_label(self._agg(spec)[1], entry)
        title_measure = measure_label[:1].upper() + measure_label[1:]
        bd = p.column(spec.breakdown) if spec.breakdown else None
        if spec.intent in ("top", "breakdown") and spec.dimension and bd and bd.unique_count <= 8 and not spec.extra.get("join"):
            agg, alias = self._agg(spec)
            sql = (
                f"SELECT {self.dim_sql(entry, spec.dimension)} AS {q(spec.dimension)},\n"
                f"       {self.dim_sql(entry, spec.breakdown)} AS {q(spec.breakdown)},\n       {agg} AS {q(alias)}\n"
                f"FROM {q(entry.name)}\n{self._where(spec) or ''}\nGROUP BY 1, 2\nORDER BY 1, 3 DESC"
            )
            return [
                PlannedCall("run_sql", {
                    "query": sql,
                    "purpose": f"Aggregate {measure_label} by {purpose_dim} and {noun(spec.breakdown)}.",
                }),
                PlannedCall("create_chart", {
                    "chart_type": "stacked_bar", "x": spec.dimension, "y": [alias], "color": spec.breakdown,
                    "title": f"{title_measure} by {purpose_dim} and {noun(spec.breakdown)}",
                    "purpose": f"Show how each {purpose_dim} splits by {noun(spec.breakdown)}.",
                }, uses_previous_result=True),
            ]
        if spec.intent in ("top", "breakdown", "share"):
            if spec.intent == "top":
                card = (p.column(spec.dimension) or p.columns[0]).unique_count if spec.dimension else 0
                limit = spec.n or (None if card <= 12 else 10)
            else:
                limit = spec.n
            spec.order = "desc"
            sql = self.group_sql(spec, entry, limit=limit, order="desc")
            chart_type = "donut" if spec.intent == "share" else "horizontal_bar"
            if spec.intent != "share":
                card = (p.column(spec.dimension) or p.columns[0]).unique_count
                chart_type = "bar" if (limit or card) <= 6 else "horizontal_bar"
            return [
                PlannedCall("run_sql", {
                    "query": sql, "display": True,
                    "purpose": f"Aggregate {measure_label} by {purpose_dim} and rank the results.",
                    "title": f"{title_measure} by {purpose_dim}",
                }),
                PlannedCall("create_chart", {
                    "chart_type": chart_type, "x": spec.dimension, "y": [self._agg(spec)[1]],
                    "title": f"{title_measure} by {purpose_dim}" + (f", {spec.year}" if spec.year else ""),
                    "purpose": f"Compare {plural(purpose_dim)} visually.",
                }, uses_previous_result=True),
            ]
        if spec.intent == "under":
            if not spec.time_col or not spec.measure:
                spec.intent, spec.order = "top", "asc"
                return [PlannedCall("run_sql", {
                    "query": self.group_sql(spec, entry, limit=spec.n or 5, order="asc"), "display": True,
                    "purpose": f"Rank {plural(purpose_dim)} from lowest {measure_label}.", "title": f"Lowest {measure_label}",
                })]
            return [
                PlannedCall("run_sql", {
                    "query": self.under_sql(spec, entry), "display": True,
                    "purpose": f"Compare each {purpose_dim}'s {measure_label} in the last 12 months with the 12 months before, "
                               "to find who is shrinking.",
                    "title": f"{pretty(plural(noun(spec.dimension)))} ranked by 12-month growth",
                }),
                PlannedCall("create_chart", {
                    "chart_type": "horizontal_bar", "x": spec.dimension, "y": ["growth_pct"],
                    "title": f"{title_measure} growth, last 12 months vs prior 12 (%)",
                    "x_label": "Growth (%)", "purpose": "Show the size of each decline.",
                }, uses_previous_result=True),
            ]
        if spec.intent == "trend":
            sql = self.trend_sql(spec, entry)
            return [
                PlannedCall("run_sql", {
                    "query": sql,
                    "purpose": f"Total {measure_label} per {spec.grain}"
                               + (f", split by {noun(spec.breakdown)}" if spec.breakdown else "")
                               + (f" (top {spec.extra['top_filter']['n']} only)" if spec.extra.get("top_filter") else "") + ".",
                    "title": f"{title_measure} by {spec.grain}",
                }),
                PlannedCall("create_chart", {
                    "chart_type": "line", "x": "period", "y": [self._agg(spec)[1]], "color": spec.breakdown,
                    "title": f"{title_measure} by {spec.grain}"
                             + (f" and {noun(spec.breakdown)}" if spec.breakdown else ""),
                    "purpose": "Plot the trend over time.",
                }, uses_previous_result=True),
            ]
        if spec.intent == "target":
            tg = spec.extra["target"]
            return [
                PlannedCall("run_sql", {
                    "query": self.target_sql(spec, entry), "display": True,
                    "purpose": f"Join actual {measure_label} from {entry.name} to {tg['table']}.{tg['column']} by "
                               f"{tg['key']}" + (" and period" if tg.get("time") else "") + " and compute attainment.",
                    "title": f"Actual vs target by {noun(tg['key'])}",
                }),
                PlannedCall("create_chart", {
                    "chart_type": "grouped_bar", "x": tg["key"], "y": ["actual", "target"],
                    "title": f"Actual vs target {measure_label} by {noun(tg['key'])}",
                    "purpose": "Compare actuals with targets side by side.",
                }, uses_previous_result=True),
            ]
        if spec.intent == "anomalies":
            return [PlannedCall("detect_anomalies", {
                "table": entry.name,
                "purpose": "Scan numeric columns for robust outliers and rule breaks, and the main measure over time "
                           "for unusual days and weeks.",
            })]
        if spec.intent == "forecast":
            freq = {"day": "D", "week": "W", "month": "M", "quarter": "Q"}.get(spec.grain, "M")
            return [PlannedCall("forecast", {
                "table": entry.name, "date_column": spec.time_col, "value_column": spec.measure,
                "periods": spec.periods, "frequency": freq,
                "purpose": f"Project {measure_label} {spec.periods} {spec.grain}s ahead with Holt-Winters smoothing.",
            })]
        if spec.intent == "quality":
            calls = [PlannedCall("profile_table", {"table": n, "purpose": f"Check {n} for missing values, duplicates and inconsistent labels."})
                     for n in self.session.tables]
            return calls[:4]
        if spec.intent == "correlation":
            cols = [c.name for c in p.by_role("measure")][:8]
            code = f"result = {entry.name}[{cols!r}].corr().round(2)"
            return [
                PlannedCall("run_pandas", {"code": code, "purpose": "Compute pairwise Pearson correlations between the numeric measures."}),
                PlannedCall("create_chart", {"chart_type": "heatmap", "x": "index", "title": "Correlation between measures",
                                             "purpose": "Show the correlation matrix."}, uses_previous_result=True),
            ]
        if spec.intent == "distribution":
            m = q(spec.measure or "")
            sql = (f"SELECT {m} FROM {q(entry.name)} WHERE {m} IS NOT NULL")
            stats = (f"SELECT COUNT(*) AS rows, MIN({m}) AS min, quantile_cont({m}, 0.25) AS p25, median({m}) AS median,\n"
                     f"       quantile_cont({m}, 0.75) AS p75, MAX({m}) AS max, ROUND(AVG({m}), 2) AS mean\n"
                     f"FROM {q(entry.name)}")
            return [
                PlannedCall("run_sql", {"query": stats, "display": True, "purpose": f"Summarise the distribution of {measure_label}.",
                                        "title": f"{title_measure} summary statistics"}),
                PlannedCall("create_chart", {"chart_type": "histogram", "sql": sql, "x": spec.measure,
                                             "title": f"Distribution of {measure_label}", "purpose": "Show the shape of the distribution."}),
            ]
        if spec.intent == "count":
            ids = [c for c in p.by_role("identifier")]
            sql = "SELECT COUNT(*) AS rows" + "".join(
                f", COUNT(DISTINCT {q(c.name)}) AS distinct_{c.name}" for c in ids[:3]) + f" FROM {q(entry.name)}"
            return [PlannedCall("run_sql", {"query": sql, "display": True, "purpose": "Count rows and distinct keys.", "title": "Counts"})]
        if spec.intent == "relate":
            return self._relate_calls(entry)
        return self._summary_calls(spec, entry)

    def _summary_calls(self, spec: Spec, entry: Any) -> list[PlannedCall]:
        p = entry.profile
        m = spec.measure or (p.primary_measure.name if p.primary_measure else None)
        dims = pick_dimensions(p, 2)
        calls: list[PlannedCall] = []
        kpi = ["COUNT(*) AS rows"]
        if m:
            kpi += [f"SUM({q(m)}) AS total_{m}", f"ROUND(AVG({q(m)}), 2) AS avg_{m}"]
        if spec.time_col:
            kpi += [f"MIN({q(spec.time_col)}) AS first_date", f"MAX({q(spec.time_col)}) AS last_date"]
        calls.append(PlannedCall("run_sql", {
            "query": f"SELECT {', '.join(kpi)} FROM {q(entry.name)}",
            "purpose": "Get headline totals and the date range.",
        }))
        if m and dims:
            s = Spec(intent="top", table=entry.name, measure=m, dimension=dims[0].name)
            calls.append(PlannedCall("run_sql", {
                "query": self.group_sql(s, entry, limit=10, order="desc"),
                "purpose": f"See which {pretty(dims[0].name).lower()}s contribute most.",
                "display": True, "title": f"{pretty(m)} by {pretty(dims[0].name).lower()}",
            }))
        if m and spec.time_col:
            s = Spec(intent="trend", table=entry.name, measure=m, time_col=spec.time_col, grain="month")
            calls.append(PlannedCall("run_sql", {"query": self.trend_sql(s, entry), "purpose": "Check the monthly trend."}))
            calls.append(PlannedCall("create_chart", {
                "chart_type": "line", "x": "period", "y": [m], "title": f"{pretty(m)} by month",
                "purpose": "Plot the monthly trend.",
            }, uses_previous_result=True))
        return calls

    def _relate_calls(self, entry: Any) -> list[PlannedCall]:
        rels = [r for r in self.session.relationships if r.cardinality == "many-to-one"]
        m = entry.profile.primary_measure
        for r in rels:
            fact, dim_table = (r.left_table, r.right_table) if r.left_table == entry.name else (r.right_table, r.left_table)
            if fact != entry.name or not m:
                continue
            other = self.session.tables[dim_table].profile
            attr = next((c for c in other.by_role("dimension")
                         if c.name not in entry.df.columns and 2 <= c.unique_count <= 12), None)
            if not attr:
                continue
            fk = r.left_column if r.left_table == entry.name else r.right_column
            pk = r.right_column if r.left_table == entry.name else r.left_column
            sql = (
                f"SELECT d.{q(attr.name)}, SUM(f.{q(m.name)}) AS {q(m.name)}, COUNT(*) AS rows,\n"
                f"       ROUND(100.0 * SUM(f.{q(m.name)}) / SUM(SUM(f.{q(m.name)})) OVER (), 1) AS pct_of_total\n"
                f"FROM {q(entry.name)} f\nJOIN {q(dim_table)} d ON f.{q(fk)} = d.{q(pk)}\n"
                f"GROUP BY 1\nORDER BY 2 DESC"
            )
            return [
                PlannedCall("run_sql", {
                    "query": sql, "display": True,
                    "purpose": f"Join {entry.name} to {dim_table} on {fk} to analyse {m.name} by {attr.name}, which only exists in {dim_table}.",
                    "title": f"{pretty(m.name)} by {pretty(attr.name).lower()} (joined from {dim_table})",
                }),
                PlannedCall("create_chart", {
                    "chart_type": "bar", "x": attr.name, "y": [m.name],
                    "title": f"{pretty(m.name)} by {pretty(attr.name).lower()}", "purpose": "Visualise the cross-file result.",
                }, uses_previous_result=True),
            ]
        return self._summary_calls(Spec(intent="summary", table=entry.name, time_col=(entry.profile.primary_time.name if entry.profile.primary_time else None)), entry)


# --------------------------------------------------------------------------- answer writing




class AnswerWriter:
    def __init__(self, planner: Planner) -> None:
        self.p = planner

    def _rows(self, result: dict[str, Any]) -> pd.DataFrame:
        """The full result while the session still holds it (the tool payload is truncated)."""
        rid = result.get("result_id")
        if rid and rid in self.p.session.results:
            return self.p.session.results[rid].df.copy()
        return pd.DataFrame(result.get("rows", []), columns=result.get("columns", []))

    @staticmethod
    def measure_label(col: str, entry: Any) -> str:
        if col == "row_count":
            ids = [c.name for c in entry.profile.by_role("identifier") if c.name.endswith("_id")]
            return plural(noun(ids[0])) if ids else "rows"
        if col.startswith("avg_"):
            return "average " + pretty(col[4:]).lower()
        return pretty(col).lower()

    def write(self, spec: Spec, results: list[dict[str, Any]], calls: list[PlannedCall]) -> str:
        entry = self.p.session.tables[spec.table]
        errors = [r["error"] for r in results if "error" in r]
        if errors and len(errors) == len(results):
            return f"I couldn't complete this analysis: {errors[0]}\n\nTry rephrasing the question or naming the columns to use."
        fn = getattr(self, f"_w_{spec.intent}", self._w_summary)
        body = fn(spec, results, entry)
        if spec.extra.get("code_request"):
            sql = next((c.args.get("query") for c in calls if c.name == "run_sql"), None)
            body += "\n\n**SQL (DuckDB)**\n\n```sql\n" + (sql or "") + "\n```"
            if spec.intent in ("top", "breakdown", "share", "trend", "under"):
                body += "\n\n**pandas**\n\n```python\n" + self.p.pandas_for(spec, entry) + "\n```"
        return body

    def _how(self, spec: Spec, entry: Any, extra: list[str] | None = None) -> str:
        lines = [f"Source: `{entry.name}` ({entry.profile.row_count:,} rows)."]
        if spec.year:
            lines.append(f"Filtered to {spec.year}.")
        for col in (spec.dimension, spec.breakdown):
            if col and self.p.label_map(entry, col):
                variants = ", ".join(f"'{k}' → '{v}'" for k, v in list(self.p.label_map(entry, col).items())[:3])
                lines.append(f"Normalised inconsistent `{col}` labels ({variants}) before grouping.")
        lines += extra or []
        return "\n\n**How I got this**\n" + "\n".join(f"- {line}" for line in lines)

    def _w_top(self, spec: Spec, results: list[dict[str, Any]], entry: Any) -> str:
        df = self._rows(results[0])
        if df.empty:
            return "The query returned no rows." + self._how(spec, entry)
        if spec.breakdown and len(df.columns) == 3 and spec.breakdown in df.columns:
            return self._w_crosstab(spec, df, entry)
        dim, val = df.columns[0], df.columns[1]
        mlabel = self.measure_label(val, entry)
        dlabel = noun(dim)
        top = df.iloc[0]
        share = f", {top['pct_of_total']:.1f}% of the total" if "pct_of_total" in df.columns else ""
        asked_n = spec.n
        if asked_n and asked_n > 1:
            head = f"The top {min(asked_n, len(df))} {plural(dlabel)} by {mlabel} are led by **{top[dim]}** with **{fmt(top[val])}**{share}."
        elif val == "row_count":
            head = f"**{top[dim]}** has the most {mlabel}: **{fmt(top[val])}**{share}."
        else:
            head = f"**{top[dim]}** has the highest {mlabel}: **{fmt(top[val])}**{share}."
        bullets = []
        for i, row in df.head(asked_n or 5).iloc[1:].iterrows():
            s = f" ({row['pct_of_total']:.1f}%)" if "pct_of_total" in df.columns else ""
            bullets.append(f"#{i + 1} {row[dim]}: {fmt(row[val])}{s}")
        if len(df) > 1 and df.iloc[1][val]:
            lead = (top[val] - df.iloc[1][val]) / abs(df.iloc[1][val])
            bullets.append(f"{top[dim]} leads #2 by {lead * 100:.1f}%.")
        if "pct_of_total" in df.columns and asked_n and asked_n > 1:
            bullets.append(f"Together the top {min(asked_n, len(df))} account for {df.head(asked_n)['pct_of_total'].sum():.1f}% of the total.")
        elif len(df) > 2 and not asked_n:
            low = df.iloc[-1]
            s = f" ({low['pct_of_total']:.1f}%)" if "pct_of_total" in df.columns else ""
            bullets.append(f"Lowest: {low[dim]} at {fmt(low[val])}{s}.")
        if spec.measure and spec.agg != "count":
            how = [f"{'Averaged' if spec.agg == 'avg' else 'Summed'} `{spec.measure}` by `{dim}` and ranked from highest to lowest."]
        else:
            how = [f"Counted rows by `{dim}` and ranked from highest to lowest."]
        join = spec.extra.get("join")
        if join:
            how.append(f"Joined `{entry.name}` to `{join['table']}` on `{join['fk']}` because `{dim}` lives in `{join['table']}`.")
        return head + "\n\n" + "\n".join(f"- {b}" for b in bullets) + self._how(spec, entry, how)

    def _w_target(self, spec: Spec, results: list[dict[str, Any]], entry: Any) -> str:
        df = self._rows(results[0])
        tg = spec.extra["target"]
        if df.empty:
            return "The actual-vs-target comparison returned no rows." + self._how(spec, entry)
        key = df.columns[0]
        missed = df[df["attainment_pct"] < 100]
        mlabel = self.measure_label(spec.measure or "", entry)
        if missed.empty:
            head = f"Every {noun(key)} met or beat its {mlabel} target."
        else:
            worst = missed.iloc[0]
            head = (f"**{len(missed)} of {len(df)} {plural(noun(key))}** missed their {mlabel} target. "
                    f"The furthest behind is **{worst[key]}** at **{worst['attainment_pct']:.1f}%** of target "
                    f"({fmt(worst['actual'])} vs {fmt(worst['target'])}).")
        bullets = []
        for _, r in df.iterrows():
            gap = r["actual"] - r["target"]
            extra = ""
            if "periods_below_target" in df.columns:
                extra = f"; below target in {int(r['periods_below_target'])} of {int(r['periods'])} periods"
            bullets.append(f"{r[key]}: {r['attainment_pct']:.1f}% of target ({'+' if gap >= 0 else '−'}{fmt(abs(gap))}{extra})")
        total_a, total_t = df["actual"].sum(), df["target"].sum()
        bullets.append(f"Overall: {fmt(total_a)} actual vs {fmt(total_t)} target ({100 * total_a / total_t:.1f}%).")
        how = [f"Targets from `{tg['table']}`.`{tg['column']}`, matched on `{tg['key']}`"
               + (" and period (both bucketed to the same grain)." if tg.get("time") else "."),
               "Attainment = actual ÷ target; a period with no sales counts as zero actual."]
        return head + "\n\n" + "\n".join(f"- {b}" for b in bullets) + self._how(spec, entry, how)

    def _w_crosstab(self, spec: Spec, df: pd.DataFrame, entry: Any) -> str:
        dim, bd, val = df.columns
        mlabel = self.measure_label(val, entry)
        totals = df.groupby(dim)[val].sum().sort_values(ascending=False)
        bd_totals = df.groupby(bd)[val].sum().sort_values(ascending=False)
        grand = totals.sum()
        head = (f"**{totals.index[0]}** leads on {mlabel} (**{fmt(totals.iloc[0])}**, {totals.iloc[0] / grand:.1%}), and "
                f"**{bd_totals.index[0]}** is the largest {noun(bd)} overall ({bd_totals.iloc[0] / grand:.1%}).")
        bullets = []
        for d_val in totals.index[:6]:
            sub = df[df[dim] == d_val].sort_values(val, ascending=False)
            lead = sub.iloc[0]
            bullets.append(f"{d_val}: {fmt(totals[d_val])}, led by {lead[bd]} ({lead[val] / totals[d_val]:.0%})")
        how = [f"Summed `{spec.measure}` by `{dim}` and `{bd}`." if spec.measure else f"Counted rows by `{dim}` and `{bd}`."]
        return head + "\n\n" + "\n".join(f"- {b}" for b in bullets) + self._how(spec, entry, how)

    _w_breakdown = _w_top

    def _w_share(self, spec: Spec, results: list[dict[str, Any]], entry: Any) -> str:
        return self._w_top(spec, results, entry)

    def _w_under(self, spec: Spec, results: list[dict[str, Any]], entry: Any) -> str:
        df = self._rows(results[0])
        if df.empty:
            return "No rows matched." + self._how(spec, entry)
        if "growth_pct" not in df.columns:
            return self._w_top(spec, results, entry)
        dim = df.columns[0]
        mlabel = pretty(spec.measure or "").lower()
        declining = df[df["growth_pct"] < 0]
        worst = df.iloc[0]
        if declining.empty:
            head = f"No {noun(dim)} is shrinking; the slowest grower is **{worst[dim]}** at **{worst['growth_pct']:+.1f}%**."
        else:
            head = (f"**{len(declining)} {plural(noun(dim)) if len(declining) > 1 else noun(dim)}** saw {mlabel} fall over the last 12 months. "
                    f"The weakest is **{worst[dim]}**, down **{abs(worst['growth_pct']):.1f}%** "
                    f"({fmt(worst['prior_12m'])} → {fmt(worst['last_12m'])}).")
        bullets = [
            f"{r[dim]}: {r['growth_pct']:+.1f}% ({fmt(r['prior_12m'])} → {fmt(r['last_12m'])}), "
            f"{r['share_pct']:.2f}% of recent {mlabel}"
            for _, r in df.head(5).iloc[1:].iterrows()
            if pd.notna(r["growth_pct"])
        ]
        if not declining.empty:
            lost = (declining["prior_12m"] - declining["last_12m"]).sum()
            bullets.append(f"Combined, the declining {plural(noun(dim))} lost {fmt(lost)} of {mlabel} year over year.")
        how = [f"Compared `{spec.measure}` per `{dim}` in the 12 months up to the latest date with the 12 months before.",
               "Ranked by growth rate (most negative first); share shows each one's weight in recent revenue."]
        return head + "\n\n" + "\n".join(f"- {b}" for b in bullets) + self._how(spec, entry, how)

    def _w_trend(self, spec: Spec, results: list[dict[str, Any]], entry: Any) -> str:
        df = self._rows(results[0])
        if df.empty:
            return "No dated rows to plot." + self._how(spec, entry)
        val = df.columns[-1]
        mlabel = pretty(val).lower().replace("row count", "rows")
        g = spec.grain
        if spec.breakdown:
            tot = df.groupby(spec.breakdown)[val].sum().sort_values(ascending=False)
            first, last = df["period"].min(), df["period"].max()
            per = df.pivot_table(index="period", columns=spec.breakdown, values=val, aggfunc="sum").sort_index()
            growth = ((per.iloc[-1] - per.iloc[0]) / per.iloc[0].abs()).sort_values()
            head = (f"Across {len(per)} {g}s, **{tot.index[0]}** contributes the most {mlabel} (**{fmt(tot.iloc[0])}**), "
                    f"and **{growth.index[-1]}** grew fastest from the first to the last {g}.")
            bullets = [f"{k}: {fmt(v)} total" for k, v in tot.head(5).items()]
            bullets.append(f"Period covered: {str(first)[:10]} to {str(last)[:10]}.")
            return head + "\n\n" + "\n".join(f"- {b}" for b in bullets) + self._how(spec, entry, [f"Summed `{val}` per {g} and `{spec.breakdown}`."])
        s = pd.Series(df[val].to_numpy(), index=pd.to_datetime(df["period"]))
        peak, low = s.idxmax(), s.idxmin()
        label = (lambda d: d.strftime("%b %Y")) if g in ("month", "quarter") else (lambda d: d.strftime("%d %b %Y"))
        bullets = [
            f"Peak: {label(peak)} at {fmt(s.max())}; lowest: {label(low)} at {fmt(s.min())}.",
            f"Average per {g}: {fmt(s.mean())}.",
        ]
        years = s.groupby(s.index.year).agg(["sum", "count"])
        full = years[years["count"] >= {"month": 12, "quarter": 4, "week": 50, "day": 360, "year": 1}[g]]
        head = f"{pretty(mlabel)} runs from **{fmt(s.iloc[0])}** in {label(s.index[0])} to **{fmt(s.iloc[-1])}** in {label(s.index[-1])}."
        if len(full) >= 2:
            a, b = full.iloc[-2], full.iloc[-1]
            yoy = (b["sum"] - a["sum"]) / abs(a["sum"])
            head = (f"{pretty(mlabel)} grew **{yoy * 100:+.1f}%** year over year "
                    f"({full.index[-2]}: {fmt(a['sum'])} → {full.index[-1]}: {fmt(b['sum'])}).")
        if g == "month" and len(s) >= 12:
            by_m = s.groupby(s.index.month).mean()
            top_m = by_m.sort_values(ascending=False).index[:2]
            names = ", ".join(pd.Timestamp(2000, int(mm), 1).strftime("%B") for mm in top_m)
            bullets.append(f"Seasonality: {names} are the strongest months on average.")
            med = s.median()
            spikes = s[s > 2 * med]
            if len(spikes):
                bullets.append(f"{label(spikes.index[0])} is unusually high ({fmt(spikes.iloc[0])}, {spikes.iloc[0] / med:.1f}× the median {g}) — "
                               "ask me to detect anomalies to see what drove it.")
        return head + "\n\n" + "\n".join(f"- {b}" for b in bullets) + self._how(
            spec, entry, [f"Grouped rows by {g} of `{spec.time_col}` and summed `{spec.measure or 'rows'}`."])

    def _w_anomalies(self, spec: Spec, results: list[dict[str, Any]], entry: Any) -> str:
        r = results[0]
        if "error" in r:
            return f"Anomaly detection failed: {r['error']}"
        items = r.get("anomalies", [])
        if not items:
            return "No anomalies stood out above the threshold." + self._how(spec, entry, [r.get("method", "")])
        head = f"{r['summary']} The most important ones:"
        bullets = []
        by_kind = {k: [a for a in items if a["kind"] == k] for k in ("outlier", "timeseries", "rule")}
        picked = by_kind["outlier"][:4] + by_kind["timeseries"][:3] + by_kind["rule"][:2]
        for a in picked:
            rec = a.get("record") or {}
            ident = next((f"{k} {v}" for k, v in rec.items() if k.endswith("_id") or k == "order_id"), None)
            tag = {"timeseries": "Period", "rule": "Rule", "outlier": "Outlier"}[a["kind"]]
            bullets.append(f"**{tag}{' · ' + ident if ident and tag != 'Period' else ''}**: {a['reason']}")
        tips = ("Negative quantities usually mean returns recorded as sales; extreme unit prices often come from a "
                "misplaced decimal; single-order spikes may be genuine bulk deals — confirm them with the source system.")
        return (head + "\n\n" + "\n".join(f"- {b}" for b in bullets) + f"\n\n{tips}"
                + self._how(spec, entry, [r.get("method", "")]))

    def _w_forecast(self, spec: Spec, results: list[dict[str, Any]], entry: Any) -> str:
        r = results[0]
        if "error" in r:
            return f"I couldn't build a forecast: {r['error']}"
        fc = self._rows(r["forecast"])
        total = fc["forecast"].sum()
        unit = r.get("frequency", "period")
        head = (f"Over the next {len(fc)} {unit}s I expect about **{fmt(total)}** of {pretty(spec.measure or 'rows').lower()} "
                f"(**{fmt(fc['forecast'].mean())}** per {unit} on average).")
        bullets = [f"{row['period']}: {fmt(row['forecast'])} (80% range {fmt(row['lower_80'])}–{fmt(row['upper_80'])})"
                   for _, row in fc.head(6).iterrows()]
        if r.get("backtest_mape_pct") is not None:
            bullets.append(f"Backtest error on held-out history: {r['backtest_mape_pct']:.1f}% MAPE.")
        how = [f"Model: {r['method']}.", *r.get("notes", [])]
        return head + "\n\n" + "\n".join(f"- {b}" for b in bullets) + self._how(spec, entry, how)

    def _w_quality(self, spec: Spec, results: list[dict[str, Any]], entry: Any) -> str:
        parts = []
        for r in results:
            if "error" in r:
                continue
            issues = r.get("issues", [])
            serious = [i for i in issues if i["severity"] != "info"]
            parts.append(f"**{r['table']}**: quality score **{r['quality_score']}/100**, {len(issues)} issue(s)")
            for i in (serious or issues)[:5]:
                col = f"`{i['column']}`: " if i.get("column") else ""
                parts.append(f"  - {col}{i['message']}")
        head = "Here's the data-quality check for the loaded tables:"
        return head + "\n\n" + "\n".join(f"- {p}" if not p.startswith("  ") else p for p in parts) + (
            "\n\nUse the fixes in the Data panel to remove duplicates or standardise labels."
            + self._how(spec, entry, ["Profiled every column: types, missing values, duplicates, label variants, negatives and extremes."]))

    def _w_correlation(self, spec: Spec, results: list[dict[str, Any]], entry: Any) -> str:
        df = self._rows(results[0])
        if df.empty:
            return "Not enough numeric columns to correlate."
        mat = df.set_index(df.columns[0])
        pairs = []
        cols = list(mat.columns)
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                if a in mat.index:
                    pairs.append((a, b, float(mat.loc[a, b])))
        pairs.sort(key=lambda x: -abs(x[2]))
        head = f"The strongest relationship is between **{pairs[0][0]}** and **{pairs[0][1]}** (r = **{pairs[0][2]:.2f}**)." if pairs else ""
        bullets = [f"{a} ↔ {b}: r = {v:.2f}" for a, b, v in pairs[1:6]]
        return head + "\n\n" + "\n".join(f"- {b}" for b in bullets) + self._how(spec, entry, ["Pearson correlation between numeric measures; correlation is not causation."])

    def _w_distribution(self, spec: Spec, results: list[dict[str, Any]], entry: Any) -> str:
        df = self._rows(results[0])
        if df.empty:
            return "No values to describe."
        r = df.iloc[0]
        skew = "right-skewed (a long tail of large values)" if r["mean"] > r["median"] * 1.2 else "fairly symmetric"
        head = f"The median {pretty(spec.measure or '').lower()} is **{fmt(r['median'])}**; the distribution is {skew}."
        bullets = [f"Middle 50% between {fmt(r['p25'])} and {fmt(r['p75'])}.", f"Range {fmt(r['min'])} to {fmt(r['max'])}; mean {fmt(r['mean'])}."]
        return head + "\n\n" + "\n".join(f"- {b}" for b in bullets) + self._how(spec, entry)

    def _w_count(self, spec: Spec, results: list[dict[str, Any]], entry: Any) -> str:
        df = self._rows(results[0])
        r = df.iloc[0]
        asked = f"distinct_{spec.dimension}" if spec.dimension else None
        if asked and asked in df.columns:
            head = (f"There are **{int(r[asked]):,} distinct {plural(noun(spec.dimension))}** in `{entry.name}` "
                    f"(across {int(r['rows']):,} rows).")
        else:
            head = f"`{entry.name}` has **{int(r['rows']):,} rows**."
        bullets = [f"{pretty(c.replace('distinct_', ''))}: {int(r[c]):,} distinct" for c in df.columns[1:] if c != asked]
        return head + "\n\n" + "\n".join(f"- {b}" for b in bullets) + self._how(spec, entry)

    def _w_relate(self, spec: Spec, results: list[dict[str, Any]], entry: Any) -> str:
        rels = self.p.session.relationships
        lines = [f"`{r.left_table}.{r.left_column}` → `{r.right_table}.{r.right_column}` ({r.cardinality}, {r.overlap:.0%} of keys match)" for r in rels[:6]]
        head = f"The {len(self.p.session.tables)} files connect through {len(rels)} shared key(s):"
        out = head + "\n\n" + "\n".join(f"- {line}" for line in lines)
        if results and "rows" in results[0]:
            df = self._rows(results[0])
            if not df.empty and len(df.columns) >= 2:
                top = df.iloc[0]
                out += (f"\n\nAs an example of cross-file analysis, joining them shows **{top.iloc[0]}** leads with "
                        f"**{fmt(top.iloc[1])}** ({top['pct_of_total']:.1f}% of the total).")
        return out + self._how(spec, entry, ["Detected join keys by matching column names and value overlap between files."])

    def _w_summary(self, spec: Spec, results: list[dict[str, Any]], entry: Any) -> str:
        ok = [r for r in results if "rows" in r]
        if not ok:
            return "I couldn't compute a summary for this table."
        kpi = self._rows(ok[0]).iloc[0]
        p = entry.profile
        m = spec.measure or (p.primary_measure.name if p.primary_measure else None)
        head = f"`{entry.name}` holds **{int(kpi['rows']):,} rows**"
        if m and f"total_{m}" in kpi:
            head += f" with total {pretty(m).lower()} of **{fmt(kpi[f'total_{m}'])}** (average {fmt(kpi[f'avg_{m}'])} per row)"
        if "first_date" in kpi:
            head += f", from {str(kpi['first_date'])[:10]} to {str(kpi['last_date'])[:10]}"
        head += "."
        bullets = []
        if len(ok) > 1:
            g = self._rows(ok[1])
            if not g.empty:
                d, v = g.columns[0], g.columns[1]
                bullets.append(f"Largest {pretty(d).lower()}: {g.iloc[0][d]} ({fmt(g.iloc[0][v])}, {g.iloc[0].get('pct_of_total', 0):.1f}%); smallest: {g.iloc[-1][d]} ({g.iloc[-1].get('pct_of_total', 0):.1f}%).")
        if len(ok) > 2:
            t = self._rows(ok[2])
            if len(t) >= 2:
                s = pd.Series(t.iloc[:, -1].to_numpy(), index=pd.to_datetime(t["period"]))
                years = s.groupby(s.index.year).agg(["sum", "count"])
                full = years[years["count"] >= 12]
                if len(full) >= 2:
                    yoy = (full["sum"].iloc[-1] - full["sum"].iloc[-2]) / abs(full["sum"].iloc[-2])
                    bullets.append(f"Year over year: {yoy * 100:+.1f}% ({full.index[-2]} → {full.index[-1]}).")
                bullets.append(f"Best month: {s.idxmax().strftime('%b %Y')} ({fmt(s.max())}); weakest: {s.idxmin().strftime('%b %Y')} ({fmt(s.min())}).")
        warn = [i for i in p.issues if i.severity != "info"]
        if warn:
            bullets.append(f"Data quality: score {p.quality_score}/100 — {warn[0].message.rstrip('.')}" + (f" (+{len(warn) - 1} more issues)." if len(warn) > 1 else "."))
        bullets.append("Next: ask which products are underperforming, or to detect anomalies.")
        return head + "\n\n" + "\n".join(f"- {b}" for b in bullets) + self._how(spec, entry)


# --------------------------------------------------------------------------- provider


class OfflineProvider:
    """Deterministic provider implementing the same streaming tool-call protocol as an LLM."""

    name = "offline"
    model = "heuristic-planner"

    async def stream(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        context: AgentContext | None = None,
    ) -> AsyncIterator[LLMEvent]:
        if context is None or not context.session.tables:
            yield LLMEvent(type="text", text="Upload a CSV file (or load the sample data) and I'll analyse it.")
            yield LLMEvent(type="done", stop_reason="stop")
            return
        planner = Planner(context.session)
        # Locate the current turn: everything after the last real user question.
        idx = max(i for i, m in enumerate(messages) if m.role == "user" and m.content == context.question)
        prior = next((m.meta.get("offline_spec") for m in reversed(messages[:idx]) if m.role == "assistant" and m.meta.get("offline_spec")), None)
        spec, calls = planner.plan(context.question, prior)
        results: list[dict[str, Any]] = []
        for m in messages[idx + 1:]:
            if m.role == "tool":
                try:
                    results.append(json.loads(m.content))
                except json.JSONDecodeError:
                    results.append({"error": m.content})
        meta = {"offline_spec": asdict(spec)}
        k = len(results)
        if tools and k < len(calls):
            call = calls[k]
            args = dict(call.args)
            if call.uses_previous_result and results and results[-1].get("result_id"):
                args["result_id"] = results[-1]["result_id"]
            elif call.uses_previous_result:
                # The query failed; skip the chart and answer with what we have.
                calls = calls[:k]
            if k < len(calls):
                yield LLMEvent(type="meta", meta=meta)
                yield LLMEvent(type="tool_call", tool_call=ToolCall(id=f"off_{k}_{call.name}", name=call.name, arguments=args))
                yield LLMEvent(type="done", stop_reason="tool_use")
                return
        answer = AnswerWriter(planner).write(spec, results, calls)
        yield LLMEvent(type="meta", meta=meta)
        for i in range(0, len(answer), 48):
            yield LLMEvent(type="text", text=answer[i : i + 48])
        yield LLMEvent(type="done", stop_reason="stop")
