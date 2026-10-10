"""Explainable anomaly detection.

Three detectors, each producing a plain-language reason for every flag:

1. **Robust outliers** — modified z-score ``0.6745·(x − median) / MAD`` per numeric column
   (Iglewicz & Hoaglin; |z| > 3.5 by default). Right-skewed positive columns are scored
   on a log scale so ordinary large orders aren't flagged. Price-like columns are scored
   *within* each product/category group, so "£390 for a £39 mouse" is caught even though
   £390 is a normal price for a monitor.
2. **Rule checks** — negative values in columns that are otherwise non-negative
   (quantities, prices, revenue) — typically refunds or keying errors.
3. **Time-series deviations** — the primary measure is aggregated per day (or week),
   compared with a centred rolling median, and residuals are scored with the same
   robust z. Spikes are explained by their largest contributing record.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.analysis.charts import _clean, pretty

_NONNEG = re.compile(r"(qty|quantity|units|price|revenue|sales|amount|cost|count|total|spend|volume)", re.I)
_GROUP_PREF = ("product_name", "product", "product_id", "sku", "item", "category", "subcategory")


@dataclass
class Anomaly:
    kind: str  # outlier | rule | timeseries
    column: str
    value: float | None
    expected: float | None
    score: float
    severity: str  # high | medium
    reason: str
    row_index: int | None = None
    date: str | None = None
    record: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnomalyReport:
    table: str
    anomalies: list[Anomaly]
    method: str
    columns_checked: list[str]
    counts: dict[str, int]
    chart: dict[str, Any] | None
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "anomalies": [a.to_dict() for a in self.anomalies],
            "method": self.method,
            "columns_checked": self.columns_checked,
            "counts": self.counts,
            "summary": self.summary,
        }


def robust_z(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0 or np.isnan(mad):
        mean_ad = np.nanmean(np.abs(x - med))
        if mean_ad == 0 or np.isnan(mean_ad):
            return pd.Series(np.where(x != med, 99.0, 0.0), index=x.index)
        return (x - med) / (1.253314 * mean_ad)
    return 0.6745 * (x - med) / mad


def _fmt(v: float | None) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    av = abs(v)
    if float(v).is_integer() and av < 1e6:
        return f"{int(v):,}"
    if av >= 1e6:
        return f"{v / 1e6:,.2f}M"
    if av >= 1e4:
        return f"{v:,.0f}"
    if av >= 100:
        return f"{v:,.1f}"
    return f"{v:,.2f}"


def _severity(score: float, threshold: float) -> str:
    return "high" if abs(score) >= threshold * 2 else "medium"


def _record(df: pd.DataFrame, idx: int, context_columns: list[str], col: str) -> dict[str, Any]:
    cols = [*[c for c in context_columns if c in df.columns and c != col][:5], col]
    row = df.loc[idx, cols]
    return {c: v for c, v in zip(cols, _clean(row.tolist()), strict=True)}


_RATE_LIKE = re.compile(r"(discount|rate|pct|percent|ratio|share|probability|score)", re.I)
_PRICE_LIKE = re.compile(r"(price|cost)", re.I)


def _pick_group(df: pd.DataFrame, col: str, group_by: str | None) -> str | None:
    """Score amounts within product-like groups: "typical" depends on what was sold.

    Rates (discount, conversion rate...) are properties of the deal, not the product,
    so they are scored across the whole column.
    """
    if group_by and group_by in df.columns:
        return group_by
    if _RATE_LIKE.search(col):
        return None
    for g in _GROUP_PREF:
        if g in df.columns and 1 < df[g].nunique() <= 500 and df[g].value_counts().median() >= 10:
            return g
    return None


def _point_outliers(
    df: pd.DataFrame, col: str, threshold: float, group_by: str | None, context_columns: list[str]
) -> list[Anomaly]:
    s = df[col].astype(float)
    valid = s.dropna()
    if len(valid) < 10 or valid.nunique() < 3:
        return []
    group = _pick_group(df, col, group_by)
    positive = valid[valid > 0]
    use_log = bool(not _RATE_LIKE.search(col) and len(positive) >= 0.95 * len(valid) and positive.skew() > 1)
    # On a log scale, non-positive values are left to the rule checks, and only prices/costs
    # are checked on the low side (tiny orders are normal; a price at 1/10th is not).
    scored = np.log(s.where(s > 0)) if use_log else s
    if group:
        z = scored.groupby(df[group]).transform(robust_z)
        expected = s.groupby(df[group]).transform("median")
    else:
        z = robust_z(scored)
        expected = pd.Series(np.nanmedian(s), index=s.index)
    if use_log and not _PRICE_LIKE.search(col):
        flagged = z[z > threshold].dropna()
    else:
        flagged = z[z.abs() > threshold].dropna()
    p99 = float(np.nanpercentile(valid, 99))
    p01 = float(np.nanpercentile(valid, 1))
    # A value shared by many rows is a pattern (e.g. a standard 20% discount), not an anomaly.
    freq = valid.round(6).value_counts()
    common = set(freq[(freq >= 5) & (freq >= 0.005 * len(valid))].index)
    out = []
    for idx, score in flagged.items():
        value = float(s.loc[idx])
        if round(value, 6) in common:
            continue
        exp = float(expected.loc[idx])
        ratio = value / exp if exp else None
        where = f" for {df.loc[idx, group]}" if group else ""
        if ratio is not None and ratio > 0 and (ratio >= 2.5 or ratio <= 0.4):
            rel = (
                f"{ratio:.1f}× the typical {_fmt(exp)}{where}"
                if ratio >= 1
                else f"{1 / ratio:.1f}× below the typical {_fmt(exp)}{where}"
            )
        elif value > exp:
            rel = f"unusually high (typical {_fmt(exp)}{where}; 99% of values are ≤ {_fmt(p99)})"
        else:
            rel = f"unusually low (typical {_fmt(exp)}{where}; 99% of values are ≥ {_fmt(p01)})"
        scale = ", log scale" if use_log else ""
        reason = f"{pretty(col)} of {_fmt(value)} is {rel} (robust z = {score:.1f}{scale})."
        out.append(
            Anomaly(
                kind="outlier",
                column=col,
                value=round(value, 4),
                expected=round(exp, 4),
                score=round(float(score), 2),
                severity=_severity(float(score), threshold),
                reason=reason,
                row_index=int(idx),
                record=_record(df, int(idx), context_columns, col),
            )
        )
    return out


def _merge_by_row(found: list[Anomaly]) -> list[Anomaly]:
    """One entry per (row, kind): the strongest flag leads, other flagged columns are listed."""
    groups: dict[tuple[int | None, str], list[Anomaly]] = {}
    for a in found:
        groups.setdefault((a.row_index, a.kind), []).append(a)
    merged: list[Anomaly] = []
    for items in groups.values():
        items.sort(key=lambda a: -abs(a.score))
        lead = items[0]
        others = [a for a in items[1:] if a.column != lead.column]
        if others:
            extra = ", ".join(f"{pretty(a.column).lower()} {_fmt(a.value)}" for a in others[:3])
            lead.reason = lead.reason.rstrip(".") + f". Also unusual on this row: {extra}."
            for a in others:
                lead.record.setdefault(a.column, a.value)
        merged.append(lead)
    return merged


def _rule_negatives(df: pd.DataFrame, col: str, context_columns: list[str]) -> list[Anomaly]:
    if not _NONNEG.search(col):
        return []
    s = df[col]
    neg = s[s < 0]
    if neg.empty or len(neg) / max(s.notna().sum(), 1) > 0.05:
        return []
    out = []
    for idx, value in neg.items():
        out.append(
            Anomaly(
                kind="rule",
                column=col,
                value=float(value),
                expected=None,
                score=0.0,
                severity="medium",
                reason=(
                    f"Negative {pretty(col).lower()} ({_fmt(float(value))}) in a column that is otherwise never negative "
                    "— likely a refund or return keyed as a sale, or an entry error."
                ),
                row_index=int(idx),
                record=_record(df, int(idx), context_columns, col),
            )
        )
    return out


def _score_series(series: pd.Series, window: int, transform: str) -> tuple[pd.Series, pd.Series]:
    """Return (robust z of residuals, baseline on the original scale)."""
    if transform == "sqrt":
        t = np.sqrt(series.clip(lower=0))
    elif transform == "log":
        t = np.log1p(series.clip(lower=0))
    else:
        t = series
    base_t = t.rolling(window, center=True, min_periods=max(3, window // 3)).median()
    z = robust_z((t - base_t).dropna()).reindex(series.index)
    if transform == "sqrt":
        baseline = base_t**2
    elif transform == "log":
        baseline = np.expm1(base_t)
    else:
        baseline = base_t
    return z, baseline


def _timeseries(
    df: pd.DataFrame, date_col: str, measure: str | None, threshold: float, context_columns: list[str]
) -> tuple[list[Anomaly], dict[str, Any] | None, str]:
    dates = pd.to_datetime(df[date_col], errors="coerce")
    if dates.notna().sum() < 20:
        return [], None, ""
    span_days = (dates.max() - dates.min()).days
    if span_days < 14:
        return [], None, ""
    rows_per_day = dates.dt.normalize().value_counts().median()
    transactional = rows_per_day > 1.5  # many rows per day -> row count is a meaningful series

    found: dict[tuple[str, str], Anomaly] = {}
    daily_measure = None
    plans: list[tuple[str, str, int]] = []
    if span_days <= 1100:
        plans.append(("D", "day", 15))
    if span_days >= 7 * 16:
        plans.append(("W", "week", 9))

    day = dates.dt.normalize()
    for freq, label, window in plans:
        periods = day if freq == "D" else day - pd.to_timedelta(day.dt.weekday, unit="D")  # Monday starts
        full = pd.date_range(periods.min(), periods.max(), freq="D" if freq == "D" else "7D")
        series_defs: list[tuple[str, pd.Series, str]] = []
        if transactional:
            series_defs.append(("Rows", periods.value_counts().reindex(full, fill_value=0).astype(float), "sqrt"))
        if measure:
            vals = pd.Series(df[measure].astype(float).to_numpy(), index=periods.to_numpy())
            totals = vals.groupby(level=0).sum().reindex(full, fill_value=0.0)
            series_defs.append((pretty(measure), totals, "sqrt" if (totals >= 0).all() else "none"))
            if freq == "D":
                daily_measure = totals
        if freq != "D":
            # Partial weeks at either end of the data would read as false drops.
            first, last = full[0], full[-1]
            keep = full
            if day.min() > first:
                keep = keep[keep != first]
            if day.max() < last + pd.Timedelta(days=6):
                keep = keep[keep != last]
            series_defs = [(m, ser.reindex(keep), t) for m, ser, t in series_defs]
        for metric, series, transform in series_defs:
            if len(series) < 3 * window // 2:
                continue
            z, baseline = _score_series(series, window, transform)
            # Single-day dips are routine noise in transactional data (weekends, holidays), so
            # drops are judged per week (sustained); spikes are judged per day and per week.
            flagged = z[z > threshold] if freq == "D" else z[z.abs() > threshold]
            for ts, score in flagged.dropna().items():
                value, exp = float(series.loc[ts]), float(baseline.loc[ts])
                direction = "spike" if value > exp else "drop"
                if freq != "D":
                    week_end = ts + pd.Timedelta(days=6)
                    covered = [a for a in found.values() if a.date and ts <= pd.Timestamp(a.date) <= week_end
                               and ("spike" in a.reason) == (direction == "spike")]
                    if covered:
                        continue  # already explained by the daily view
                key = (ts.date().isoformat(), label)
                ratio = (value / exp) if exp else None
                if ratio is not None and ratio >= 1:
                    rel = f"{ratio:.1f}× the expected {_fmt(exp)}"
                elif ratio is not None:
                    rel = f"{(1 - ratio) * 100:.0f}% below the expected {_fmt(exp)}"
                else:
                    rel = f"vs an expected {_fmt(exp)}"
                when = ts.date().isoformat() if freq == "D" else f"the week of {ts.date().isoformat()}"
                metric_txt = "Number of records" if metric == "Rows" else metric
                reason = f"{metric_txt} on {when}: {_fmt(value)}, a {direction} — {rel} (robust z = {score:.1f})."
                record: dict[str, Any] = {}
                if direction == "spike" and measure and metric != "Rows" and freq == "D":
                    mask = (periods == ts).to_numpy()
                    if mask.any():
                        top_idx = df.loc[mask, measure].astype(float).idxmax()
                        share = float(df.loc[top_idx, measure]) / value if value else 0
                        if share >= 0.3:
                            record = _record(df, int(top_idx), context_columns, measure)
                            lead = next((f"{k} {v}" for k, v in record.items() if k.endswith("id")), "a single record")
                            reason += f" Driven mostly by {lead} ({share:.0%} of the {label}'s total)."
                if key in found:
                    prev = found[key]
                    prev.reason = prev.reason.rstrip(".") + "; " + reason[0].lower() + reason[1:]
                    prev.score = max(prev.score, round(float(score), 2), key=abs)
                    prev.severity = _severity(prev.score, threshold)
                    prev.record = prev.record or record
                    continue
                found[key] = Anomaly(
                    kind="timeseries",
                    column=measure or "row_count",
                    value=round(value, 2),
                    expected=round(exp, 2),
                    score=round(float(score), 2),
                    severity=_severity(float(score), threshold),
                    reason=reason,
                    date=ts.date().isoformat(),
                    record=record,
                )

    anomalies = sorted(found.values(), key=lambda a: -abs(a.score))
    chart = None
    label = "day"
    plot_series = daily_measure
    if plot_series is None and transactional:
        plot_series = dates.dt.normalize().value_counts().sort_index().astype(float)
    if plot_series is not None:
        _, baseline = _score_series(plot_series, 15, "sqrt" if (plot_series >= 0).all() else "none")
        day_points = [pd.Timestamp(a.date) for a in anomalies if a.date and "week of" not in a.reason]
        day_points = [d for d in day_points if d in plot_series.index]
        week_windows = [pd.Timestamp(a.date) for a in anomalies if a.date and "week of" in a.reason]
        metric = pretty(measure) if measure else "Records"
        chart = {
            "data": [
                {
                    "type": "scatter", "mode": "lines", "name": f"{metric} per day",
                    "meta": {"role": "series", "index": 0},
                    "x": _clean(plot_series.index), "y": _clean(plot_series.values), "line": {"width": 1.5},
                    "hovertemplate": "%{x|%d %b %Y}: %{y:,.0f}<extra></extra>",
                },
                {
                    "type": "scatter", "mode": "lines", "name": "Expected (rolling median)",
                    "meta": {"role": "baseline"},
                    "x": _clean(baseline.index), "y": _clean(baseline.values), "line": {"width": 2, "dash": "dot"},
                    "hoverinfo": "skip",
                },
                {
                    "type": "scatter", "mode": "markers", "name": "Flagged day",
                    "meta": {"role": "anomaly"},
                    "x": _clean(day_points), "y": _clean(plot_series.loc[day_points].values),
                    "marker": {"size": 10, "symbol": "circle"},
                    "hovertemplate": "%{x|%d %b %Y}: %{y:,.0f} (flagged)<extra></extra>",
                },
            ],
            "layout": {
                "xaxis": {"type": "date", "title": {"text": ""}},
                "yaxis": {"title": {"text": metric}, "rangemode": "tozero"},
                "showlegend": True,
                "hovermode": "closest",
                "margin": {"l": 56, "r": 16, "t": 16, "b": 40},
                "shapes": [
                    {
                        "type": "rect", "xref": "x", "yref": "paper", "x0": w.isoformat(),
                        "x1": (w + pd.Timedelta(days=7)).isoformat(), "y0": 0, "y1": 1,
                        "fillcolor": "rgba(208, 59, 59, 0.12)", "line": {"width": 0}, "layer": "below",
                        "name": "flagged-week",
                    }
                    for w in week_windows
                ],
            },
        }
    return anomalies, chart, f"{pretty(measure) if measure else 'record count'} over time"


def detect_anomalies(
    df: pd.DataFrame,
    table: str,
    *,
    measures: list[str],
    context_columns: list[str],
    date_column: str | None = None,
    primary_measure: str | None = None,
    group_by: str | None = None,
    sensitivity: float = 3.5,
    max_items: int = 40,
) -> AnomalyReport:
    threshold = float(np.clip(sensitivity, 2.0, 10.0))
    found: list[Anomaly] = []
    checked: list[str] = []
    for col in measures:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        checked.append(col)
        found.extend(_rule_negatives(df, col, context_columns))
        found.extend(_point_outliers(df, col, threshold, group_by, context_columns))

    # The same record is often extreme in several derived columns (revenue, cost, profit).
    point = sorted(_merge_by_row(found), key=lambda a: (a.kind != "rule", -abs(a.score)))

    ts_anoms: list[Anomaly] = []
    chart = None
    ts_label = ""
    if date_column and date_column in df.columns:
        ts_anoms, chart, ts_label = _timeseries(df, date_column, primary_measure, threshold, context_columns)

    counts = {
        "outlier": sum(1 for a in point if a.kind == "outlier"),
        "rule": sum(1 for a in point if a.kind == "rule"),
        "timeseries": len(ts_anoms),
    }
    # Interleave so both kinds are represented when truncating.
    budget_ts = min(len(ts_anoms), max(8, max_items // 3))
    selected = ts_anoms[:budget_ts] + point[: max_items - budget_ts]

    if chart is None and point:
        col = point[0].column
        s = df[col].astype(float)
        flagged_idx = [a.row_index for a in point if a.column == col and a.row_index is not None]
        chart = {
            "data": [
                {"type": "scatter", "mode": "markers", "name": pretty(col), "meta": {"role": "series", "index": 0},
                 "x": list(range(len(s))), "y": _clean(s.values), "marker": {"size": 5, "opacity": 0.5}},
                {"type": "scatter", "mode": "markers", "name": "Flagged", "meta": {"role": "anomaly"},
                 "x": flagged_idx, "y": _clean(s.loc[flagged_idx].values), "marker": {"size": 10}},
            ],
            "layout": {"xaxis": {"title": {"text": "Row"}}, "yaxis": {"title": {"text": pretty(col)}},
                       "showlegend": True, "margin": {"l": 56, "r": 16, "t": 16, "b": 40}},
        }

    parts = []
    if counts["outlier"]:
        parts.append(f"{counts['outlier']} unusual value(s)")
    if counts["rule"]:
        parts.append(f"{counts['rule']} negative value(s) where only positives are expected")
    if counts["timeseries"]:
        parts.append(f"{counts['timeseries']} unusual period(s) in {ts_label}")
    summary = ("Flagged " + ", ".join(parts) + ".") if parts else "No anomalies above the threshold."
    method = (
        f"Robust z-score (median/MAD, |z| > {threshold:g}) per numeric column; price-like columns scored within "
        "product groups; right-skewed columns on a log scale; negative-value rule for non-negative measures"
        + (
            "; daily and weekly totals (and record counts) vs a centred rolling median for time-series deviations."
            if ts_label
            else "."
        )
    )
    return AnomalyReport(
        table=table, anomalies=selected, method=method, columns_checked=checked, counts=counts, chart=chart, summary=summary
    )
