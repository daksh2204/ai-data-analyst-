"""Build Plotly figure specs (plain JSON) from tabular results.

Traces are built by hand rather than through ``plotly.express`` so the payload is plain
lists (no binary-encoded arrays), the dependency footprint stays small, and every mark
follows one spec. Colours are *not* baked in: traces carry a semantic ``meta.role`` and
the client applies the light or dark palette (see ``frontend/src/lib/chartTheme.ts``).
"""

from __future__ import annotations

import contextlib
import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.core.errors import ToolError

CHART_TYPES = (
    "bar", "horizontal_bar", "stacked_bar", "grouped_bar", "line", "area", "pie", "donut",
    "scatter", "histogram", "box", "heatmap",
)
MAX_CATEGORIES = 30
MAX_POINTS = 5000
MAX_SERIES = 8  # categorical palette size; more series fold into "Other"


@dataclass
class ChartSpec:
    chart_type: str
    x: str | None = None
    y: list[str] = field(default_factory=list)
    color: str | None = None
    title: str = ""
    x_label: str | None = None
    y_label: str | None = None
    sort: str | None = None  # "asc" | "desc" | None
    top_n: int | None = None


def pretty(name: str) -> str:
    text = re.sub(r"_+", " ", str(name)).strip()
    return text[:1].upper() + text[1:] if text else text


def resolve_column(df: pd.DataFrame, name: str | None, *, required: bool = True) -> str | None:
    if name is None or name == "":
        if required:
            raise ToolError("A column name is required for this chart.")
        return None
    if name in df.columns:
        return name
    norm = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in df.columns}
    key = re.sub(r"[^a-z0-9]", "", str(name).lower())
    if key in norm:
        return str(norm[key])
    raise ToolError(f"Column '{name}' isn't in the result. Available columns: {', '.join(map(str, df.columns))}.")


def _clean(values: Any) -> list[Any]:
    out = []
    for v in values:
        if v is None:
            out.append(None)
        elif isinstance(v, (float, np.floating)):
            out.append(None if (math.isnan(v) or math.isinf(v)) else round(float(v), 6))
        elif isinstance(v, (np.integer,)):
            out.append(int(v))
        elif isinstance(v, (pd.Timestamp,)):
            out.append(v.isoformat() if v.time() != pd.Timestamp(0).time() else v.date().isoformat())
        elif hasattr(v, "isoformat"):
            out.append(v.isoformat())
        elif v is pd.NA or v is pd.NaT:
            out.append(None)
        else:
            out.append(v if isinstance(v, (int, float, str, bool)) else str(v))
    return out


def _fold_series(df: pd.DataFrame, color: str, value_cols: list[str]) -> pd.DataFrame:
    """Keep the largest MAX_SERIES-1 groups; fold the rest into 'Other'."""
    groups = df[color].astype(str)
    if groups.nunique() <= MAX_SERIES:
        return df.assign(**{color: groups})
    totals = df.assign(_g=groups).groupby("_g")[value_cols[0]].sum().sort_values(ascending=False)
    keep = set(totals.index[: MAX_SERIES - 1])
    return df.assign(**{color: np.where(groups.isin(keep), groups, "Other")})


def _aggregate_if_needed(df: pd.DataFrame, keys: list[str], values: list[str]) -> tuple[pd.DataFrame, bool]:
    if df.duplicated(subset=keys).any():
        numeric = [v for v in values if pd.api.types.is_numeric_dtype(df[v])]
        agg = df.groupby(keys, as_index=False, sort=False)[numeric].sum()
        return agg, True
    return df, False


def _base_layout(spec: ChartSpec, x: str | None, y: list[str]) -> dict[str, Any]:
    y_title = spec.y_label or (pretty(y[0]) if len(y) == 1 else "Value")
    return {
        "xaxis": {"title": {"text": spec.x_label or (pretty(x) if x else "")}, "automargin": True},
        "yaxis": {"title": {"text": y_title}, "automargin": True, "rangemode": "tozero"},
        "showlegend": False,
        "hovermode": "closest",
        "margin": {"l": 56, "r": 16, "t": 16, "b": 48},
    }


def build_chart(df: pd.DataFrame, spec: ChartSpec) -> dict[str, Any]:
    if spec.chart_type not in CHART_TYPES:
        raise ToolError(f"Unsupported chart type '{spec.chart_type}'. Choose one of: {', '.join(CHART_TYPES)}.")
    if df.empty:
        raise ToolError("The result is empty, so there is nothing to chart.")
    builders = {
        "bar": _bar, "horizontal_bar": _bar, "stacked_bar": _bar, "grouped_bar": _bar,
        "line": _line, "area": _line, "pie": _pie, "donut": _pie, "scatter": _scatter,
        "histogram": _histogram, "box": _box, "heatmap": _heatmap,
    }
    figure, notes = builders[spec.chart_type](df.copy(), spec)
    return {"figure": figure, "notes": notes}


# --------------------------------------------------------------------------- builders


def _bar(df: pd.DataFrame, spec: ChartSpec) -> tuple[dict[str, Any], list[str]]:
    notes: list[str] = []
    x = resolve_column(df, spec.x)
    assert x is not None
    ys = [c for c in (resolve_column(df, y) for y in spec.y) if c] or [
        c for c in df.columns if c != x and pd.api.types.is_numeric_dtype(df[c])
    ][:1]
    if not ys:
        raise ToolError("A bar chart needs a numeric column for y.")
    color = resolve_column(df, spec.color, required=False)
    horizontal = spec.chart_type == "horizontal_bar"
    keys = [x] + ([color] if color else [])
    df, aggregated = _aggregate_if_needed(df, keys, ys)
    if aggregated:
        notes.append(f"Summed {', '.join(ys)} for repeated {x} values.")
    if color:
        df = _fold_series(df, color, ys)
    if not color and len(ys) == 1:
        if spec.sort:
            df = df.sort_values(ys[0], ascending=spec.sort == "asc")
        limit = spec.top_n or MAX_CATEGORIES
        if len(df) > limit:
            notes.append(f"Showing {limit} of {len(df)} categories.")
            df = df.head(limit)
    if pd.api.types.is_datetime64_any_dtype(df[x]):
        df = df.sort_values(x)

    traces: list[dict[str, Any]] = []
    if color:
        for i, (name, grp) in enumerate(df.groupby(color, sort=False)):
            traces.append(_bar_trace(grp[x], grp[ys[0]], str(name), i, horizontal))
    else:
        for i, y in enumerate(ys):
            traces.append(_bar_trace(df[x], df[y], pretty(y), i, horizontal))
    layout = _base_layout(spec, x, ys)
    layout["barmode"] = "stack" if spec.chart_type == "stacked_bar" else "group"
    layout["bargap"] = 0.35
    layout["barcornerradius"] = 4
    layout["showlegend"] = len(traces) > 1
    if horizontal:
        layout["xaxis"], layout["yaxis"] = layout["yaxis"], layout["xaxis"]
        layout["yaxis"]["autorange"] = "reversed"
        layout["yaxis"].pop("rangemode", None)
        layout["xaxis"]["rangemode"] = "tozero"
        layout["margin"]["l"] = 120
    else:
        layout["xaxis"]["type"] = "category" if not pd.api.types.is_datetime64_any_dtype(df[x]) else "date"
    return {"data": traces, "layout": layout}, notes


def _bar_trace(xv: pd.Series, yv: pd.Series, name: str, idx: int, horizontal: bool) -> dict[str, Any]:
    xs, ys = _clean(xv), _clean(yv)
    trace: dict[str, Any] = {
        "type": "bar",
        "name": name,
        "meta": {"role": "series", "index": idx},
        "x": ys if horizontal else xs,
        "y": xs if horizontal else ys,
        "orientation": "h" if horizontal else "v",
        "hovertemplate": ("%{y}: %{x:,.2f}" if horizontal else "%{x}: %{y:,.2f}") + f"<extra>{name}</extra>",
    }
    return trace


def _line(df: pd.DataFrame, spec: ChartSpec) -> tuple[dict[str, Any], list[str]]:
    notes: list[str] = []
    x = resolve_column(df, spec.x)
    assert x is not None
    ys = [c for c in (resolve_column(df, y) for y in spec.y) if c] or [
        c for c in df.columns if c != x and pd.api.types.is_numeric_dtype(df[c])
    ][:1]
    if not ys:
        raise ToolError("A line chart needs a numeric column for y.")
    color = resolve_column(df, spec.color, required=False)
    keys = [x] + ([color] if color else [])
    df, aggregated = _aggregate_if_needed(df, keys, ys)
    if aggregated:
        notes.append(f"Summed {', '.join(ys)} for repeated {x} values.")
    if color:
        df = _fold_series(df, color, ys)
    with contextlib.suppress(TypeError):
        df = df.sort_values(x)
    if len(df) > MAX_POINTS:
        notes.append(f"Down-sampled to {MAX_POINTS:,} points.")
        df = df.iloc[np.linspace(0, len(df) - 1, MAX_POINTS).astype(int)]
    area = spec.chart_type == "area"
    traces = []
    series = list(df.groupby(color, sort=False)) if color else [(pretty(y), df) for y in ys]
    for i, (name, grp) in enumerate(series):
        ycol = ys[0] if color else ys[i]
        trace = {
            "type": "scatter",
            "mode": "lines+markers" if len(grp) <= 60 else "lines",
            "name": str(name),
            "meta": {"role": "series", "index": i},
            "x": _clean(grp[x]),
            "y": _clean(grp[ycol]),
            "line": {"width": 2, "shape": "linear"},
            "marker": {"size": 7},
            "hovertemplate": "%{x}: %{y:,.2f}" + f"<extra>{name}</extra>",
        }
        if area:
            trace["fill"] = "tonexty" if (color and i > 0) else "tozeroy"
            if color:
                trace["stackgroup"] = "one"
        traces.append(trace)
    layout = _base_layout(spec, x, ys)
    layout["hovermode"] = "x unified"
    layout["showlegend"] = len(traces) > 1
    if pd.api.types.is_datetime64_any_dtype(df[x]):
        layout["xaxis"]["type"] = "date"
    return {"data": traces, "layout": layout}, notes


def _pie(df: pd.DataFrame, spec: ChartSpec) -> tuple[dict[str, Any], list[str]]:
    notes: list[str] = []
    names = resolve_column(df, spec.x or spec.color)
    assert names is not None
    values = resolve_column(df, spec.y[0] if spec.y else None, required=False) or next(
        (c for c in df.columns if c != names and pd.api.types.is_numeric_dtype(df[c])), None
    )
    if values is None:
        raise ToolError("A pie chart needs a numeric column for the slice sizes.")
    df, aggregated = _aggregate_if_needed(df, [names], [values])
    if aggregated:
        notes.append(f"Summed {values} for repeated {names} values.")
    if (df[values] < 0).any():
        raise ToolError("Pie charts can't show negative values; use a bar chart instead.")
    df = df.sort_values(values, ascending=False)
    if len(df) > MAX_SERIES:
        head, tail = df.head(MAX_SERIES - 1), df.iloc[MAX_SERIES - 1 :]
        other = pd.DataFrame({names: ["Other"], values: [tail[values].sum()]})
        df = pd.concat([head, other], ignore_index=True)
        notes.append(f"Grouped {len(tail)} smaller slices into 'Other'.")
    trace = {
        "type": "pie",
        "meta": {"role": "categorical"},
        "labels": _clean(df[names].astype(str)),
        "values": _clean(df[values]),
        "hole": 0.55 if spec.chart_type == "donut" else 0.0,
        "sort": False,
        "direction": "clockwise",
        "textinfo": "percent",
        "textposition": "inside",
        "insidetextorientation": "horizontal",
        "hovertemplate": "%{label}: %{value:,.2f} (%{percent})<extra></extra>",
    }
    layout = {"showlegend": True, "margin": {"l": 16, "r": 16, "t": 16, "b": 16}}
    return {"data": [trace], "layout": layout}, notes


def _scatter(df: pd.DataFrame, spec: ChartSpec) -> tuple[dict[str, Any], list[str]]:
    notes: list[str] = []
    x = resolve_column(df, spec.x)
    y = resolve_column(df, spec.y[0] if spec.y else None)
    assert x is not None and y is not None
    color = resolve_column(df, spec.color, required=False)
    if len(df) > MAX_POINTS:
        notes.append(f"Sampled {MAX_POINTS:,} of {len(df):,} points.")
        df = df.sample(MAX_POINTS, random_state=0)
    traces = []
    if color:
        df = df.assign(**{color: df[color].astype(str)})
        if df[color].nunique() > 3:
            # Scatter colours only stay distinguishable for 3 series; fold the rest.
            top = df[color].value_counts().index[:2]
            df = df.assign(**{color: np.where(df[color].isin(top), df[color], "Other")})
            notes.append("Grouped smaller colour groups into 'Other' to keep points distinguishable.")
        groups = list(df.groupby(color, sort=False))
    else:
        groups = [("Points", df)]
    for i, (name, grp) in enumerate(groups):
        traces.append({
            "type": "scatter",
            "mode": "markers",
            "name": str(name),
            "meta": {"role": "series", "index": i},
            "x": _clean(grp[x]),
            "y": _clean(grp[y]),
            "marker": {"size": 8, "opacity": 0.75},
            "hovertemplate": f"{pretty(x)}: %{{x:,.2f}}<br>{pretty(y)}: %{{y:,.2f}}<extra>{name}</extra>",
        })
    # Least-squares trend line with r, when both axes are numeric.
    if pd.api.types.is_numeric_dtype(df[x]) and pd.api.types.is_numeric_dtype(df[y]):
        sub = df[[x, y]].dropna()
        if len(sub) >= 3 and sub[x].nunique() > 1:
            slope, intercept = np.polyfit(sub[x].astype(float), sub[y].astype(float), 1)
            r = float(np.corrcoef(sub[x].astype(float), sub[y].astype(float))[0, 1])
            xs = np.linspace(sub[x].min(), sub[x].max(), 50)
            traces.append({
                "type": "scatter",
                "mode": "lines",
                "name": f"Trend (r = {r:.2f})",
                "meta": {"role": "trend"},
                "x": _clean(xs),
                "y": _clean(slope * xs + intercept),
                "line": {"width": 2, "dash": "dot"},
                "hoverinfo": "skip",
            })
            notes.append(f"Correlation r = {r:.2f}.")
    layout = _base_layout(spec, x, [y])
    layout["yaxis"].pop("rangemode", None)
    layout["showlegend"] = len(traces) > 1
    return {"data": traces, "layout": layout}, notes


def _histogram(df: pd.DataFrame, spec: ChartSpec) -> tuple[dict[str, Any], list[str]]:
    x = resolve_column(df, spec.x or (spec.y[0] if spec.y else None))
    assert x is not None
    trace = {
        "type": "histogram",
        "name": pretty(x),
        "meta": {"role": "series", "index": 0},
        "x": _clean(df[x].dropna()),
        "nbinsx": 40,
        "hovertemplate": "%{x}: %{y:,} rows<extra></extra>",
    }
    layout = _base_layout(spec, x, ["count"])
    layout["yaxis"]["title"]["text"] = "Rows"
    layout["bargap"] = 0.06
    return {"data": [trace], "layout": layout}, []


def _box(df: pd.DataFrame, spec: ChartSpec) -> tuple[dict[str, Any], list[str]]:
    y = resolve_column(df, spec.y[0] if spec.y else None)
    x = resolve_column(df, spec.x, required=False)
    assert y is not None
    trace: dict[str, Any] = {
        "type": "box",
        "name": pretty(y),
        "meta": {"role": "series", "index": 0},
        "y": _clean(df[y]),
        "boxpoints": "outliers",
        "marker": {"size": 6},
    }
    if x:
        trace["x"] = _clean(df[x].astype(str))
    layout = _base_layout(spec, x, [y])
    layout["yaxis"].pop("rangemode", None)
    return {"data": [trace], "layout": layout}, []


def _heatmap(df: pd.DataFrame, spec: ChartSpec) -> tuple[dict[str, Any], list[str]]:
    """Either a long table (x, y, value) pivoted, or an already-square matrix (e.g. correlations)."""
    x = resolve_column(df, spec.x, required=False)
    color_val = resolve_column(df, spec.y[1] if len(spec.y) > 1 else None, required=False)
    y_col = resolve_column(df, spec.y[0] if spec.y else None, required=False)
    is_corr = False
    if x and y_col and color_val:
        pivot = df.pivot_table(index=y_col, columns=x, values=color_val, aggfunc="sum")
    else:
        label_col = x or df.columns[0]
        pivot = df.set_index(label_col).select_dtypes("number")
        is_corr = bool(pivot.shape[0] == pivot.shape[1] and ((pivot.abs() <= 1.0001) | pivot.isna()).all().all())
    trace: dict[str, Any] = {
        "type": "heatmap",
        "meta": {"role": "diverging" if is_corr else "sequential"},
        "z": [_clean(row) for row in pivot.to_numpy()],
        "x": _clean(pivot.columns.astype(str)),
        "y": _clean(pivot.index.astype(str)),
        "hovertemplate": "%{y} × %{x}: %{z:,.2f}<extra></extra>",
        "xgap": 2,
        "ygap": 2,
    }
    if is_corr:
        trace.update({"zmin": -1, "zmax": 1, "texttemplate": "%{z:.2f}"})
    layout = {
        "xaxis": {"automargin": True, "type": "category"},
        "yaxis": {"automargin": True, "type": "category", "autorange": "reversed"},
        "margin": {"l": 100, "r": 16, "t": 16, "b": 80},
    }
    return {"data": [trace], "layout": layout}, []
