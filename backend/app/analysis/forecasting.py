"""Explainable forecasting with Holt-Winters exponential smoothing (pure numpy).

* Aggregates a measure to a regular frequency (auto: monthly / weekly / daily).
* Drops a trailing incomplete period and caps clear one-off periods (robust z > 5 against a
  rolling median) so a single spike doesn't bend the forecast.
* Fits additive Holt-Winters with a damped trend when there are at least two full
  seasons, otherwise damped Holt (level + trend). Smoothing parameters are chosen by grid
  search on one-step-ahead error.
* Reports a hold-out backtest (MAPE) and 80% / 95% prediction intervals.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.analysis.anomalies import robust_z
from app.analysis.charts import _clean, pretty
from app.core.errors import ToolError

SEASON = {"M": 12, "W": 52, "D": 7, "Q": 4}
FREQ_ALIAS = {"M": "MS", "W": "W-MON", "D": "D", "Q": "QS"}
FREQ_LABEL = {"M": "month", "W": "week", "D": "day", "Q": "quarter"}


@dataclass
class ForecastResult:
    freq: str
    method: str
    params: dict[str, float]
    history: pd.DataFrame  # period, value
    forecast: pd.DataFrame  # period, forecast, lower_80, upper_80, lower_95, upper_95
    backtest_mape: float | None
    notes: list[str] = field(default_factory=list)
    chart: dict[str, Any] | None = None

    def summary(self) -> dict[str, Any]:
        f = self.forecast
        return {
            "method": self.method,
            "frequency": FREQ_LABEL[self.freq],
            "params": self.params,
            "backtest_mape_pct": None if self.backtest_mape is None else round(self.backtest_mape, 1),
            "notes": self.notes,
            "forecast_total": round(float(f["forecast"].sum()), 2),
            "history_last_period_total": round(float(self.history["value"].iloc[-1]), 2),
        }


def _fit(y: np.ndarray, m: int | None, alpha: float, beta: float, gamma: float, phi: float) -> tuple[np.ndarray, tuple]:
    """Return one-step-ahead fitted values and final state."""
    n = len(y)
    fitted = np.full(n, np.nan)
    if m:
        level = y[:m].mean()
        trend = (y[m : 2 * m].mean() - y[:m].mean()) / m if n >= 2 * m else 0.0
        season = list(y[:m] - level)
        start = m
    else:
        level, trend, season, start = y[0], (y[1] - y[0]) if n > 1 else 0.0, [], 1
    for t in range(start, n):
        s = season[t - m] if m else 0.0
        fitted[t] = level + phi * trend + s
        prev_level = level
        level = alpha * (y[t] - s) + (1 - alpha) * (level + phi * trend)
        trend = beta * (level - prev_level) + (1 - beta) * phi * trend
        if m:
            season.append(gamma * (y[t] - level) + (1 - gamma) * s)
    return fitted, (level, trend, season)


def _predict(state: tuple, m: int | None, phi: float, horizon: int, n: int) -> np.ndarray:
    level, trend, season = state
    out = []
    damp = 0.0
    for h in range(1, horizon + 1):
        damp += phi**h
        s = season[n - m + ((h - 1) % m)] if m else 0.0
        out.append(level + damp * trend + s)
    return np.array(out)


def _select(y: np.ndarray, m: int | None) -> tuple[dict[str, float], np.ndarray, tuple]:
    best: tuple[float, dict[str, float], np.ndarray, tuple] | None = None
    grid = itertools.product(
        (0.1, 0.3, 0.5, 0.7, 0.9), (0.01, 0.05, 0.15, 0.3), (0.05, 0.15, 0.3) if m else (0.0,), (0.9, 0.98)
    )
    for alpha, beta, gamma, phi in grid:
        fitted, state = _fit(y, m, alpha, beta, gamma, phi)
        err = y - fitted
        sse = float(np.nansum(err**2))
        if best is None or sse < best[0]:
            best = (sse, {"alpha": alpha, "beta": beta, "gamma": gamma, "phi": phi}, fitted, state)
    assert best is not None
    return best[1], best[2], best[3]


def _auto_freq(dates: pd.Series) -> str:
    span = (dates.max() - dates.min()).days
    if span >= 540:
        return "M"
    if span >= 120:
        return "W"
    return "D"


def forecast_series(
    df: pd.DataFrame,
    date_column: str,
    value_column: str | None,
    *,
    periods: int = 6,
    freq: str = "auto",
    agg: str = "sum",
    cap_anomalies: bool = True,
) -> ForecastResult:
    if date_column not in df.columns:
        raise ToolError(f"Column '{date_column}' not found.")
    if value_column and value_column not in df.columns:
        raise ToolError(f"Column '{value_column}' not found.")
    dates = pd.to_datetime(df[date_column], errors="coerce")
    if dates.notna().sum() < 6:
        raise ToolError("Not enough dated rows to forecast.")
    periods = int(np.clip(periods, 1, 36))
    f = _auto_freq(dates) if freq in ("auto", None, "") else freq.upper()[0]
    if f not in SEASON:
        raise ToolError("frequency must be one of auto, D, W, M, Q.")
    notes: list[str] = []

    alias = FREQ_ALIAS[f]
    bucket = dates.dt.to_period({"MS": "M", "W-MON": "W-SUN", "D": "D", "QS": "Q"}[alias]).dt.start_time
    if value_column:
        vals = df[value_column].astype(float)
        series = vals.groupby(bucket).agg(agg if agg in ("sum", "mean", "count", "max", "min") else "sum")
    else:
        series = bucket.value_counts().sort_index().astype(float)
    full = pd.date_range(series.index.min(), series.index.max(), freq=alias)
    series = series.reindex(full, fill_value=0.0).astype(float)

    # Drop a trailing partial period (it would look like a collapse).
    last_start = series.index[-1]
    next_start = last_start + pd.tseries.frequencies.to_offset(alias)
    coverage = (dates.max().normalize() - last_start).days + 1
    expected_days = (next_start - last_start).days
    if len(series) > 4 and coverage < 0.9 * expected_days:
        notes.append(f"Excluded the incomplete last {FREQ_LABEL[f]} ({last_start.date()}).")
        series = series.iloc[:-1]

    if len(series) < 4:
        raise ToolError(f"Need at least 4 {FREQ_LABEL[f]}s of history to forecast; found {len(series)}.")

    y_raw = series.to_numpy()
    y = y_raw.copy()
    season_len = SEASON[f]
    if cap_anomalies and len(y) >= 8:
        base = pd.Series(y).rolling(5, center=True, min_periods=2).median().to_numpy()
        z = robust_z(pd.Series(y - base)).to_numpy()
        mask = np.abs(z) > 4.0
        # A peak that recurs a season earlier/later is seasonality, not a one-off: keep it when
        # its year-over-year ratio is in line with the typical year-over-year ratio.
        if len(y) >= 2 * season_len:
            with np.errstate(divide="ignore", invalid="ignore"):
                yoy = y[season_len:] / y[:-season_len]
            typical = float(np.nanmedian(yoy[np.isfinite(yoy)])) if np.isfinite(yoy).any() else 1.0
            for i in np.flatnonzero(mask):
                ratios = []
                if i - season_len >= 0 and y[i - season_len] > 0:
                    ratios.append((y[i] / y[i - season_len]) / typical)
                if i + season_len < len(y) and y[i] > 0:
                    ratios.append((y[i + season_len] / y[i]) / typical)
                if ratios and all(1 / 1.5 <= r <= 1.5 for r in ratios):
                    mask[i] = False
        if mask.any():
            y[mask] = base[mask]
            capped = ", ".join(str(d.date()) for d in series.index[mask][:4])
            notes.append(
                f"Replaced {int(mask.sum())} one-off {FREQ_LABEL[f]}(s) ({capped}) with the local median before fitting."
            )

    m = season_len if len(y) >= 2 * season_len else None
    params, fitted, state = _select(y, m)
    method = (
        f"Holt-Winters additive (damped trend, {m}-{FREQ_LABEL[f]} seasonality)" if m else "Holt's damped linear trend"
    )
    if not m:
        notes.append(f"Seasonality not modelled: needs at least {2 * SEASON[f]} {FREQ_LABEL[f]}s of history.")
    pred = _predict(state, m, params["phi"], periods, len(y))
    resid = (y - fitted)[~np.isnan(fitted)]
    sigma = float(np.std(resid, ddof=1)) if len(resid) > 2 else float(np.std(y)) * 0.25
    h = np.arange(1, periods + 1)
    width = sigma * np.sqrt(1 + (h - 1) * params["alpha"] ** 2)
    lower_bound = 0.0 if (y_raw >= 0).all() else -np.inf
    future_idx = pd.date_range(series.index[-1], periods=periods + 1, freq=alias)[1:]
    forecast = pd.DataFrame(
        {
            "period": future_idx,
            "forecast": np.maximum(pred, lower_bound),
            "lower_80": np.maximum(pred - 1.2816 * width, lower_bound),
            "upper_80": pred + 1.2816 * width,
            "lower_95": np.maximum(pred - 1.96 * width, lower_bound),
            "upper_95": pred + 1.96 * width,
        }
    )
    value_cols = ["forecast", "lower_80", "upper_80", "lower_95", "upper_95"]
    forecast[value_cols] = forecast[value_cols].round(2)

    # Hold-out backtest with the selected parameters.
    k = min(periods, max(1, len(y) // 4))
    backtest = None
    if len(y) - k >= (SEASON[f] + 2 if m else 4):
        _, st = _fit(y[:-k], m, params["alpha"], params["beta"], params["gamma"], params["phi"])
        bt = _predict(st, m, params["phi"], k, len(y) - k)
        actual = y_raw[-k:]
        nz = actual != 0
        if nz.any():
            backtest = float(np.mean(np.abs((actual[nz] - bt[nz]) / actual[nz])) * 100)
            notes.append(f"Backtest on the last {k} {FREQ_LABEL[f]}(s): mean absolute percentage error {backtest:.1f}%.")

    history = pd.DataFrame({"period": series.index, "value": y_raw})
    label = pretty(value_column) if value_column else "Rows"
    chart = {
        "data": [
            {
                "type": "scatter", "mode": "lines", "name": "95% interval", "meta": {"role": "band"},
                "x": _clean(list(forecast["period"]) + list(forecast["period"][::-1])),
                "y": _clean(list(forecast["upper_95"]) + list(forecast["lower_95"][::-1])),
                "fill": "toself", "line": {"width": 0}, "hoverinfo": "skip", "showlegend": True,
            },
            {
                "type": "scatter", "mode": "lines", "name": "80% interval", "meta": {"role": "band-inner"},
                "x": _clean(list(forecast["period"]) + list(forecast["period"][::-1])),
                "y": _clean(list(forecast["upper_80"]) + list(forecast["lower_80"][::-1])),
                "fill": "toself", "line": {"width": 0}, "hoverinfo": "skip", "showlegend": True,
            },
            {
                "type": "scatter", "mode": "lines+markers", "name": "Actual", "meta": {"role": "series", "index": 0},
                "x": _clean(history["period"]), "y": _clean(history["value"]), "line": {"width": 2},
                "marker": {"size": 6}, "hovertemplate": "%{x|%b %Y}: %{y:,.0f}<extra>Actual</extra>",
            },
            {
                "type": "scatter", "mode": "lines+markers", "name": "Forecast", "meta": {"role": "forecast"},
                "x": _clean([history["period"].iloc[-1], *forecast["period"]]),
                "y": _clean([history["value"].iloc[-1], *forecast["forecast"]]),
                "line": {"width": 2, "dash": "dash"}, "marker": {"size": 6},
                "hovertemplate": "%{x|%b %Y}: %{y:,.0f}<extra>Forecast</extra>",
            },
        ],
        "layout": {
            "xaxis": {"type": "date", "title": {"text": ""}},
            "yaxis": {"title": {"text": label}, "rangemode": "tozero"},
            "showlegend": True,
            "hovermode": "x unified",
            "margin": {"l": 56, "r": 16, "t": 16, "b": 40},
        },
    }
    return ForecastResult(
        freq=f,
        method=method,
        params=params,
        history=history,
        forecast=forecast,
        backtest_mape=backtest,
        notes=notes,
        chart=chart,
    )
