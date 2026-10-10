"""Column profiling, semantic roles and data-quality checks.

The profile serves three audiences: the UI (data panel), the LLM (schema context) and
the deterministic features (dashboard, suggested questions, offline planner). Roles:

* ``time``        — datetime columns
* ``measure``     — numeric quantities worth aggregating (revenue, quantity, ...)
* ``dimension``   — low/medium-cardinality categories to group by (region, product, ...)
* ``identifier``  — keys (order_id, customer_id) — counted, never summed
* ``text``        — free text / high-cardinality strings
* ``flag``        — booleans
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

_ID_NAME_RE = re.compile(r"(^id$|_id$|^id_|_key$|_code$|_no$|_number$|^uuid|_uuid$|^sku$|_sku$)", re.I)
_NONNEG_NAME_RE = re.compile(
    r"(qty|quantity|units|price|revenue|sales|amount|cost|count|total|spend|volume|weight|age|duration)", re.I
)
_MEASURE_PRIORITY = ["revenue", "sales", "amount", "total", "turnover", "income", "value", "profit", "spend", "price"]
_ENTITY_NAME_RE = re.compile(r"(_name$|^name$|title$)", re.I)


@dataclass
class ColumnProfile:
    name: str
    original_name: str
    dtype: str  # integer | float | datetime | date | boolean | string
    role: str
    null_count: int
    null_pct: float
    unique_count: int
    unique_pct: float
    sample_values: list[Any] = field(default_factory=list)
    top_values: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QualityIssue:
    severity: str  # info | warning | critical
    kind: str
    message: str
    column: str | None = None
    count: int | None = None
    fix: str | None = None  # id of an automatic fix, if one exists

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TableProfile:
    name: str
    row_count: int
    column_count: int
    columns: list[ColumnProfile]
    issues: list[QualityIssue]
    quality_score: int
    duplicate_rows: int
    memory_mb: float

    def column(self, name: str) -> ColumnProfile | None:
        return next((c for c in self.columns if c.name == name), None)

    def by_role(self, role: str) -> list[ColumnProfile]:
        return [c for c in self.columns if c.role == role]

    @property
    def primary_measure(self) -> ColumnProfile | None:
        measures = self.by_role("measure")
        for pref in _MEASURE_PRIORITY:
            for c in measures:
                if pref in c.name.lower():
                    return c
        return measures[0] if measures else None

    @property
    def primary_time(self) -> ColumnProfile | None:
        times = self.by_role("time")
        for c in times:
            if any(k in c.name for k in ("order", "date", "created", "transaction")):
                return c
        return times[0] if times else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": [c.to_dict() for c in self.columns],
            "issues": [i.to_dict() for i in self.issues],
            "quality_score": self.quality_score,
            "duplicate_rows": self.duplicate_rows,
            "memory_mb": self.memory_mb,
        }


def _jsonable(v: Any) -> Any:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else round(float(v), 6)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat() if v.time() != pd.Timestamp(0).time() else v.date().isoformat()
    if v is pd.NA or v is pd.NaT:
        return None
    return v


def _dtype_label(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_integer_dtype(s):
        return "integer"
    if pd.api.types.is_float_dtype(s):
        non_null = s.dropna()
        if len(non_null) and np.all(np.mod(non_null, 1) == 0) and non_null.abs().max() < 2**53:
            return "integer"
        return "float"
    if pd.api.types.is_datetime64_any_dtype(s):
        non_null = s.dropna()
        if len(non_null) and (non_null.dt.normalize() == non_null).all():
            return "date"
        return "datetime"
    return "string"


def _infer_role(name: str, s: pd.Series, dtype: str, n_rows: int, unique: int) -> str:
    if dtype in ("date", "datetime"):
        return "time"
    if dtype == "boolean":
        return "flag"
    unique_ratio = unique / max(n_rows, 1)
    if dtype in ("integer", "float"):
        if _ID_NAME_RE.search(name) and unique_ratio > 0.5:
            return "identifier"
        if dtype == "integer" and unique_ratio > 0.98 and n_rows > 50 and s.is_monotonic_increasing:
            return "identifier"
        if dtype == "integer" and unique <= 12 and re.search(r"(year|quarter|month|week|rank|level|tier)", name, re.I):
            return "dimension"
        return "measure"
    # strings
    if _ID_NAME_RE.search(name):
        return "identifier"
    if _ENTITY_NAME_RE.search(name) or unique <= max(50, int(0.05 * n_rows)):
        return "dimension"
    if unique_ratio > 0.9:
        return "identifier" if s.dropna().astype(str).str.len().median() < 24 else "text"
    return "text"


def profile_column(df: pd.DataFrame, name: str, original_name: str) -> ColumnProfile:
    s = df[name]
    n = len(s)
    dtype = _dtype_label(s)
    null_count = int(s.isna().sum())
    non_null = s.dropna()
    unique = int(non_null.nunique())
    role = _infer_role(name, s, dtype, n, unique)
    stats: dict[str, Any] = {}
    top_values: list[dict[str, Any]] = []

    if dtype in ("integer", "float") and len(non_null):
        x = non_null.astype(float)
        stats = {
            "min": _jsonable(x.min()),
            "max": _jsonable(x.max()),
            "mean": _jsonable(x.mean()),
            "median": _jsonable(x.median()),
            "std": _jsonable(x.std()) if len(x) > 1 else 0.0,
            "p25": _jsonable(x.quantile(0.25)),
            "p75": _jsonable(x.quantile(0.75)),
            "sum": _jsonable(x.sum()),
            "zeros": int((x == 0).sum()),
            "negatives": int((x < 0).sum()),
        }
    elif dtype in ("date", "datetime") and len(non_null):
        span_days = (non_null.max() - non_null.min()).days
        stats = {
            "min": _jsonable(non_null.min()),
            "max": _jsonable(non_null.max()),
            "span_days": int(span_days),
            "granularity": _granularity(non_null),
        }
    if role in ("dimension", "flag") or (dtype == "string" and unique <= 200):
        vc = non_null.astype(str).value_counts().head(10)
        top_values = [{"value": k, "count": int(v)} for k, v in vc.items()]

    sample_values = [_jsonable(v) for v in non_null.drop_duplicates().head(5).tolist()]
    return ColumnProfile(
        name=name,
        original_name=original_name,
        dtype=dtype,
        role=role,
        null_count=null_count,
        null_pct=round(100 * null_count / max(n, 1), 2),
        unique_count=unique,
        unique_pct=round(100 * unique / max(n, 1), 2),
        sample_values=sample_values,
        top_values=top_values,
        stats=stats,
    )


def _granularity(dates: pd.Series) -> str:
    uniq = pd.Series(dates.dropna().unique()).sort_values()
    if len(uniq) < 3:
        return "unknown"
    median_gap = uniq.diff().dropna().median()
    days = median_gap / pd.Timedelta(days=1)
    if days < 1 / 24:
        return "sub-hourly"
    if days < 1:
        return "hourly"
    if days < 6:
        return "daily"
    if days < 20:
        return "weekly"
    if days < 80:
        return "monthly"
    if days < 200:
        return "quarterly"
    return "yearly"


# --------------------------------------------------------------------------- quality


def _casing_variants(s: pd.Series) -> dict[str, list[str]]:
    values = pd.Series(s.dropna().astype(str).unique())
    if values.empty or len(values) > 500:
        return {}
    norm = values.str.strip().str.lower()
    groups: dict[str, list[str]] = {}
    for key, grp in values.groupby(norm):
        if len(grp) > 1:
            groups[str(key)] = sorted(grp.tolist())
    return groups


def quality_checks(df: pd.DataFrame, columns: list[ColumnProfile]) -> tuple[list[QualityIssue], int]:
    issues: list[QualityIssue] = []
    n = len(df)
    dup_rows = int(df.duplicated().sum())
    if dup_rows:
        issues.append(
            QualityIssue(
                "warning",
                "duplicate_rows",
                f"{dup_rows:,} row(s) are exact duplicates of another row.",
                count=dup_rows,
                fix="drop_duplicates",
            )
        )

    for c in columns:
        s = df[c.name]
        if c.null_count == n:
            issues.append(QualityIssue("critical", "empty_column", "Column is completely empty.", c.name, n))
            continue
        if c.null_pct > 0:
            sev = "critical" if c.null_pct >= 50 else "warning" if c.null_pct >= 5 else "info"
            issues.append(
                QualityIssue(sev, "missing_values", f"{c.null_count:,} missing value(s) ({c.null_pct:.1f}%).", c.name, c.null_count)
            )
        if c.unique_count == 1 and n > 1:
            issues.append(QualityIssue("info", "constant_column", "Every row has the same value.", c.name))
        if c.role == "identifier" and c.dtype == "string" and c.unique_count < n - c.null_count:
            dup_ids = (n - c.null_count) - c.unique_count
            if c.unique_pct > 90:
                issues.append(
                    QualityIssue(
                        "warning",
                        "duplicate_ids",
                        f"Looks like a unique key but {dup_ids:,} value(s) repeat.",
                        c.name,
                        dup_ids,
                    )
                )
        if c.dtype == "string" and c.role in ("dimension", "identifier", "text"):
            variants = _casing_variants(s)
            if variants:
                examples = "; ".join(" / ".join(repr(v) for v in vs) for vs in list(variants.values())[:3])
                issues.append(
                    QualityIssue(
                        "warning",
                        "inconsistent_labels",
                        f"The same label is written in different ways: {examples}.",
                        c.name,
                        len(variants),
                        fix="normalize_labels",
                    )
                )
        if c.dtype in ("integer", "float") and c.role == "measure":
            neg = int(c.stats.get("negatives", 0))
            if neg and _NONNEG_NAME_RE.search(c.name) and neg / max(n - c.null_count, 1) < 0.05:
                issues.append(
                    QualityIssue(
                        "warning",
                        "unexpected_negatives",
                        f"{neg:,} negative value(s) in a column that is normally positive (refunds or entry errors?).",
                        c.name,
                        neg,
                    )
                )
            x = s.dropna().astype(float)
            if len(x) >= 20:
                q1, q3 = np.percentile(x, [25, 75])
                iqr = q3 - q1
                if iqr > 0:
                    extreme = int(((x < q1 - 3 * iqr) | (x > q3 + 3 * iqr)).sum())
                    # Many "extremes" means a skewed distribution, not a data problem.
                    if 0 < extreme <= max(1, 0.01 * len(x)):
                        issues.append(
                            QualityIssue(
                                "info",
                                "extreme_values",
                                f"{extreme:,} extreme value(s) beyond 3×IQR — worth checking with anomaly detection.",
                                c.name,
                                extreme,
                            )
                        )
        if c.dtype in ("date", "datetime"):
            future = int((s > pd.Timestamp.now() + pd.Timedelta(days=1)).sum())
            if future:
                issues.append(QualityIssue("warning", "future_dates", f"{future:,} date(s) are in the future.", c.name, future))

    order = {"critical": 0, "warning": 1, "info": 2}
    issues.sort(key=lambda i: (order[i.severity], i.column or ""))
    return issues, dup_rows


def profile_table(name: str, df: pd.DataFrame, original_columns: dict[str, str] | None = None) -> TableProfile:
    original_columns = original_columns or {}
    cols = [profile_column(df, c, original_columns.get(c, c)) for c in df.columns]
    issues, dup_rows = quality_checks(df, cols)
    return TableProfile(
        name=name,
        row_count=len(df),
        column_count=df.shape[1],
        columns=cols,
        issues=issues,
        quality_score=_score(issues),
        duplicate_rows=dup_rows,
        memory_mb=round(df.memory_usage(deep=True).sum() / 1_048_576, 2),
    )


def _score(issues: list[QualityIssue]) -> int:
    weights = {"critical": 15, "warning": 5, "info": 1}
    return int(max(0, min(100, round(100 - sum(weights[i.severity] for i in issues)))))


# --------------------------------------------------------------------------- fixes


def apply_fix(df: pd.DataFrame, fix: str, column: str | None = None) -> tuple[pd.DataFrame, str]:
    """Apply an automatic data-quality fix. Returns the new frame and a description."""
    if fix == "drop_duplicates":
        before = len(df)
        out = df.drop_duplicates().reset_index(drop=True)
        return out, f"Removed {before - len(out):,} duplicate row(s)."
    if fix == "normalize_labels":
        if not column or column not in df.columns:
            raise ValueError("normalize_labels needs a column")
        s = df[column]
        stripped = s.astype("string").str.strip()
        # Map each variant to its most frequent spelling.
        canon = stripped.dropna().groupby(stripped.dropna().str.lower()).agg(lambda v: v.value_counts().index[0])
        mapped = stripped.str.lower().map(canon)
        out = df.copy()
        out[column] = mapped.where(stripped.notna(), pd.NA)
        changed = int((out[column].astype("string") != s.astype("string")).fillna(False).sum())
        return out, f"Standardised {changed:,} label(s) in {column}."
    raise ValueError(f"Unknown fix '{fix}'")
