"""CSV ingestion: decoding, dialect sniffing, validation and type inference.

The goal is to accept the messy CSVs people actually export (BOMs, Windows-1252,
semicolons, currency strings, blank header cells, ragged rows) and turn them into a
typed DataFrame with SQL-safe column names, while recording every adjustment so the
user (and the LLM) can see what was changed.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from charset_normalizer import from_bytes

from app.core.errors import FileTooLargeError, ValidationError
from app.core.logging import get_logger
from app.data.naming import table_name_from_filename, unique_identifiers

log = get_logger(__name__)

ALLOWED_EXTENSIONS = (".csv", ".tsv", ".txt")
NULL_TOKENS = ["", "na", "n/a", "NA", "N/A", "null", "NULL", "None", "none", "nan", "NaN", "-", "--", "#N/A"]

_NUMERIC_RE = re.compile(r"^\(?[-+]?[$€£¥₹]?\s?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?\)?\s?%?$|^[-+]?\.\d+%?$")
_DATE_HINT_RE = re.compile(r"(date|time|_at$|_on$|day|month|period|timestamp|dt$)", re.I)
_DATEISH_VALUE_RE = re.compile(
    r"^\d{4}[-/.]\d{1,2}(?:[-/.]\d{1,2})?(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?|^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[ \-]\d",
    re.I,
)
_BOOL_MAP = {"true": True, "false": False, "yes": True, "no": False, "y": True, "n": False, "t": True, "f": False}


@dataclass
class IngestNote:
    """Something ingestion changed or noticed, surfaced in the UI."""

    level: str  # "info" | "warning"
    message: str
    column: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"level": self.level, "message": self.message, "column": self.column}


@dataclass
class LoadedTable:
    name: str
    filename: str
    df: pd.DataFrame
    original_columns: dict[str, str]  # sanitized -> original header
    encoding: str
    delimiter: str
    size_bytes: int
    fingerprint: str
    notes: list[IngestNote] = field(default_factory=list)


# --------------------------------------------------------------------------- decoding


def detect_encoding(raw: bytes) -> str:
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    try:
        raw[:2_000_000].decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        pass
    best = from_bytes(raw[:200_000]).best()
    if best is not None and best.encoding:
        return best.encoding
    return "latin-1"


def sniff_delimiter(sample: str, filename: str) -> str:
    if filename.lower().endswith(".tsv"):
        return "\t"
    lines = [ln for ln in sample.splitlines()[:50] if ln.strip()]
    if not lines:
        return ","
    try:
        dialect = csv.Sniffer().sniff("\n".join(lines), delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        # Fall back to the candidate that appears most consistently per line.
        best, best_score = ",", -1.0
        for cand in (",", ";", "\t", "|"):
            counts = [ln.count(cand) for ln in lines]
            if min(counts) == 0:
                continue
            score = float(np.mean(counts)) - float(np.std(counts))
            if score > best_score:
                best, best_score = cand, score
        return best


# --------------------------------------------------------------------------- type inference


def _numeric_match_ratio(s: pd.Series) -> float:
    values = s.dropna().astype(str).str.strip()
    if values.empty:
        return 0.0
    sample = values.sample(5000, random_state=0) if len(values) > 5000 else values
    return float(sample.str.match(_NUMERIC_RE).mean())


def _try_numeric(s: pd.Series) -> pd.Series | None:
    values = s.dropna().astype(str).str.strip()
    if values.empty:
        return None
    sample = values.sample(5000, random_state=0) if len(values) > 5000 else values
    if float(sample.str.match(_NUMERIC_RE).mean()) < 0.97:
        return None
    # Leading zeros ("02134", "007") mean an identifier or code, not a quantity.
    if sample.str.match(r"^0\d+$").mean() > 0.02:
        return None
    is_pct = sample.str.endswith("%").mean() > 0.5
    cleaned = (
        s.astype("string")
        .str.strip()
        .str.replace(r"[$€£¥₹,\s%]", "", regex=True)
        .str.replace(r"^\((.*)\)$", r"-\1", regex=True)  # (123) accounting negatives
    )
    out = pd.to_numeric(cleaned, errors="coerce")
    if is_pct:
        out = out / 100.0
    # Only accept if we didn't destroy many values.
    if out.notna().sum() < 0.97 * s.notna().sum():
        return None
    return out


def _try_datetime(s: pd.Series, name: str) -> pd.Series | None:
    values = s.dropna().astype(str).str.strip()
    if values.empty:
        return None
    sample = values.head(500)
    looks_dateish = sample.str.contains(_DATEISH_VALUE_RE).mean()
    if looks_dateish < 0.9 and not (_DATE_HINT_RE.search(name) and looks_dateish > 0.5):
        return None
    for kwargs in ({"format": "ISO8601"}, {"format": "mixed", "dayfirst": False}, {"format": "mixed", "dayfirst": True}):
        try:
            parsed = pd.to_datetime(s, errors="coerce", **kwargs)
        except (ValueError, TypeError):
            continue
        if parsed.notna().sum() >= 0.95 * s.notna().sum():
            if getattr(parsed.dt, "tz", None) is not None:
                parsed = parsed.dt.tz_convert(None)
            return parsed
    return None


def _try_bool(s: pd.Series) -> pd.Series | None:
    values = s.dropna().astype(str).str.strip().str.lower()
    if values.empty or values.nunique() > 2 or not set(values.unique()) <= set(_BOOL_MAP):
        return None
    return s.astype(str).str.strip().str.lower().map(_BOOL_MAP).astype("boolean").where(s.notna())


def infer_types(df: pd.DataFrame) -> tuple[pd.DataFrame, list[IngestNote]]:
    notes: list[IngestNote] = []
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if s.dtype != object and not pd.api.types.is_string_dtype(s):
            continue
        # Trim surrounding whitespace in text values (a common export artefact).
        stripped = s.astype("string").str.strip()
        n_trimmed = int((stripped != s.astype("string")).fillna(False).sum())
        stripped = stripped.replace("", pd.NA)
        if n_trimmed:
            notes.append(IngestNote("info", f"Trimmed surrounding whitespace in {n_trimmed:,} value(s).", col))
        s_obj = stripped.astype(object).where(stripped.notna(), None)

        if (num := _try_numeric(s_obj)) is not None:
            out[col] = num
            if s_obj.dropna().astype(str).str.contains(r"[$€£¥₹,%()]").any():
                notes.append(IngestNote("info", "Parsed formatted numbers (currency, separators or %) as numeric.", col))
            continue
        if (dt := _try_datetime(s_obj, col)) is not None:
            out[col] = dt
            continue
        if (b := _try_bool(s_obj)) is not None:
            out[col] = b
            continue
        ratio = _numeric_match_ratio(s_obj)
        if 0.6 <= ratio < 0.97:
            bad = s_obj.dropna().astype(str)
            bad = bad[~bad.str.match(_NUMERIC_RE)].unique()[:3]
            examples = ", ".join(repr(b) for b in bad)
            notes.append(
                IngestNote(
                    "warning",
                    f"Mostly numeric ({ratio:.0%}) but kept as text because of values like {examples}.",
                    col,
                )
            )
        out[col] = stripped
    return out, notes


def _downcast_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise datetime columns to ns precision (DuckDB maps these to TIMESTAMP)."""
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype("datetime64[ns]")
    return df


# --------------------------------------------------------------------------- main entry


def load_csv(
    raw: bytes,
    filename: str,
    *,
    existing_tables: set[str],
    max_bytes: int,
    max_rows: int,
    max_columns: int,
) -> LoadedTable:
    """Validate and parse an uploaded CSV. Raises :class:`ValidationError` on unusable input."""
    if not filename.lower().endswith(ALLOWED_EXTENSIONS):
        raise ValidationError(
            f"'{filename}' isn't a CSV file. Upload files ending in .csv, .tsv or .txt.",
            details={"filename": filename},
        )
    if len(raw) > max_bytes:
        raise FileTooLargeError(
            f"'{filename}' is {len(raw) / 1_048_576:.1f} MB; the limit is {max_bytes // 1_048_576} MB.",
            details={"filename": filename},
        )
    if not raw.strip():
        raise ValidationError(f"'{filename}' is empty.", details={"filename": filename})
    if b"\x00" in raw[:4096] and not raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        raise ValidationError(
            f"'{filename}' looks like a binary file, not a CSV (for Excel files, export as CSV first).",
            details={"filename": filename},
        )

    notes: list[IngestNote] = []
    encoding = detect_encoding(raw)
    try:
        text = raw.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        encoding = "latin-1"
        text = raw.decode(encoding)
    if encoding not in ("utf-8", "utf-8-sig", "ascii"):
        notes.append(IngestNote("info", f"Decoded the file as {encoding}."))

    delimiter = sniff_delimiter(text[:20_000], filename)
    if delimiter != ",":
        notes.append(IngestNote("info", f"Detected {'tab' if delimiter == chr(9) else repr(delimiter)} as the delimiter."))

    read_kwargs: dict[str, Any] = {
        "sep": delimiter,
        "dtype": str,
        "keep_default_na": False,
        "na_values": NULL_TOKENS,
        "skip_blank_lines": True,
    }
    bad_lines: list[int] = []
    try:
        df = pd.read_csv(io.StringIO(text), **read_kwargs, engine="c", on_bad_lines="error")
    except pd.errors.ParserError:
        # Ragged file: keep rows that fit the header, report the ones with extra fields.
        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        header = next(reader, [])
        buf = io.StringIO()
        writer = csv.writer(buf, delimiter=delimiter)
        writer.writerow(header)
        for row in reader:
            if len(row) > len(header) and any(v.strip() for v in row[len(header):]):
                bad_lines.append(len(row))
                continue
            writer.writerow(row[: len(header)])
        df = pd.read_csv(io.StringIO(buf.getvalue()), **read_kwargs, engine="c")
    except pd.errors.EmptyDataError as exc:
        raise ValidationError(f"'{filename}' has no columns to read.", details={"filename": filename}) from exc

    if bad_lines:
        notes.append(IngestNote("warning", f"Skipped {len(bad_lines):,} malformed row(s) with the wrong number of fields."))
    if df.shape[1] == 0:
        raise ValidationError(f"'{filename}' has no columns.", details={"filename": filename})
    if df.empty:
        raise ValidationError(f"'{filename}' has a header but no data rows.", details={"filename": filename})
    if df.shape[1] == 1 and len(df) > 1 and df.columns[0].count(" ") > 3:
        notes.append(IngestNote("warning", "Only one column was found. Check that the file uses a standard delimiter."))
    if len(df) > max_rows:
        raise ValidationError(
            f"'{filename}' has {len(df):,} rows; the limit is {max_rows:,}.", details={"filename": filename}
        )
    if df.shape[1] > max_columns:
        raise ValidationError(
            f"'{filename}' has {df.shape[1]:,} columns; the limit is {max_columns:,}.", details={"filename": filename}
        )

    # Drop columns and rows that are completely empty (trailing delimiters, spacer rows).
    empty_cols = [c for c in df.columns if df[c].isna().all() and str(c).startswith("Unnamed:")]
    if empty_cols:
        df = df.drop(columns=empty_cols)
        notes.append(IngestNote("info", f"Dropped {len(empty_cols)} empty unnamed column(s)."))
    n_before = len(df)
    df = df.dropna(how="all")
    if len(df) < n_before:
        notes.append(IngestNote("info", f"Dropped {n_before - len(df):,} completely blank row(s)."))

    original = [str(c) for c in df.columns]
    sanitized = unique_identifiers(original)
    renamed = {o: s for o, s in zip(original, sanitized, strict=True) if o != s}
    df.columns = sanitized
    if renamed:
        sample = ", ".join(f"'{o}' → {s}" for o, s in list(renamed.items())[:4])
        more = f" and {len(renamed) - 4} more" if len(renamed) > 4 else ""
        notes.append(IngestNote("info", f"Renamed columns to SQL-friendly names: {sample}{more}."))

    df, type_notes = infer_types(df)
    notes.extend(type_notes)
    df = _downcast_datetimes(df).reset_index(drop=True)

    name = table_name_from_filename(filename, existing_tables)
    fingerprint = hashlib.sha256(raw).hexdigest()[:16]
    log.info("csv loaded", extra={"table": name, "rows": len(df), "cols": df.shape[1], "encoding": encoding})
    return LoadedTable(
        name=name,
        filename=filename,
        df=df,
        original_columns=dict(zip(sanitized, original, strict=True)),
        encoding=encoding,
        delimiter=delimiter,
        size_bytes=len(raw),
        fingerprint=fingerprint,
        notes=notes,
    )
