"""Turn arbitrary file and column names into safe, readable SQL identifiers."""

from __future__ import annotations

import keyword
import re
import unicodedata

# Keywords DuckDB reserves, which would need quoting as bare identifiers.
SQL_RESERVED = {
    "all", "and", "any", "as", "asc", "between", "by", "case", "cast", "check", "column",
    "constraint", "create", "cross", "default", "delete", "desc", "distinct", "drop", "else",
    "end", "except", "exists", "false", "fetch", "for", "foreign", "from", "full", "group",
    "having", "in", "inner", "insert", "intersect", "into", "is", "join", "left", "like", "limit",
    "natural", "not", "null", "offset", "on", "or", "order", "outer", "primary", "references",
    "right", "select", "set", "table", "then", "to", "true", "union", "unique", "update", "using",
    "values", "when", "where", "window", "with", "pivot", "unpivot", "qualify",
}


def to_identifier(name: str, *, fallback: str = "column") -> str:
    """``"Order Date (UTC)"`` -> ``"order_date_utc"``."""
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)  # camelCase -> camel_Case
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower()
    text = re.sub(r"_+", "_", text)
    if not text:
        text = fallback
    if text[0].isdigit():
        text = f"{fallback[0]}_{text}"
    if text in SQL_RESERVED or keyword.iskeyword(text):
        text = f"{text}_"
    return text[:63]


def unique_identifiers(names: list[str], *, fallback: str = "column") -> list[str]:
    """Sanitise a list of names, de-duplicating with numeric suffixes."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for i, raw in enumerate(names):
        raw_s = "" if raw is None else str(raw)
        if not raw_s.strip() or raw_s.startswith("Unnamed:"):
            raw_s = f"{fallback}_{i + 1}"
        base = to_identifier(raw_s, fallback=fallback)
        candidate = base
        while candidate in seen:
            seen[base] += 1
            candidate = f"{base}_{seen[base]}"
        seen.setdefault(candidate, 1)
        seen.setdefault(base, 1)
        out.append(candidate)
    return out


def table_name_from_filename(filename: str, existing: set[str]) -> str:
    stem = re.sub(r"\.(csv|tsv|txt)$", "", filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1], flags=re.I)
    base = to_identifier(stem, fallback="table")
    name = base
    n = 2
    while name in existing:
        name = f"{base}_{n}"
        n += 1
    return name
