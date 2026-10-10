"""Detect likely join keys between uploaded tables (for multi-file analysis)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from app.data.profiling import TableProfile


@dataclass
class Relationship:
    left_table: str
    left_column: str
    right_table: str
    right_column: str
    overlap: float  # share of left distinct values found on the right
    cardinality: str  # "many-to-one" | "one-to-many" | "one-to-one" | "many-to-many"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def sql_hint(self) -> str:
        return f"{self.left_table}.{self.left_column} = {self.right_table}.{self.right_column}"


def _candidate_pairs(a: TableProfile, b: TableProfile) -> list[tuple[str, str]]:
    pairs = []
    for ca in a.columns:
        if ca.role in ("measure", "text") or ca.dtype in ("float",):
            continue
        for cb in b.columns:
            if cb.role in ("measure", "text") or cb.dtype != ca.dtype:
                continue
            same = ca.name == cb.name
            # e.g. orders.customer_id <-> customers.id
            suffix = ca.name.endswith("_id") and cb.name == "id" and ca.name[:-3].rstrip("s") in b.name
            if same or suffix:
                pairs.append((ca.name, cb.name))
    return pairs


def detect_relationships(tables: dict[str, pd.DataFrame], profiles: dict[str, TableProfile]) -> list[Relationship]:
    rels: list[Relationship] = []
    names = list(tables)
    for i, ta in enumerate(names):
        for tb in names[i + 1 :]:
            for ca, cb in _candidate_pairs(profiles[ta], profiles[tb]):
                va = pd.Series(tables[ta][ca].dropna().unique())
                vb = pd.Series(tables[tb][cb].dropna().unique())
                if va.empty or vb.empty:
                    continue
                if va.dtype == object or pd.api.types.is_string_dtype(va):
                    va = va.astype(str).str.strip().str.lower()
                    vb = vb.astype(str).str.strip().str.lower()
                inter = len(set(va) & set(vb))
                overlap = inter / min(len(set(va)), len(set(vb)))
                if overlap < 0.6:
                    continue
                a_unique = tables[ta][ca].dropna().is_unique
                b_unique = tables[tb][cb].dropna().is_unique
                card = {
                    (False, True): "many-to-one",
                    (True, False): "one-to-many",
                    (True, True): "one-to-one",
                    (False, False): "many-to-many",
                }[(a_unique, b_unique)]
                rels.append(Relationship(ta, ca, tb, cb, round(overlap, 3), card))
    return rels
