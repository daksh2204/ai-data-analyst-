"""CSV validation, decoding, dialect sniffing and type inference."""

from __future__ import annotations

import pandas as pd
import pytest

from app.core.errors import FileTooLargeError, ValidationError
from app.data.ingestion import load_csv
from app.data.naming import table_name_from_filename, to_identifier, unique_identifiers
from tests.conftest import read_sample

LIMITS = {"max_bytes": 5 * 1024 * 1024, "max_rows": 100_000, "max_columns": 100}


def load(raw: bytes | str, name: str = "data.csv", existing: set[str] | None = None, **overrides):
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    return load_csv(raw, name, existing_tables=existing or set(), **{**LIMITS, **overrides})


def test_sample_orders_types_are_inferred():
    t = load(read_sample("sales_orders.csv"), "sales_orders.csv")
    df = t.df
    assert t.name == "sales_orders"
    assert pd.api.types.is_datetime64_any_dtype(df["order_date"])
    assert pd.api.types.is_numeric_dtype(df["revenue"])
    assert pd.api.types.is_numeric_dtype(df["quantity"])
    assert len(df) == 6462


def test_messy_file_is_cleaned_and_every_change_is_reported():
    t = load(read_sample("messy_hr_example.csv"), "messy_hr_example.csv")
    df = t.df
    # Headers are sanitised and de-duplicated ("Rating", "Rating" -> rating, rating_2).
    assert {"employee_id", "full_name", "salary", "bonus", "rating", "rating_1"} <= set(df.columns)
    assert t.original_columns["full_name"] == " Full Name "
    # Currency strings become numbers, percentages become fractions.
    assert df["salary"].max() == pytest.approx(1_250_000)
    assert df["salary"].min() == pytest.approx(-5000)
    assert df["bonus"].dropna().max() == pytest.approx(0.15)
    # Whitespace around values is trimmed and the blank spacer row is dropped.
    assert "Chen Wei" in set(df["full_name"])
    assert len(df) == 11
    messages = " ".join(n.message for n in t.notes)
    assert "blank row" in messages
    assert "Renamed columns" in messages
    assert "formatted numbers" in messages


def test_semicolon_delimited_latin1_file():
    raw = "Stadt;Umsatz;Datum\nMünchen;1200,5;2024-01-03\nKöln;800;2024-01-04\n".encode("latin-1")
    t = load(raw, "umsatz.csv")
    assert t.delimiter == ";"
    assert t.encoding.lower().replace("_", "-") in {"latin-1", "iso-8859-1", "cp1252", "windows-1252", "cp1250", "iso8859-1"}
    assert list(t.df.columns) == ["stadt", "umsatz", "datum"]
    assert "München" in set(t.df["stadt"])


def test_tsv_and_bom_are_handled():
    t = load("﻿a\tb\n1\t2\n3\t4\n", "x.tsv")
    assert t.delimiter == "\t"
    assert list(t.df.columns) == ["a", "b"]
    assert t.df["b"].sum() == 6


def test_identifier_like_codes_keep_leading_zeros():
    t = load("zip,amount\n02134,10\n00501,20\n90210,30\n")
    assert t.df["zip"].tolist() == ["02134", "00501", "90210"]


def test_mostly_numeric_column_with_text_is_flagged_not_coerced():
    rows = "\n".join(f"{i},{i * 10}" for i in range(30)) + "\n31,unknown\n32,tbd\n"
    t = load("id,value\n" + rows)
    assert not pd.api.types.is_numeric_dtype(t.df["value"])
    assert any(n.level == "warning" and n.column == "value" for n in t.notes)


@pytest.mark.parametrize(
    ("raw", "name", "error", "fragment"),
    [
        (b"a,b\n1,2\n", "report.xlsx", ValidationError, "isn't a CSV"),
        (b"   \n", "empty.csv", ValidationError, "empty"),
        (b"a,b,c\n", "header_only.csv", ValidationError, "no data rows"),
        (b"PK\x03\x04\x00\x00binary", "fake.csv", ValidationError, "binary"),
    ],
)
def test_unusable_files_are_rejected_with_a_clear_message(raw, name, error, fragment):
    with pytest.raises(error) as exc:
        load(raw, name)
    assert fragment in exc.value.message


def test_size_and_shape_limits():
    with pytest.raises(FileTooLargeError):
        load("a\n" + "1\n" * 100, max_bytes=50)
    with pytest.raises(ValidationError, match="rows"):
        load("a\n" + "1\n" * 20, max_rows=10)
    with pytest.raises(ValidationError, match="columns"):
        load(",".join(f"c{i}" for i in range(12)) + "\n" + ",".join("1" * 12) + "\n", max_columns=10)


def test_ragged_rows_are_skipped_and_reported():
    t = load("a,b\n1,2\n3,4,5,6\n7,8\n")
    assert len(t.df) == 2
    assert any("malformed" in n.message for n in t.notes)


def test_naming_helpers():
    assert to_identifier("Order Date (UTC)") == "order_date_utc"
    assert to_identifier("customerName") == "customer_name"
    assert to_identifier("2024 Sales") == "c_2024_sales"
    assert to_identifier("select") == "select_"
    assert unique_identifiers(["Region", "region", "", "Unnamed: 3"]) == ["region", "region_2", "column_3", "column_4"]
    assert table_name_from_filename("Q1 Sales.csv", {"q1_sales"}) == "q1_sales_2"
