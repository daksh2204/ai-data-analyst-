"""Profiling, data-quality fixes, join detection, anomalies, forecasting, charts and dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analysis.anomalies import detect_anomalies
from app.analysis.charts import ChartSpec, build_chart
from app.analysis.dashboard import build_dashboard, suggest_questions
from app.analysis.forecasting import forecast_series
from app.core.errors import AppError
from app.data.profiling import apply_fix, profile_table
from app.sessions.store import Session

# --------------------------------------------------------------------------- profiling


def test_roles_are_inferred_from_the_sample_orders(session: Session):
    p = session.tables["sales_orders"].profile
    roles = {c.name: c.role for c in p.columns}
    assert roles["order_id"] == "identifier"
    assert roles["customer_id"] == "identifier"
    assert roles["order_date"] == "time"
    assert roles["region"] == "dimension"
    assert roles["revenue"] == "measure"
    assert p.primary_measure is not None and p.primary_measure.name == "revenue"
    assert p.primary_time is not None and p.primary_time.name == "order_date"


def test_quality_checks_find_the_injected_problems(session: Session):
    p = session.tables["sales_orders"].profile
    kinds = {(i.kind, i.column) for i in p.issues}
    assert ("duplicate_rows", None) in kinds
    assert ("duplicate_ids", "order_id") in kinds
    assert ("unexpected_negatives", "quantity") in kinds
    assert ("missing_values", "discount") in kinds
    assert p.duplicate_rows == 3
    assert 0 < p.quality_score < 100
    assert session.tables["customers"].profile.quality_score == 100


def test_inconsistent_labels_are_detected_and_fixable():
    df = pd.DataFrame({"dept": ["Sales", "sales ", "Engineering", "engineering", "Sales", "HR"] * 5, "x": range(30)})
    p = profile_table("t", df)
    issue = next(i for i in p.issues if i.kind == "inconsistent_labels")
    assert issue.column == "dept" and issue.fix == "normalize_labels"
    fixed, message = apply_fix(df, "normalize_labels", "dept")
    assert set(fixed["dept"]) == {"Sales", "Engineering", "HR"}
    assert "Standardised" in message


def test_drop_duplicates_fix(session: Session):
    df = session.tables["sales_orders"].df
    fixed, message = apply_fix(df, "drop_duplicates")
    assert len(fixed) == len(df) - 3
    assert "3" in message


def test_relationships_between_sample_files(session: Session):
    rels = {(r.left_table, r.left_column, r.right_table, r.right_column, r.cardinality) for r in session.relationships}
    assert ("sales_orders", "customer_id", "customers", "customer_id", "many-to-one") in rels
    assert ("sales_orders", "product_id", "products", "product_id", "many-to-one") in rels


# --------------------------------------------------------------------------- anomalies


def _orders_report(session: Session):
    entry = session.tables["sales_orders"]
    return detect_anomalies(
        entry.df,
        "sales_orders",
        measures=["quantity", "unit_price", "discount", "revenue"],
        context_columns=["order_id", "order_date", "product_id"],
        date_column="order_date",
        primary_measure="revenue",
    )


def test_anomaly_detector_finds_every_injected_anomaly_type(session: Session):
    report = _orders_report(session)
    items = [a.to_dict() for a in report.anomalies]
    reasons = " ".join(a["reason"] for a in items)

    # Bulk orders (quantity 450-600) and the x100 unit-price typos.
    big_qty = [a for a in items if a["kind"] == "outlier" and a["column"] == "quantity" and (a["value"] or 0) >= 450]
    assert len(big_qty) >= 3
    price = [a for a in items if a["column"] == "unit_price" and a["kind"] == "outlier"]
    assert len(price) >= 2
    # Negative quantities keyed as sales.
    negatives = [a for a in items if a["kind"] == "rule" and a["column"] == "quantity"]
    assert len(negatives) == 3
    # The flash-sale spike day.
    assert any(a["kind"] == "timeseries" and a.get("date") == "2024-11-29" for a in items)
    # Every flag comes with a human-readable explanation.
    assert all(a["reason"] for a in items)
    assert "typical" in reasons or "expected" in reasons
    assert report.chart is not None


def test_anomaly_detector_is_quiet_on_clean_data():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "day": pd.date_range("2024-01-01", periods=200, freq="D"),
        "amount": rng.normal(100, 5, 200).round(2),
    })
    report = detect_anomalies(df, "clean", measures=["amount"], context_columns=["day"], date_column="day",
                              primary_measure="amount")
    assert len(report.anomalies) <= 2


# --------------------------------------------------------------------------- forecasting


def test_monthly_forecast_has_ordered_intervals(session: Session):
    df = session.tables["sales_orders"].df
    result = forecast_series(df, "order_date", "revenue", periods=6, freq="M")
    f = result.forecast
    assert len(f) == 6
    assert f["period"].min() > df["order_date"].max()
    assert (f["lower_95"] <= f["lower_80"]).all()
    assert (f["lower_80"] <= f["forecast"]).all()
    assert (f["forecast"] <= f["upper_80"]).all()
    assert (f["upper_80"] <= f["upper_95"]).all()
    assert (f["forecast"] >= 0).all()
    assert result.backtest_mape is not None and result.backtest_mape < 50
    assert result.chart is not None


def test_forecast_rejects_bad_columns(session: Session):
    with pytest.raises(AppError):
        forecast_series(session.tables["sales_orders"].df, "nope", "revenue")


def test_forecast_learns_a_clean_trend():
    idx = pd.date_range("2021-01-01", periods=36, freq="MS")
    df = pd.DataFrame({"month": idx, "value": 100 + 5 * np.arange(36)})
    result = forecast_series(df, "month", "value", periods=3, freq="M")
    assert result.forecast["forecast"].iloc[0] == pytest.approx(280, rel=0.05)


# --------------------------------------------------------------------------- charts


@pytest.mark.parametrize("chart_type", ["bar", "horizontal_bar", "line", "area", "pie", "donut"])
def test_charts_build_from_a_result(chart_type: str):
    df = pd.DataFrame({"region": ["East", "West", "North"], "revenue": [30.0, 20.0, 10.0]})
    built = build_chart(df, ChartSpec(chart_type=chart_type, x="region", y=["revenue"], title="Revenue"))
    fig = built["figure"]
    assert fig["data"], fig
    assert fig["layout"]


def test_multi_series_and_scatter_charts():
    df = pd.DataFrame({
        "month": pd.date_range("2024-01-01", periods=4, freq="MS").repeat(2),
        "region": ["A", "B"] * 4,
        "revenue": [1, 2, 3, 4, 5, 6, 7, 8],
    })
    line = build_chart(df, ChartSpec(chart_type="line", x="month", y=["revenue"], color="region", title="t"))
    assert len(line["figure"]["data"]) == 2
    scatter = build_chart(
        pd.DataFrame({"a": range(10), "b": range(10)}), ChartSpec(chart_type="scatter", x="a", y=["b"], title="t")
    )
    assert scatter["figure"]["data"]


def test_chart_with_unknown_column_explains_the_problem():
    df = pd.DataFrame({"region": ["A"], "revenue": [1.0]})
    with pytest.raises(AppError) as exc:
        build_chart(df, ChartSpec(chart_type="bar", x="regoin", y=["revenue"], title="t"))
    assert "region" in exc.value.message  # lists the available columns


# --------------------------------------------------------------------------- dashboard


def test_dashboard_is_built_without_an_llm(session: Session):
    dash = build_dashboard(session.db, session.tables["sales_orders"].profile)
    assert dash["kpis"] and dash["charts"]
    titles = " ".join(c["title"] for c in dash["charts"])
    assert "month" in titles.lower()


def test_suggested_questions_match_the_data(session: Session):
    qs = suggest_questions([t.profile for t in session.tables.values()])
    assert "Which region generated the highest revenue?" in qs
    assert "Which products are underperforming?" in qs
    assert "What are the top five customers?" in qs
