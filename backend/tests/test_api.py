"""HTTP API: sessions, uploads, datasets, chat streaming, reports, auth and observability."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.conftest import SAMPLES, read_sample, sse_events


def test_health_and_config(client: TestClient):
    health = client.get("/api/health").json()
    assert health["status"] == "ok" and health["provider"] == "offline"
    config = client.get("/api/config").json()
    assert config["offline"] is True and config["auth_required"] is False
    assert config["limits"]["max_upload_mb"] == 50


def test_upload_validates_each_file_independently(client: TestClient):
    sid = client.post("/api/sessions").json()["id"]
    files = [
        ("files", ("sales_orders.csv", read_sample("sales_orders.csv"), "text/csv")),
        ("files", ("customers.csv", read_sample("customers.csv"), "text/csv")),
        ("files", ("notes.pdf", b"%PDF-1.4", "application/pdf")),
        ("files", ("empty.csv", b"", "text/csv")),
    ]
    body = client.post(f"/api/sessions/{sid}/datasets", files=files).json()
    ok = {r["filename"]: r["ok"] for r in body["results"]}
    assert ok == {"sales_orders.csv": True, "customers.csv": True, "notes.pdf": False, "empty.csv": False}
    errors = {r["filename"]: r["error"]["message"] for r in body["results"] if not r["ok"]}
    assert "isn't a CSV" in errors["notes.pdf"]
    names = [d["name"] for d in body["session"]["datasets"]]
    assert names == ["sales_orders", "customers"]
    assert body["session"]["relationships"][0]["left_column"] == "customer_id"


def test_dataset_detail_fix_and_delete(client: TestClient, api_session: str):
    base = f"/api/sessions/{api_session}"
    detail = client.get(f"{base}/datasets/sales_orders").json()
    assert detail["profile"]["row_count"] == 6462
    assert len(detail["preview"]["rows"]) == 100

    fixed = client.post(f"{base}/datasets/sales_orders/fix", json={"fix": "drop_duplicates"}).json()
    assert fixed["dataset"]["rows"] == 6459
    assert fixed["profile"]["duplicate_rows"] == 0

    assert client.get(f"{base}/datasets/nope").status_code == 404
    after = client.delete(f"{base}/datasets/regional_targets").json()
    assert "regional_targets" not in [d["name"] for d in after["datasets"]]


def test_direct_query_endpoint_is_read_only(client: TestClient, api_session: str):
    base = f"/api/sessions/{api_session}"
    ok = client.post(f"{base}/query", json={"sql": "SELECT COUNT(*) AS n FROM customers"}).json()
    assert ok["rows"] == [[320]]
    bad = client.post(f"{base}/query", json={"sql": "DROP TABLE customers"})
    assert bad.status_code == 400
    assert bad.json()["error"]["code"] == "unsafe_code"


def test_dashboard_endpoint(client: TestClient, api_session: str):
    dash = client.get(f"/api/sessions/{api_session}/dashboard").json()
    assert dash["kpis"] and dash["charts"]


def test_chat_streams_events_and_keeps_history(client: TestClient, api_session: str):
    events = sse_events(client, api_session, "Which region generated the highest revenue?")
    types = [e["type"] for e in events]
    assert types[0] == "start" and types[-1] == "final"
    assert {"step_start", "step_end", "artifact", "token"} <= set(types)
    final = events[-1]["message"]
    assert "East" in final["answer"]

    sse_events(client, api_session, "Generate SQL for this analysis.")
    state = client.get(f"/api/sessions/{api_session}").json()
    assert [m["role"] for m in state["messages"]] == ["user", "assistant", "user", "assistant"]
    assert "```sql" in state["messages"][-1]["answer"]

    reset = client.post(f"/api/sessions/{api_session}/reset").json()
    assert reset["messages"] == [] and reset["datasets"]


def test_chat_requires_data_and_valid_input(client: TestClient):
    sid = client.post("/api/sessions").json()["id"]
    r = client.post(f"/api/sessions/{sid}/chat", json={"message": "hi"})
    assert r.status_code == 422 and "Upload" in r.json()["error"]["message"]
    r = client.post(f"/api/sessions/{sid}/chat", json={"message": ""})
    assert r.status_code == 422


def test_report_export(client: TestClient, api_session: str):
    sse_events(client, api_session, "Show monthly sales trends.")
    html = client.get(f"/api/sessions/{api_session}/report?format=html")
    assert html.status_code == 200 and "text/html" in html.headers["content-type"]
    assert "Show monthly sales trends." in html.text and "attachment" in html.headers["content-disposition"]
    md = client.get(f"/api/sessions/{api_session}/report?format=md")
    assert md.text.startswith("#") and "```sql" in md.text


def test_unknown_session_is_a_clean_404(client: TestClient):
    r = client.get("/api/sessions/does-not-exist")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"


def test_metrics_and_request_ids(client: TestClient, api_session: str):
    sse_events(client, api_session, "Detect anomalies in the dataset.")
    r = client.get("/api/metrics")
    assert r.headers["x-request-id"]
    names = {c["name"] for c in r.json()["counters"]}
    assert {"http_requests", "tool_calls", "llm_calls", "sql_queries"} <= names
    prom = client.get("/api/metrics?format=prometheus").text
    assert "tool_calls_total" in prom


def test_access_token_protects_the_api():
    app = create_app(Settings(llm_provider="offline", app_access_token="s3cret", sample_data_dir=SAMPLES))
    c = TestClient(app)
    assert c.get("/api/health").status_code == 200  # health stays open for probes
    assert c.get("/api/config").json()["auth_required"] is True
    denied = c.post("/api/sessions")
    assert denied.status_code == 401 and denied.json()["error"]["code"] == "unauthorized"
    assert c.post("/api/sessions", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert c.post("/api/sessions", headers={"Authorization": "Bearer s3cret"}).status_code == 201


def test_session_file_limit():
    c = TestClient(create_app(Settings(llm_provider="offline", max_files_per_session=1)))
    sid = c.post("/api/sessions").json()["id"]
    files = [("files", (f"f{i}.csv", b"a,b\n1,2\n", "text/csv")) for i in range(2)]
    r = c.post(f"/api/sessions/{sid}/datasets", files=files)
    assert r.status_code == 429


def test_chat_is_rate_limited_per_client():
    c = TestClient(create_app(Settings(llm_provider="offline", chat_rate_limit_per_minute=2, sample_data_dir=SAMPLES)))
    sid = c.post("/api/sessions").json()["id"]
    c.post(f"/api/sessions/{sid}/datasets/samples")
    for _ in range(2):
        sse_events(c, sid, "Which region generated the highest revenue?")
    r = c.post(f"/api/sessions/{sid}/chat", json={"message": "again"})
    assert r.status_code == 429
    assert r.json()["error"]["details"]["retry_after_seconds"] > 0
