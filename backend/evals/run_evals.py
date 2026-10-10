"""Evaluation harness for the analyst agent.

Runs every case in ``cases.yaml`` against the sample data with the configured provider,
checks the answer against ground truth computed by SQL, and writes a Markdown + JSON report.

Usage (from ``backend/``)::

    python -m evals.run_evals                       # provider from .env / environment
    python -m evals.run_evals --provider offline    # deterministic baseline, no API key
    python -m evals.run_evals --provider gemini --model gemini-2.5-flash --only top_region,anomalies

Exit code is non-zero when the pass rate is below ``--min-pass`` (useful in CI).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from app.agent.agent import AnalystAgent
from app.agent.llm.factory import create_provider
from app.agent.llm.offline import OfflineProvider
from app.config import Settings, get_settings
from app.core.cache import TTLCache
from app.core.logging import configure_logging
from app.data.ingestion import load_csv
from app.sessions.store import Session, SessionStore

HERE = Path(__file__).resolve().parent
SAMPLE_FILES = ("sales_orders.csv", "customers.csv", "products.csv", "regional_targets.csv")
_NUM_RE = re.compile(r"(?<![\w.])[-−]?\$?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?\s*([KMB])?(?![\w])", re.I)


@dataclass
class CaseResult:
    id: str
    question: str
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: list[str] = field(default_factory=list)
    tool_errors: int = 0
    answer: str = ""


def numbers_in(text: str) -> list[float]:
    out = []
    for m in _NUM_RE.finditer(text):
        whole = float(m.group(1).replace(",", ""))
        value = whole + (float("0." + m.group(2)) if m.group(2) else 0.0)
        scale = {"k": 1e3, "m": 1e6, "b": 1e9}.get((m.group(3) or "").lower(), 1.0)
        out.append(value * scale)
    return out


def contains_number(text: str, expected: float, rel: float = 0.015) -> bool:
    return any(abs(n - expected) <= max(abs(expected) * rel, 0.5) for n in numbers_in(text))


def build_session(settings: Settings) -> Session:
    store = SessionStore(ttl_minutes=60, max_sessions=4, query_timeout=settings.query_timeout_seconds,
                         query_max_rows=settings.query_max_rows)
    session = store.create()
    for fname in SAMPLE_FILES:
        raw = (settings.sample_data_dir / fname).read_bytes()
        loaded = load_csv(raw, fname, existing_tables=set(session.tables), max_bytes=settings.max_upload_bytes,
                          max_rows=settings.max_rows_per_file, max_columns=settings.max_columns_per_file)
        session.add_table(loaded)
    return session


def scalar(session: Session, sql: str) -> Any:
    df = session.db.execute(sql).df
    return df.iloc[0, 0]


async def ask(agent: AnalystAgent, question: str) -> dict[str, Any]:
    final: dict[str, Any] | None = None
    error: str | None = None
    async for ev in agent.run(question):
        if ev["type"] == "final":
            final = ev["message"]
        elif ev["type"] == "error":
            error = ev["message"]
    if final is None:
        raise RuntimeError(error or "no final message")
    return final


def score(case: dict[str, Any], message: dict[str, Any], session: Session) -> tuple[dict[str, bool], list[str]]:
    checks: dict[str, bool] = {}
    failures: list[str] = []
    answer = message["answer"]
    tables_text = " ".join(
        " ".join(str(v) for row in a.get("rows", []) for v in row) for a in message["artifacts"] if a["kind"] == "table"
    )
    ok_tools = {s["tool"] for s in message["steps"] if s.get("kind") == "tool" and s.get("status") == "ok"}

    if sql := case.get("expect_text_sql"):
        expected = str(scalar(session, sql))
        checks["text"] = expected.lower() in answer.lower()
        if not checks["text"]:
            failures.append(f"answer does not mention '{expected}'")
    if sql := case.get("expect_number_sql"):
        expected_n = float(scalar(session, sql))
        checks["number"] = contains_number(answer, expected_n) or contains_number(tables_text, expected_n)
        if not checks["number"]:
            failures.append(f"expected number ≈ {expected_n:,.2f} not found")
    for s in case.get("expect_contains", []):
        key = f"contains:{s}"
        checks[key] = s.lower() in answer.lower()
        if not checks[key]:
            failures.append(f"answer lacks '{s}'")
    for t in case.get("expect_tools", []):
        checks[f"tool:{t}"] = t in ok_tools
        if t not in ok_tools:
            failures.append(f"tool {t} not called successfully")
    kinds = {a["kind"] for a in message["artifacts"]}
    for k in case.get("expect_artifacts", []):
        checks[f"artifact:{k}"] = k in kinds
        if k not in kinds:
            failures.append(f"no {k} artifact")
    if wanted := case.get("expect_chart"):
        types = {a.get("chart_type") for a in message["artifacts"] if a["kind"] == "chart"}
        checks["chart"] = bool(types & set(wanted))
        if not checks["chart"]:
            failures.append(f"chart type {sorted(t for t in types if t)} not in {wanted}")
    if (n_rows := case.get("code_returns_rows")) is not None:
        m = re.search(r"```sql\n(.*?)```", answer, re.S)
        ran = False
        if m:
            try:
                ran = len(session.db.execute(m.group(1)).df) == n_rows
            except Exception as exc:  # report any failure as a check failure
                failures.append(f"generated SQL failed: {exc}")
        checks["code_runs"] = ran
        if not ran and m:
            failures.append(f"generated SQL did not return {n_rows} rows")
    return checks, failures


async def run(args: argparse.Namespace) -> int:
    overrides: dict[str, Any] = {}
    if args.provider:
        overrides["llm_provider"] = args.provider
    if args.model:
        overrides["llm_model"] = args.model
    settings = get_settings().model_copy(update=overrides) if overrides else get_settings()
    configure_logging("WARNING", "text")
    provider = OfflineProvider() if settings.resolved_provider() == "offline" else create_provider(settings)

    cases: list[dict[str, Any]] = yaml.safe_load((HERE / "cases.yaml").read_text())
    by_id = {c["id"]: c for c in cases}
    selected = [c for c in cases if not args.only or c["id"] in args.only.split(",")]

    results: list[CaseResult] = []
    print(f"Evaluating {len(selected)} case(s) with {provider.name}/{provider.model}\n")
    for case in selected:
        session = build_session(settings)  # fresh context per case

        def agent(session: Session = session) -> AnalystAgent:
            return AnalystAgent(session, provider, settings, fallback=None,
                                answer_cache=TTLCache("answers", 1, 1), tool_cache=TTLCache("q", 64, 600))

        start = time.perf_counter()
        try:
            if case.get("after"):
                await ask(agent(), by_id[case["after"]]["question"])
            message = await ask(agent(), case["question"])
            checks, failures = score(case, message, session)
        except Exception as exc:  # a crashed case is a failed case
            message = {"answer": "", "steps": [], "usage": {}, "artifacts": []}
            checks, failures = {"completed": False}, [f"{type(exc).__name__}: {exc}"]
        elapsed = time.perf_counter() - start
        steps = [s for s in message["steps"] if s.get("kind") == "tool"]
        r = CaseResult(
            id=case["id"],
            question=case["question"],
            passed=all(checks.values()) and bool(checks),
            checks=checks,
            failures=failures,
            elapsed_s=round(elapsed, 2),
            input_tokens=message.get("usage", {}).get("input_tokens", 0),
            output_tokens=message.get("usage", {}).get("output_tokens", 0),
            tool_calls=[s["tool"] for s in steps],
            tool_errors=sum(1 for s in steps if s.get("status") == "error"),
            answer=message["answer"],
        )
        results.append(r)
        mark = "PASS" if r.passed else "FAIL"
        print(f"  [{mark}] {r.id:<26} {r.elapsed_s:>6.2f}s  tools={','.join(r.tool_calls) or '-'}")
        for f in r.failures:
            print(f"         - {f}")

    passed = sum(r.passed for r in results)
    rate = passed / max(len(results), 1)
    print(f"\n{passed}/{len(results)} passed ({rate:.0%})")
    write_report(results, provider.name, provider.model, Path(args.out))
    return 0 if rate >= args.min_pass else 1


def write_report(results: list[CaseResult], provider: str, model: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "-", f"{provider}-{model}".lower()).strip("-")
    passed = sum(r.passed for r in results)
    total_in = sum(r.input_tokens for r in results)
    total_out = sum(r.output_tokens for r in results)
    lat = sorted(r.elapsed_s for r in results) or [0.0]
    lines = [
        f"# Eval report: {provider} / {model}",
        "",
        f"- Run: {datetime.now().isoformat(timespec='seconds')}",
        f"- Pass rate: **{passed}/{len(results)} ({passed / max(len(results), 1):.0%})**",
        f"- Latency: median {lat[len(lat) // 2]:.2f}s, max {lat[-1]:.2f}s",
        f"- Tokens: {total_in:,} in / {total_out:,} out",
        f"- Tool errors (self-corrected or not): {sum(r.tool_errors for r in results)}",
        "",
        "| Case | Result | Time | Tools | Notes |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        notes = "; ".join(r.failures) or "-"
        lines.append(f"| `{r.id}` | {'✅' if r.passed else '❌'} | {r.elapsed_s:.2f}s | {', '.join(r.tool_calls) or '-'} | {notes} |")
    (out_dir / f"{slug}-{stamp}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / f"{slug}-latest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / f"{slug}-{stamp}.json").write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
    print(f"Report written to {out_dir / f'{slug}-latest.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", help="Override LLM_PROVIDER (e.g. offline, anthropic, openai, gemini, groq)")
    parser.add_argument("--model", help="Override LLM_MODEL")
    parser.add_argument("--only", help="Comma-separated case ids")
    parser.add_argument("--min-pass", type=float, default=0.8, help="Minimum pass rate for exit code 0")
    parser.add_argument("--out", default=str(HERE / "results"), help="Report directory")
    sys.exit(asyncio.run(run(parser.parse_args())))


if __name__ == "__main__":
    main()
