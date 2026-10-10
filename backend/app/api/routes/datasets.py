"""Upload, inspect, clean and query datasets."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, File, UploadFile

from app.agent.tools import frame_payload
from app.analysis.dashboard import build_dashboard
from app.api.deps import AppState, get_session, get_state, require_auth
from app.api.schemas import FixRequest, QueryRequest, dataset_summary, session_state
from app.core.errors import AppError, FileTooLargeError, ValidationError
from app.core.logging import get_logger
from app.core.metrics import metrics
from app.data.ingestion import load_csv
from app.data.profiling import apply_fix
from app.sessions.store import Session

log = get_logger(__name__)
router = APIRouter(prefix="/api/sessions/{session_id}", tags=["datasets"], dependencies=[Depends(require_auth)])

SAMPLE_FILES = ("sales_orders.csv", "customers.csv", "products.csv", "regional_targets.csv")


async def _read_limited(upload: UploadFile, limit: int) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while chunk := await upload.read(1024 * 1024):
        size += len(chunk)
        if size > limit:
            raise FileTooLargeError(
                f"'{upload.filename}' is larger than the {limit // 1_048_576} MB limit.",
                details={"filename": upload.filename},
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _ingest(session: Session, raw: bytes, filename: str, state: AppState) -> dict[str, Any]:
    s = state.settings
    loaded = load_csv(
        raw,
        filename,
        existing_tables=set(session.tables),
        max_bytes=s.max_upload_bytes,
        max_rows=s.max_rows_per_file,
        max_columns=s.max_columns_per_file,
    )
    entry = session.add_table(loaded)
    metrics.inc("files_uploaded", status="ok")
    return dataset_summary(entry)


@router.post("/datasets")
async def upload_datasets(
    files: list[UploadFile] = File(..., description="One or more CSV files."),
    session: Session = Depends(get_session),
    state: AppState = Depends(get_state),
) -> dict[str, Any]:
    """Upload CSVs. Each file is validated independently, so one bad file doesn't block the rest."""
    if not files:
        raise ValidationError("Choose at least one CSV file to upload.")
    state.sessions.check_file_limit(session, len(files), state.settings.max_files_per_session)
    results: list[dict[str, Any]] = []
    for upload in files:
        name = upload.filename or "upload.csv"
        try:
            raw = await _read_limited(upload, state.settings.max_upload_bytes)
            summary = await asyncio.to_thread(_ingest, session, raw, name, state)
            results.append({"filename": name, "ok": True, "dataset": summary})
        except AppError as exc:
            metrics.inc("files_uploaded", status="rejected")
            log.info("upload rejected", extra={"upload": name, "reason": exc.code})
            results.append({"filename": name, "ok": False, "error": exc.to_dict()})
    return {"results": results, "session": session_state(session, state.provider.name, state.provider.model)}


@router.post("/datasets/samples")
async def load_samples(session: Session = Depends(get_session), state: AppState = Depends(get_state)) -> dict[str, Any]:
    """Load the bundled sample datasets (sales, customers, products, regional targets)."""
    folder = state.settings.sample_data_dir
    results = []
    for fname in SAMPLE_FILES:
        path = folder / fname
        if not path.exists() or path.stem in session.tables:
            continue
        summary = await asyncio.to_thread(_ingest, session, path.read_bytes(), fname, state)
        results.append({"filename": fname, "ok": True, "dataset": summary})
    if not results and not session.tables:
        raise ValidationError("Sample data isn't available in this deployment.")
    return {"results": results, "session": session_state(session, state.provider.name, state.provider.model)}


@router.get("/datasets/{table}")
def get_dataset(table: str, session: Session = Depends(get_session)) -> dict[str, Any]:
    entry = session.get_table(table)
    preview = session.db.execute(f'SELECT * FROM "{entry.name}" LIMIT 100')
    return {
        "dataset": dataset_summary(entry),
        "profile": entry.profile.to_dict(),
        "preview": frame_payload(preview.df, 100),
    }


@router.delete("/datasets/{table}")
def delete_dataset(table: str, session: Session = Depends(get_session), state: AppState = Depends(get_state)) -> dict[str, Any]:
    session.remove_table(table)
    return session_state(session, state.provider.name, state.provider.model)


@router.post("/datasets/{table}/fix")
def fix_dataset(
    table: str, body: FixRequest, session: Session = Depends(get_session), state: AppState = Depends(get_state)
) -> dict[str, Any]:
    entry = session.get_table(table)
    try:
        df, message = apply_fix(entry.df, body.fix, body.column)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    entry = session.replace_table_df(table, df)
    state.tool_cache.clear()
    return {
        "message": message,
        "dataset": dataset_summary(entry),
        "profile": entry.profile.to_dict(),
        "session": session_state(session, state.provider.name, state.provider.model),
    }


@router.get("/dashboard")
def dashboard(table: str | None = None, session: Session = Depends(get_session)) -> dict[str, Any]:
    entry = session.get_table(table) if table else session.primary_table
    if entry is None:
        raise ValidationError("Upload a dataset to build a dashboard.")
    return build_dashboard(session.db, entry.profile)


@router.post("/query")
def run_query(body: QueryRequest, session: Session = Depends(get_session)) -> dict[str, Any]:
    """Run a read-only SQL query directly (used by the 'Run' button on SQL snippets)."""
    res = session.db.execute(body.sql, max_rows=1000)
    return {"sql": res.sql, "elapsed_ms": res.elapsed_ms, **frame_payload(res.df, 1000)}
