"""Health, configuration and metrics endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse

from app import __version__
from app.api.deps import AppState, get_state, require_auth
from app.core.metrics import metrics

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health")
def health(state: AppState = Depends(get_state)) -> dict[str, Any]:
    return {
        "status": "ok",
        "version": __version__,
        "provider": state.provider.name,
        "model": state.provider.model,
        "sessions": len(state.sessions),
    }


@router.get("/config")
def config(state: AppState = Depends(get_state)) -> dict[str, Any]:
    s = state.settings
    return {
        "app_name": s.app_name,
        "provider": state.provider.name,
        "model": state.provider.model,
        "offline": state.provider.name == "offline",
        "auth_required": bool(s.app_access_token),
        "limits": {
            "max_upload_mb": s.max_upload_mb,
            "max_files_per_session": s.max_files_per_session,
            "max_rows_per_file": s.max_rows_per_file,
        },
        "samples_available": s.sample_data_dir.exists(),
    }


@router.get("/metrics", dependencies=[Depends(require_auth)], response_model=None)
def get_metrics(format: str = "json") -> Any:
    if format == "prometheus":
        return PlainTextResponse(metrics.prometheus(), media_type="text/plain; version=0.0.4")
    return metrics.snapshot()
