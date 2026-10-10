"""Export the session's analysis."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends
from fastapi.responses import Response

from app.api.deps import get_session, require_auth
from app.reports.exporter import render_html, render_markdown
from app.sessions.store import Session

router = APIRouter(prefix="/api/sessions/{session_id}", tags=["reports"], dependencies=[Depends(require_auth)])


@router.get("/report")
def export_report(format: Literal["html", "md"] = "html", session: Session = Depends(get_session)) -> Response:
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    if format == "md":
        return Response(
            render_markdown(session),
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="analysis-report-{stamp}.md"'},
        )
    return Response(
        render_html(session),
        media_type="text/html; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="analysis-report-{stamp}.html"'},
    )
