"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api.deps import AppState
from app.api.routes import chat, datasets, health, reports, sessions
from app.config import Settings, get_settings
from app.core.errors import AppError
from app.core.logging import configure_logging, get_logger, request_id_var
from app.core.metrics import metrics

log = get_logger("app")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_format)
    state = AppState.build(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        log.info(
            "starting",
            extra={"version": __version__, "provider": state.provider.name, "model": state.provider.model},
        )

        async def sweeper() -> None:
            while True:
                await asyncio.sleep(60)
                state.sessions.evict_expired()

        task = asyncio.create_task(sweeper())
        yield
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description="Upload CSV files and analyse them in natural language: answers, charts, SQL, anomalies, forecasts.",
        lifespan=lifespan,
    )
    app.state.app_state = state

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        token = request_id_var.set(rid)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        elapsed = time.perf_counter() - start
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        if path.startswith("/api"):
            metrics.inc("http_requests", method=request.method, route=path, status=str(response.status_code))
            metrics.observe("http_request_seconds", elapsed, route=path)
            log.info(
                "request",
                extra={"method": request.method, "path": request.url.path, "status": response.status_code,
                       "ms": round(elapsed * 1000, 1), "rid": rid},
            )
        response.headers["x-request-id"] = rid
        return response

    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"error": exc.to_dict()})

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        field = ".".join(str(p) for p in first.get("loc", []) if p != "body")
        message = f"{field}: {first.get('msg', 'invalid input')}" if field else first.get("msg", "Invalid request.")
        return JSONResponse(status_code=422, content={"error": {"code": "validation_error", "message": message}})

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception) -> JSONResponse:  # pragma: no cover
        log.exception("unhandled error")
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": "Something went wrong on the server. It has been logged."}},
        )

    for router in (health.router, sessions.router, datasets.router, chat.router, reports.router):
        app.include_router(router)

    # Serve the built React app (single-container deployment).
    dist = settings.frontend_dist
    if dist.exists() and (dist / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        async def spa(full_path: str) -> FileResponse:
            candidate = dist / full_path
            if full_path and candidate.is_file() and dist in candidate.resolve().parents:
                return FileResponse(candidate)
            return FileResponse(dist / "index.html")

    return app


app = create_app()
