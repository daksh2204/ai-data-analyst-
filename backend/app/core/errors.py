"""Domain errors and their HTTP mapping.

Every error raised on purpose inherits from :class:`AppError`, carries a stable machine
readable ``code`` and a message written for the person using the app.
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    status_code: int = 400
    code: str = "app_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            body["details"] = self.details
        return body


class ValidationError(AppError):
    status_code = 422
    code = "validation_error"


class FileTooLargeError(AppError):
    status_code = 413
    code = "file_too_large"


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class AuthError(AppError):
    status_code = 401
    code = "unauthorized"


class LimitExceededError(AppError):
    status_code = 429
    code = "limit_exceeded"


class QueryError(AppError):
    """A SQL / pandas query failed. Shown to the LLM so it can correct itself."""

    status_code = 400
    code = "query_error"


class UnsafeCodeError(QueryError):
    code = "unsafe_code"


class QueryTimeoutError(QueryError):
    code = "query_timeout"


class ToolError(AppError):
    code = "tool_error"


class LLMError(AppError):
    status_code = 502
    code = "llm_error"
