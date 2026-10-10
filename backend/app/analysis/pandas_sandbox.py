"""Restricted execution of LLM-written pandas code.

Defence in depth, in this order:

1. **AST allow-list** — imports, ``global``/``nonlocal``, ``while`` loops, class definitions,
   decorators, dunder/private attribute access, string-evaluation APIs (``eval``, ``query``,
   ``str.format``) and every file/network I/O method on pandas/numpy are rejected before
   anything runs.
2. **Minimal namespace** — only ``pd``, ``np``, the session's DataFrames and a short list of
   pure builtins. No ``open``, ``__import__``, ``getattr``, ``type`` ...
3. **Resource limits** — a line-level deadline enforced with ``sys.settrace`` (stops Python
   loops) plus a wall-clock timeout on the worker thread, a capped ``range`` and copy-on-write
   DataFrames so session data cannot be mutated.

This is a pragmatic guard for a single-tenant analysis tool. A multi-tenant deployment
should run this tool in an isolated sandbox (gVisor/Firecracker/a separate container).
"""

from __future__ import annotations

import ast
import concurrent.futures
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.core.errors import QueryError, QueryTimeoutError, UnsafeCodeError

pd.set_option("mode.copy_on_write", True)

_FORBIDDEN_NODES: tuple[type[ast.AST], ...] = (
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
    ast.While,
    ast.ClassDef,
    ast.AsyncFunctionDef,
    ast.AsyncFor,
    ast.AsyncWith,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
    ast.With,
    ast.Try,
    ast.Raise,
    ast.Delete,
)

_FORBIDDEN_NAMES = {
    "eval", "exec", "compile", "open", "__import__", "globals", "locals", "vars", "getattr",
    "setattr", "delattr", "input", "breakpoint", "exit", "quit", "help", "dir", "type",
    "object", "memoryview", "super", "classmethod", "staticmethod", "property", "id", "hash",
}

_FORBIDDEN_ATTRS = {
    # string evaluation
    "eval", "query", "format", "format_map",
    # I/O
    "to_csv", "to_pickle", "to_parquet", "to_excel", "to_sql", "to_hdf", "to_feather", "to_stata",
    "to_clipboard", "to_latex", "to_html", "to_xml", "to_orc", "to_gbq", "to_json", "tofile", "save",
    "savez", "savez_compressed", "savetxt", "load", "loadtxt", "fromfile", "genfromtxt", "memmap",
    # library internals that expose file handles or the interpreter
    "io", "lib", "core", "util", "testing", "compat", "os", "sys", "ctypes", "ctypeslib", "f2py",
    "distutils", "plotting", "options", "set_option",
    # escape hatches
    "system", "popen", "spawn", "fork", "kill", "remove", "rmdir", "unlink", "subprocess",
    "builtins", "modules", "f_globals", "f_locals", "f_back", "f_builtins", "f_code", "gi_frame",
    "cr_frame", "ag_frame", "tb_frame", "co_code",
}

_SAFE_BUILTINS: dict[str, Any] = {
    name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)  # type: ignore[index]
    for name in (
        "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float", "frozenset",
        "int", "isinstance", "len", "list", "map", "max", "min", "next", "pow", "reversed", "round",
        "set", "slice", "sorted", "str", "sum", "tuple", "zip", "ValueError", "KeyError",
        "TypeError", "ZeroDivisionError", "Exception",
    )
}
_SAFE_BUILTINS.update({"True": True, "False": False, "None": None})

MAX_RANGE = 1_000_000


def _capped_range(*args: int) -> range:
    r = range(*args)
    if len(r) > MAX_RANGE:
        raise UnsafeCodeError(f"range() is limited to {MAX_RANGE:,} items in the sandbox.")
    return r


class _Validator(ast.NodeVisitor):
    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, _FORBIDDEN_NODES):
            raise UnsafeCodeError(
                f"'{type(node).__name__}' statements aren't allowed in the pandas sandbox. "
                "Use vectorised pandas operations; pd and np are already available."
            )
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in _FORBIDDEN_NAMES or node.id.startswith("__"):
            raise UnsafeCodeError(f"Use of '{node.id}' isn't allowed in the pandas sandbox.")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_"):
            raise UnsafeCodeError(f"Access to private attribute '{node.attr}' isn't allowed.")
        if node.attr in _FORBIDDEN_ATTRS or node.attr.startswith("read_"):
            hint = " Filter with boolean masks instead (df[df['col'] > 5])." if node.attr in ("query", "eval") else ""
            raise UnsafeCodeError(f"'.{node.attr}' isn't allowed in the pandas sandbox.{hint}")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.decorator_list:
            raise UnsafeCodeError("Decorators aren't allowed in the pandas sandbox.")
        self.generic_visit(node)


def validate_code(code: str) -> ast.Module:
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise QueryError(f"Python syntax error on line {exc.lineno}: {exc.msg}") from exc
    _Validator().visit(tree)
    # Notebook semantics: a trailing bare expression becomes `result`.
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = tree.body[-1]
        tree.body[-1] = ast.Assign(targets=[ast.Name(id="result", ctx=ast.Store())], value=last.value)
        ast.fix_missing_locations(tree)
    return tree


@dataclass
class SandboxResult:
    value: Any
    stdout: str
    elapsed_ms: float


def run_pandas(code: str, frames: dict[str, pd.DataFrame], *, timeout: float = 10.0) -> SandboxResult:
    """Validate and execute ``code``. The code must assign ``result`` (or end in an expression)."""
    tree = validate_code(code)
    compiled = compile(tree, "<analysis>", "exec")
    printed: list[str] = []

    def _print(*args: Any, **_: Any) -> None:
        if sum(len(p) for p in printed) < 10_000:
            printed.append(" ".join(str(a) for a in args))

    namespace: dict[str, Any] = {
        "__builtins__": {**_SAFE_BUILTINS, "range": _capped_range, "print": _print},
        "pd": pd,
        "np": np,
        "dfs": {k: v.copy(deep=False) for k, v in frames.items()},
    }
    for name, frame in frames.items():
        namespace[name] = frame.copy(deep=False)
    if len(frames) >= 1:
        namespace.setdefault("df", next(iter(frames.values())).copy(deep=False))

    deadline = time.monotonic() + timeout
    cancelled = threading.Event()

    def _tracer(frame: Any, event: str, arg: Any) -> Any:
        if cancelled.is_set() or time.monotonic() > deadline:
            raise QueryTimeoutError(f"The pandas code ran longer than {timeout:.0f}s and was stopped.")
        # Line-trace only the user's code; library frames get a cheap call-level check.
        return _tracer if frame.f_code.co_filename == "<analysis>" else None

    def _target() -> Any:
        sys.settrace(_tracer)
        try:
            exec(compiled, namespace)
        finally:
            sys.settrace(None)
        if "result" not in namespace:
            raise QueryError("The code must assign its output to a variable named `result`.")
        return namespace["result"]

    start = time.perf_counter()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="pandas-sandbox")
    future = pool.submit(_target)
    try:
        value = future.result(timeout=timeout + 1)
    except concurrent.futures.TimeoutError as exc:
        cancelled.set()
        raise QueryTimeoutError(f"The pandas code ran longer than {timeout:.0f}s and was stopped.") from exc
    except (QueryError, UnsafeCodeError):
        raise
    except Exception as exc:  # user-code errors are reported back to the LLM to self-correct
        raise QueryError(f"{type(exc).__name__}: {exc}"[:800]) from exc
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return SandboxResult(value=value, stdout="\n".join(printed), elapsed_ms=round((time.perf_counter() - start) * 1000, 1))


def result_to_frame(value: Any) -> pd.DataFrame | None:
    """Normalise a sandbox result into a DataFrame when it is tabular."""
    if isinstance(value, pd.DataFrame):
        df = value
        meaningful_index = (
            isinstance(df.index, pd.MultiIndex)
            or any(n is not None for n in df.index.names)
            or not pd.api.types.is_integer_dtype(df.index)
        )
        if meaningful_index:
            df = df.reset_index()
        elif not isinstance(df.index, pd.RangeIndex):
            df = df.reset_index(drop=True)  # a shuffled positional index after sort/filter
        df.columns = [" / ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in df.columns]
        return df
    if isinstance(value, pd.Series):
        name = value.name if value.name is not None else "value"
        return value.rename(str(name)).reset_index()
    if isinstance(value, (list, tuple)) and value and all(isinstance(v, dict) for v in value):
        return pd.DataFrame(list(value))
    if isinstance(value, dict) and value and all(np.isscalar(v) for v in value.values()):
        return pd.DataFrame({"key": list(value.keys()), "value": list(value.values())})
    return None
