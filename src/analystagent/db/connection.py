from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DENIED_FUNCTIONS = frozenset(
    {
        "load_extension",
        "readfile",
        "writefile",
        "edit",
        "zeroblob",
        "randomblob",
        "random",
        "fts3_tokenizer",
    }
)

MAX_VALUE_LENGTH = 1_000_000
MAX_SQL_LENGTH = 5_000
MAX_EXPR_DEPTH = 60
PROGRESS_INTERVAL = 2_000


class QueryError(Exception):
    pass


class QueryTimeout(QueryError):
    pass


@dataclass(frozen=True)
class QueryResult:
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    truncated: bool
    elapsed_s: float

    @property
    def row_count(self) -> int:
        return len(self.rows)


def _json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return f"<blob {len(value)} bytes>"
    return value


def _make_authorizer(denied_tables: frozenset[str]):
    denied = {name.lower() for name in denied_tables}

    def authorize(action: int, arg1: str | None, arg2: str | None, *_rest: str | None) -> int:
        if action == sqlite3.SQLITE_SELECT or action == sqlite3.SQLITE_RECURSIVE:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ:
            table = (arg1 or "").lower()
            if table.startswith("sqlite_") or table in denied:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION:
            name = (arg2 or "").lower()
            return sqlite3.SQLITE_DENY if name in DENIED_FUNCTIONS else sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    return authorize


def connect_readonly(path: str | Path, allowed_tables: frozenset[str]) -> sqlite3.Connection:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"database not found: {resolved}")
    uri = resolved.as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    existing = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")
    }
    denied = frozenset(existing) - allowed_tables
    conn.execute("PRAGMA query_only = ON")
    conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, MAX_VALUE_LENGTH)
    conn.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, MAX_SQL_LENGTH * 2)
    conn.setlimit(sqlite3.SQLITE_LIMIT_EXPR_DEPTH, MAX_EXPR_DEPTH)
    conn.set_authorizer(_make_authorizer(denied))
    return conn


def execute_query(
    conn: sqlite3.Connection, sql: str, max_rows: int, timeout_s: float
) -> QueryResult:
    deadline = time.monotonic() + timeout_s
    timed_out = False

    def watchdog() -> int:
        nonlocal timed_out
        if time.monotonic() > deadline:
            timed_out = True
            return 1
        return 0

    conn.set_progress_handler(watchdog, PROGRESS_INTERVAL)
    started = time.perf_counter()
    cursor = None
    try:
        cursor = conn.execute(sql)
        columns = tuple(description[0] for description in (cursor.description or ()))
        fetched = cursor.fetchmany(max_rows + 1)
    except sqlite3.Error as exc:
        if timed_out:
            raise QueryTimeout(f"query exceeded the {timeout_s:g}s time limit") from exc
        raise QueryError(str(exc)) from exc
    finally:
        conn.set_progress_handler(None, 0)
        if cursor is not None:
            cursor.close()
    truncated = len(fetched) > max_rows
    rows = tuple(tuple(_json_safe(cell) for cell in row) for row in fetched[:max_rows])
    return QueryResult(columns, rows, truncated, time.perf_counter() - started)
