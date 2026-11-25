from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    finished_at TEXT,
    question TEXT NOT NULL,
    language TEXT,
    mode TEXT,
    provider TEXT,
    model TEXT,
    status TEXT,
    stop_reason TEXT,
    answer TEXT,
    cited_sql TEXT,
    grounded INTEGER,
    confidence TEXT,
    steps INTEGER,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    security_blocks INTEGER,
    query_errors INTEGER,
    grounding_rejections INTEGER,
    wall_s REAL,
    meta_json TEXT
);
CREATE TABLE IF NOT EXISTS steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    idx INTEGER NOT NULL,
    kind TEXT NOT NULL,
    tool TEXT,
    args_json TEXT,
    result_preview TEXT,
    ok INTEGER,
    security_block INTEGER,
    latency_s REAL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER
);
CREATE INDEX IF NOT EXISTS idx_steps_run ON steps(run_id, idx);
"""

PREVIEW_CHARS = 500
ARGS_CHARS = 2000
RUN_FIELDS = frozenset(
    {
        "status",
        "stop_reason",
        "answer",
        "cited_sql",
        "grounded",
        "confidence",
        "steps",
        "prompt_tokens",
        "completion_tokens",
        "security_blocks",
        "query_errors",
        "grounding_rejections",
        "wall_s",
    }
)


def _clip(text: str | None, limit: int) -> str | None:
    if text is None:
        return None
    return text if len(text) <= limit else text[:limit] + "...[clipped]"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class RunStore:
    def __init__(self, path: str | Path = ":memory:") -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)
        self._counters: dict[int, int] = {}

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> RunStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def create_run(
        self, question: str, language: str, mode: str, meta: dict[str, Any] | None = None
    ) -> int:
        info = meta or {}
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO runs (created_at, question, language, mode, provider, model, "
                "meta_json) VALUES (?,?,?,?,?,?,?)",
                (
                    _now(),
                    question,
                    language,
                    mode,
                    info.get("provider"),
                    info.get("model"),
                    json.dumps(info, sort_keys=True, default=str),
                ),
            )
            self._conn.commit()
            run_id = int(cursor.lastrowid or 0)
            self._counters[run_id] = 0
        return run_id

    def add_step(
        self,
        run_id: int,
        kind: str,
        tool: str | None = None,
        args: dict[str, Any] | None = None,
        result: str | None = None,
        ok: bool | None = None,
        security_block: bool = False,
        latency_s: float | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        args_json = (
            _clip(json.dumps(args, sort_keys=True, default=str, ensure_ascii=False), ARGS_CHARS)
            if args
            else None
        )
        with self._lock:
            idx = self._counters.get(run_id, 0)
            self._counters[run_id] = idx + 1
            self._conn.execute(
                "INSERT INTO steps (run_id, idx, kind, tool, args_json, result_preview, ok, "
                "security_block, latency_s, prompt_tokens, completion_tokens) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    idx,
                    kind,
                    tool,
                    args_json,
                    _clip(result, PREVIEW_CHARS),
                    None if ok is None else int(ok),
                    int(security_block),
                    latency_s,
                    prompt_tokens,
                    completion_tokens,
                ),
            )
            self._conn.commit()

    def finish_run(self, run_id: int, **fields: Any) -> None:
        unknown = set(fields) - RUN_FIELDS
        if unknown:
            raise ValueError(f"unknown run fields: {sorted(unknown)}")
        values = {
            key: (int(value) if isinstance(value, bool) else value) for key, value in fields.items()
        }
        assignments = ", ".join(f"{key} = ?" for key in values)
        with self._lock:
            self._conn.execute(
                f"UPDATE runs SET finished_at = ?, {assignments} WHERE id = ?",  # noqa: S608
                (_now(), *values.values(), run_id),
            )
            self._conn.commit()

    def get_run(self, run_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def steps_for(self, run_id: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM steps WHERE run_id = ? ORDER BY idx", (run_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]
