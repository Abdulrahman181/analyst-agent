from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _first_existing(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def default_db_path() -> Path:
    override = os.environ.get("ANALYST_DB")
    if override:
        return Path(override)
    return _first_existing(
        [Path.cwd() / "data" / "chinook.sqlite", repo_root() / "data" / "chinook.sqlite"]
    )


def default_questions_path() -> Path:
    return _first_existing(
        [
            Path.cwd() / "benchmarks" / "questions.json",
            repo_root() / "benchmarks" / "questions.json",
        ]
    )
