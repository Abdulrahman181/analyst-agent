from __future__ import annotations

import itertools
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

NUMERIC_TEXT = re.compile(r"^-?(0|[1-9]\d*)(\.\d+)?$")
MAX_ASSIGNMENTS = 500


@dataclass(frozen=True)
class Match:
    strict: bool
    lenient: bool


def normalize_cell(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if NUMERIC_TEXT.match(text):
            return normalize_cell(float(text))
        return text
    if isinstance(value, int | float):
        number = float(value)
        if number == int(number) and abs(number) < 1e15:
            return int(number)
        return round(number, 2)
    return str(value)


def _normalize_rows(rows: Sequence[Sequence[Any]]) -> list[tuple[Any, ...]]:
    return [tuple(normalize_cell(cell) for cell in row) for row in rows]


def _key(row: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple((type(cell).__name__, str(cell)) for cell in row)


def _same(left: list[tuple[Any, ...]], right: list[tuple[Any, ...]], ordered: bool) -> bool:
    if len(left) != len(right):
        return False
    if ordered:
        return [_key(row) for row in left] == [_key(row) for row in right]
    return Counter(_key(row) for row in left) == Counter(_key(row) for row in right)


def _column(rows: list[tuple[Any, ...]], index: int) -> list[tuple[Any, ...]]:
    return [(row[index],) for row in rows]


def compare_results(
    golden_rows: Sequence[Sequence[Any]],
    candidate_rows: Sequence[Sequence[Any]],
    ordered: bool,
) -> Match:
    golden = _normalize_rows(golden_rows)
    candidate = _normalize_rows(candidate_rows)
    if len(golden) != len(candidate):
        return Match(False, False)
    if not golden:
        return Match(True, True)
    golden_width = len(golden[0])
    candidate_width = len(candidate[0]) if candidate else 0
    strict = golden_width == candidate_width and _same(golden, candidate, ordered)
    if strict:
        return Match(True, True)
    if candidate_width < golden_width:
        return Match(False, False)

    options: list[list[int]] = []
    for g_index in range(golden_width):
        target = _column(golden, g_index)
        matches = [
            c_index
            for c_index in range(candidate_width)
            if _same(target, _column(candidate, c_index), ordered)
        ]
        if not matches:
            return Match(False, False)
        options.append(matches)

    tried = 0
    for assignment in itertools.product(*options):
        if len(set(assignment)) != len(assignment):
            continue
        tried += 1
        if tried > MAX_ASSIGNMENTS:
            break
        projected = [tuple(row[i] for i in assignment) for row in candidate]
        if _same(golden, projected, ordered):
            return Match(False, True)
    return Match(False, False)
