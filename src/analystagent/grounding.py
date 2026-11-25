from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

from analystagent.text import normalize_digits

_DATE = re.compile(r"\d{4}-\d{2}(?:-\d{2})?")
_LIST_MARKER = re.compile(r"(?m)^\s*\d{1,3}[.)]\s")
_NUMBER = re.compile(r"(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


DEFAULT_SHOWN_ROWS = 30


@dataclass(frozen=True)
class ExecutedResult:
    sql: str
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    truncated: bool = False
    shown_limit: int = DEFAULT_SHOWN_ROWS

    @property
    def shown_rows(self) -> tuple[tuple[Any, ...], ...]:
        return self.rows[: self.shown_limit]

    @property
    def total_rows(self) -> int:
        return len(self.rows)


@dataclass(frozen=True)
class GroundingReport:
    grounded: bool
    checked: int
    ungrounded: tuple[str, ...] = field(default_factory=tuple)

    def feedback(self) -> str:
        listed = ", ".join(self.ungrounded)
        return (
            f"These values in the answer do not appear in any query result you ran: {listed}. "
            "Compute every figure in SQL with run_sql and quote it exactly, "
            "or remove it from the answer."
        )


def _to_decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool):
        return Decimal(int(value))
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        try:
            return Decimal(repr(value))
        except InvalidOperation:
            return None
    return None


def _decimals(token: str) -> int:
    return len(token.split(".", 1)[1]) if "." in token else 0


def _matches(answer_value: Decimal, decimals: int, source: Decimal) -> bool:
    if answer_value == source:
        return True
    quantum = Decimal(1).scaleb(-decimals)
    for mode in (ROUND_HALF_UP, ROUND_HALF_EVEN):
        if source.quantize(quantum, rounding=mode) == answer_value:
            return True
    return False


def _source_values(results: Sequence[ExecutedResult]) -> list[Decimal]:
    values: list[Decimal] = []
    for result in results:
        values.append(Decimal(result.total_rows))
        values.append(Decimal(len(result.shown_rows)))
        for row in result.shown_rows:
            for cell in row:
                decimal = _to_decimal(cell)
                if decimal is not None:
                    values.append(decimal)
    return values


def _string_cells(results: Sequence[ExecutedResult]) -> list[str]:
    cells: list[str] = []
    for result in results:
        for row in result.shown_rows:
            for cell in row:
                if isinstance(cell, str) and cell.strip():
                    cells.append(cell.strip())
    return cells


def _strip_quoted_cells(text: str, cells: Iterable[str]) -> str:
    stripped = text
    for cell in sorted({c for c in cells if any(ch.isdigit() for ch in c)}, key=len, reverse=True):
        stripped = re.sub(re.escape(normalize_digits(cell)), " ", stripped, flags=re.IGNORECASE)
    return stripped


def check_grounding(
    answer: str, question: str, results: Sequence[ExecutedResult]
) -> GroundingReport:
    text = normalize_digits(answer)
    cells = _string_cells(results)
    labels = [name for result in results for name in result.columns]
    text = _strip_quoted_cells(text, [*cells, *labels])
    question_text = normalize_digits(question)
    corpus = " ".join(normalize_digits(cell) for cell in cells)

    ungrounded: list[str] = []
    checked = 0

    for token in _DATE.findall(text):
        checked += 1
        if token not in corpus and token not in question_text:
            ungrounded.append(token)
    text = _DATE.sub(" ", text)
    text = _LIST_MARKER.sub(" ", text)

    sources = _source_values(results)
    question_numbers = {_to_decimal_text(m) for m in _NUMBER.findall(_DATE.sub(" ", question_text))}
    for token in _NUMBER.findall(text):
        checked += 1
        clean = token.replace(",", "")
        try:
            value = Decimal(clean)
        except InvalidOperation:
            ungrounded.append(token)
            continue
        if _to_decimal_text(token) in question_numbers:
            continue
        decimals = _decimals(clean)
        if any(_matches(value, decimals, source) for source in sources):
            continue
        ungrounded.append(token)
    unique = tuple(dict.fromkeys(ungrounded))
    return GroundingReport(not unique, checked, unique)


def _to_decimal_text(token: str) -> Decimal | None:
    try:
        return Decimal(token.replace(",", ""))
    except InvalidOperation:
        return None
