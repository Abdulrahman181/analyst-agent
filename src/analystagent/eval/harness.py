from __future__ import annotations

import platform
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from analystagent import __version__
from analystagent.agent.loop import AnswerResult
from analystagent.database import Database
from analystagent.db.connection import QueryError, execute_query
from analystagent.eval.dataset import EvalItem
from analystagent.eval.metrics import aggregate
from analystagent.eval.scoring import compare_results

GOLDEN_MAX_ROWS = 1000
GOLDEN_TIMEOUT_S = 10.0


class Analyst(Protocol):
    def ask(self, question: str, language: str | None = None) -> AnswerResult: ...


AnalystFactory = Callable[[EvalItem, int], Analyst]
ResultCallback = Callable[["ItemResult"], None]


@dataclass
class ItemResult:
    item_id: str
    intent_id: str
    language: str
    category: str
    difficulty: str
    expect: str
    repeat: int
    mode: str
    answer_source: str
    status: str
    stop_reason: str
    correct: bool
    correct_strict: bool
    valid_sql: bool
    declined: bool
    grounded: bool | None
    confidence: str
    security_blocks: int
    query_errors: int
    grounding_rejections: int
    steps: int
    prompt_tokens: int
    completion_tokens: int
    wall_s: float
    run_id: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def golden_rows(db: Database, sql: str) -> tuple[tuple[Any, ...], ...]:
    conn = db.connect()
    try:
        return execute_query(conn, sql, GOLDEN_MAX_ROWS, GOLDEN_TIMEOUT_S).rows
    except QueryError as exc:
        raise RuntimeError(f"golden SQL failed: {exc}") from exc
    finally:
        conn.close()


def score_item(
    item: EvalItem, outcome: AnswerResult, expected: Sequence[Any] | None
) -> tuple[bool, bool, bool]:
    if item.expect == "decline":
        declined = outcome.status == "answered" and not outcome.sql
        return declined, declined, False
    if outcome.result is None or expected is None:
        return False, False, False
    match = compare_results(expected, outcome.result.rows, item.ordered)
    return match.lenient, match.strict, True


def build_record(
    item: EvalItem, repeat: int, outcome: AnswerResult, expected: Sequence[Any] | None
) -> ItemResult:
    correct, strict, valid = score_item(item, outcome, expected)
    return ItemResult(
        item_id=item.id,
        intent_id=item.intent_id,
        language=item.language,
        category=item.category,
        difficulty=item.difficulty,
        expect=item.expect,
        repeat=repeat,
        mode=outcome.mode,
        answer_source=outcome.answer_source,
        status=outcome.status,
        stop_reason=outcome.stop_reason,
        correct=correct,
        correct_strict=strict,
        valid_sql=valid,
        declined=outcome.declined,
        grounded=outcome.grounded,
        confidence=outcome.confidence,
        security_blocks=outcome.security_blocks,
        query_errors=outcome.query_errors,
        grounding_rejections=outcome.grounding_rejections,
        steps=outcome.steps,
        prompt_tokens=outcome.prompt_tokens,
        completion_tokens=outcome.completion_tokens,
        wall_s=round(outcome.wall_s, 3),
        run_id=outcome.run_id,
    )


def run_eval(
    items: Sequence[EvalItem],
    db: Database,
    factory: AnalystFactory,
    repeats: int = 1,
    on_result: ResultCallback | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = datetime.now(UTC).isoformat(timespec="seconds")
    expected_cache: dict[str, tuple[tuple[Any, ...], ...]] = {}
    records: list[ItemResult] = []
    for item in items:
        expected = None
        if item.expect == "answer" and item.golden_sql:
            if item.intent_id not in expected_cache:
                expected_cache[item.intent_id] = golden_rows(db, item.golden_sql)
            expected = expected_cache[item.intent_id]
        for repeat in range(repeats):
            outcome = factory(item, repeat).ask(item.question, item.language)
            record = build_record(item, repeat, outcome, expected)
            records.append(record)
            if on_result is not None:
                on_result(record)
    rows = [record.to_dict() for record in records]
    return {
        "schema_version": 1,
        "meta": {
            **(meta or {}),
            "started_at": started,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "analystagent": __version__,
            "n_items": len(items),
            "repeats": repeats,
        },
        "aggregate": aggregate(rows),
        "results": rows,
    }
