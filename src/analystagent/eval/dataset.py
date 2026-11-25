from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

LANGUAGES = ("en", "ar")


class DatasetError(Exception):
    pass


@dataclass(frozen=True)
class EvalItem:
    id: str
    intent_id: str
    language: str
    question: str
    category: str
    difficulty: str
    expect: str
    golden_sql: str | None
    ordered: bool


def load_items(
    path: str | Path,
    languages: Sequence[str] = LANGUAGES,
    only: Sequence[str] | None = None,
) -> list[EvalItem]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DatasetError(f"could not read questions file: {exc}") from exc
    items: list[EvalItem] = []
    wanted = set(only) if only else None
    seen: set[str] = set()
    for intent in raw.get("intents", []):
        intent_id = str(intent["id"])
        if intent_id in seen:
            raise DatasetError(f"duplicate intent id: {intent_id}")
        seen.add(intent_id)
        if wanted is not None and intent_id not in wanted:
            continue
        for language in languages:
            question = (intent.get("questions") or {}).get(language)
            if not question:
                raise DatasetError(f"intent {intent_id} has no {language} question")
            items.append(
                EvalItem(
                    id=f"{intent_id}.{language}",
                    intent_id=intent_id,
                    language=language,
                    question=str(question),
                    category=str(intent.get("category", "general")),
                    difficulty=str(intent.get("difficulty", "medium")),
                    expect=str(intent.get("expect", "answer")),
                    golden_sql=intent.get("golden_sql"),
                    ordered=bool(intent.get("ordered", False)),
                )
            )
    if wanted:
        missing = wanted - seen
        if missing:
            raise DatasetError(f"unknown intent ids: {', '.join(sorted(missing))}")
    return items
