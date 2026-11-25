from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import yaml

from analystagent.text import normalize_arabic

MIN_TERM_LENGTH = 2
_STRIP_PREFIX = re.compile("^(ال|وال|بال|لل)")


class GlossaryError(Exception):
    pass


@dataclass(frozen=True)
class GlossaryEntry:
    id: str
    kind: str
    terms_en: tuple[str, ...]
    terms_ar: tuple[str, ...]
    meaning: str


@dataclass(frozen=True)
class Glossary:
    notes: tuple[str, ...]
    entries: tuple[GlossaryEntry, ...]

    def match(self, question: str) -> list[GlossaryEntry]:
        lowered = question.lower()
        arabic = normalize_arabic(question)
        arabic_tokens = [_STRIP_PREFIX.sub("", token) for token in re.findall(r"[\w]+", arabic)]
        arabic_text = " ".join(arabic_tokens)
        hits: list[GlossaryEntry] = []
        for entry in self.entries:
            if any(_match_english(term, lowered) for term in entry.terms_en) or any(
                _match_arabic(term, arabic_text) for term in entry.terms_ar
            ):
                hits.append(entry)
        return hits

    def render_hints(self, question: str) -> str:
        matched = self.match(question)
        lines = [f"- {entry.meaning}" for entry in matched]
        lines += [f"- {note}" for note in self.notes]
        return "\n".join(lines)


def _match_english(term: str, lowered: str) -> bool:
    if len(term) < MIN_TERM_LENGTH:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])", lowered) is not None


def _match_arabic(term: str, arabic_text: str) -> bool:
    words = [_STRIP_PREFIX.sub("", word) for word in normalize_arabic(term).split()]
    needle = " ".join(words)
    if len(needle) < MIN_TERM_LENGTH:
        return False
    return re.search(rf"(?<![\w]){re.escape(needle)}(?![\w])", arabic_text) is not None


def _parse(raw: object) -> Glossary:
    if not isinstance(raw, dict):
        raise GlossaryError("glossary must be a mapping")
    entries: list[GlossaryEntry] = []
    seen: set[str] = set()
    for item in raw.get("entries") or []:
        if not isinstance(item, dict):
            raise GlossaryError("each glossary entry must be a mapping")
        try:
            entry_id = str(item["id"])
            meaning = str(item["meaning"])
        except KeyError as exc:
            raise GlossaryError(f"glossary entry missing {exc.args[0]}") from exc
        if entry_id in seen:
            raise GlossaryError(f"duplicate glossary id: {entry_id}")
        seen.add(entry_id)
        entries.append(
            GlossaryEntry(
                id=entry_id,
                kind=str(item.get("kind", "entity")),
                terms_en=tuple(str(t) for t in item.get("terms_en") or []),
                terms_ar=tuple(str(t) for t in item.get("terms_ar") or []),
                meaning=meaning,
            )
        )
    notes = tuple(str(n) for n in raw.get("notes") or [])
    return Glossary(notes, tuple(entries))


def load_glossary(path: str | Path | None = None) -> Glossary:
    try:
        if path is None:
            text = (
                resources.files("analystagent.semantic")
                .joinpath("resources/glossary.yaml")
                .read_text(encoding="utf-8")
            )
        else:
            text = Path(path).read_text(encoding="utf-8")
        return _parse(yaml.safe_load(text))
    except (OSError, yaml.YAMLError) as exc:
        raise GlossaryError(f"could not load glossary: {exc}") from exc
