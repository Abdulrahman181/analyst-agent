from __future__ import annotations

import re

_DIGIT_MAP = {
    **{ord("٠") + i: ord("0") + i for i in range(10)},
    **{ord("۰") + i: ord("0") + i for i in range(10)},
    ord("٫"): ord("."),
    ord("٬"): ord(","),
    ord("،"): ord(","),
}

_DIACRITICS = re.compile("[ً-ٰٟـ]")
_ARABIC_LETTER = re.compile("[؀-ۿ]")
_LATIN_LETTER = re.compile("[A-Za-z]")


def normalize_digits(text: str) -> str:
    return text.translate(_DIGIT_MAP)


def normalize_arabic(text: str) -> str:
    cleaned = _DIACRITICS.sub("", text)
    cleaned = re.sub("[أإآٱ]", "ا", cleaned)
    cleaned = cleaned.replace("ى", "ي").replace("ة", "ه")
    return cleaned.lower()


def detect_language(text: str) -> str:
    arabic = len(_ARABIC_LETTER.findall(text))
    latin = len(_LATIN_LETTER.findall(text))
    if arabic == 0 and latin == 0:
        return "en"
    return "ar" if arabic >= latin else "en"
